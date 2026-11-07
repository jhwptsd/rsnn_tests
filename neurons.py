"""
Interchangeable spiking-neuron populations (PyTorch).

Every class is a *population*: one object holds the state of ``n`` neurons as
tensors and advances all of them in a single vectorised step. There are no
per-neuron Python objects, so cost scales with tensor kernels, not interpreter
overhead, and the same code runs on CPU or GPU.

Common interface (what a network class should rely on)
------------------------------------------------------
    neuron = make_neuron("lif", n=128, dt=1.0, ...)   # or LIF(128, ...)
    neuron.reset_state()                    # forget state (new sequence/batch)
    spikes = neuron(current)                # one step; current: (n,) or (B, n)
    spikes = neuron.sequence(currents)      # (T, B, n) or (T, n) -> same shape
    neuron.detach_state()                   # cut the graph (truncated BPTT)
    neuron.membrane                         # last membrane variable (or None)
    neuron.n                                # population size

Spikes are returned as floats (0./1.) with the dtype/device of the input.

Units
-----
Time is in milliseconds (``dt``, ``tau_*``, ``refractory``); rates in Hz.
IF / LIF / ALIF voltages are *relative to rest*: rest = 0, the default
threshold is 1, and the input is a per-step charge (``v <- beta*v + R*I``), the
same convention as snnTorch, so these three are drop-in interchangeable.
``Izhikevich`` uses the model's native mV scale and current units.

Models
------
    IFNeuron            non-leaky integrate-and-fire
    LIF                 leaky integrate-and-fire (optionally learnable tau)
    AdaptiveLIF         LIF with an adaptive threshold (spike-frequency
                        adaptation; long memory for recurrent nets)
    Izhikevich          two-variable model; presets for RS/IB/CH/FS/LTS/TC/RZ
    StochasticLIF       LIF with escape noise (probabilistic spiking)
    PoissonNeuron       input is a rate in Hz; spikes are Bernoulli draws

Any model also accepts ``noise_std`` (diffusive Gaussian input noise,
Euler-Maruyama scaled by sqrt(dt)), making every deterministic model stochastic.

Training
--------
Deterministic spikes use a fast-sigmoid surrogate gradient in the backward
pass. Stochastic spikes use a straight-through estimator through the spike
probability. The reset is detached from the graph (standard practice).

Efficiency notes
----------------
* Scalar parameters stay Python floats (no tensor allocation per step);
  per-neuron parameters are buffers and broadcast.
* With autograd disabled (``torch.no_grad()`` / ``inference_mode()``) steps run
  *in place* on the state tensors; with autograd enabled they run out of place
  so BPTT stays correct. Call ``reset_state()`` when switching between modes.
* ``sequence`` writes into one preallocated output tensor.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

__all__ = [
    "SpikingNeuron",
    "ThresholdNeuron",
    "IFNeuron",
    "LIF",
    "AdaptiveLIF",
    "Izhikevich",
    "StochasticLIF",
    "PoissonNeuron",
    "NEURON_TYPES",
    "make_neuron",
]

Param = Union[float, Tensor]
State = Dict[str, Tensor]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FastSigmoidSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid derivative 1/(slope*|x|+1)^2 backward."""

    @staticmethod
    def forward(ctx, x: Tensor, slope: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (x,) = ctx.saved_tensors
        denom = ctx.slope * x.abs() + 1.0
        return grad_out / (denom * denom), None


def spike_fn(x: Tensor, slope: float) -> Tensor:
    """Spike where ``x >= 0``; surrogate gradient only when one is needed."""
    if x.requires_grad and torch.is_grad_enabled():
        return _FastSigmoidSpike.apply(x, slope)
    return (x >= 0).to(x.dtype)


def _register(module: nn.Module, name: str, value: Param) -> None:
    """Tensors become buffers (move with .to(), broadcast per neuron);
    scalars stay Python floats (fastest path, no allocation per step)."""
    if isinstance(value, Tensor):
        module.register_buffer(name, value.detach().clone().to(torch.get_default_dtype()))
    else:
        setattr(module, name, float(value))


def _is_one(x: Param) -> bool:
    return isinstance(x, float) and x == 1.0


def _mul(x: Tensor, c: Param) -> Tensor:
    return x if _is_one(c) else x * c


def _axpy_(dst: Tensor, src: Tensor, coef: Param) -> Tensor:
    """dst += coef * src, in place, for float or per-neuron-tensor coef."""
    if isinstance(coef, Tensor):
        return dst.addcmul_(coef, src)
    return dst.add_(src, alpha=coef)


def _fill(v: Tensor, mask: Tensor, value: Param, inplace: bool) -> Tensor:
    """Set v[mask] = value (float or per-neuron tensor)."""
    if inplace:
        if isinstance(value, Tensor):
            return v.copy_(torch.where(mask, value, v))
        return v.masked_fill_(mask, value)
    return torch.where(mask, value, v)


# --------------------------------------------------------------------------- #
# Base class: the polymorphic interface
# --------------------------------------------------------------------------- #
class SpikingNeuron(nn.Module, ABC):
    """Abstract population of ``n`` spiking neurons.

    Subclasses implement ``_init_state`` and ``_step``; everything a network
    needs (state handling, noise, sequences, truncated-BPTT support) lives here.
    """

    def __init__(
        self,
        n: int,
        dt: float = 1.0,
        noise_std: float = 0.0,
        surrogate_slope: float = 25.0,
    ):
        super().__init__()
        self.n = int(n)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.surrogate_slope = float(surrogate_slope)
        self._state: Optional[State] = None
        self._keep_pre = False  # record pre-reset voltage (used by sequence())

    # ---- to implement --------------------------------------------------- #
    @abstractmethod
    def _init_state(self, like: Tensor) -> State:
        """Fresh state matching ``like`` (shape/dtype/device). Must contain 'v'."""

    @abstractmethod
    def _step(self, current: Tensor, state: State) -> Tuple[Tensor, State]:
        """Advance one time step. Returns (spikes, new_state)."""

    # ---- shared behaviour ----------------------------------------------- #
    def reset_state(self) -> None:
        self._state = None

    def detach_state(self) -> None:
        if self._state is not None:
            self._state = {k: t.detach() for k, t in self._state.items()}

    @property
    def membrane(self) -> Optional[Tensor]:
        if self._state is None:
            return None
        return self._state.get("v_pre", self._state["v"])

    def forward(self, current: Tensor) -> Tensor:
        st = self._state
        if st is not None:
            v = st["v"]
            if v.shape != current.shape or v.dtype != current.dtype or v.device != current.device:
                st = None
        if st is None:
            st = self._init_state(current)
        if self.noise_std > 0.0:
            current = current + torch.randn_like(current).mul_(self.noise_std * math.sqrt(self.dt))
        spikes, self._state = self._step(current, st)
        return spikes

    def sequence(self, currents: Tensor, record_membrane: bool = False):
        """Run T steps. ``currents``: (T, ..., n). Returns spikes (and the
        pre-reset membrane trace if ``record_membrane``)."""
        if currents.shape[-1] != self.n:
            raise ValueError(f"expected last dim {self.n}, got {tuple(currents.shape)}")
        spikes = torch.empty_like(currents)
        trace = torch.empty_like(currents) if record_membrane else None
        self._keep_pre = record_membrane
        try:
            for t in range(currents.shape[0]):
                spikes[t] = self(currents[t])
                if trace is not None:
                    trace[t] = self.membrane
        finally:
            self._keep_pre = False
        return (spikes, trace) if record_membrane else spikes

    def extra_repr(self) -> str:
        return f"n={self.n}, dt={self.dt}"


# --------------------------------------------------------------------------- #
# Threshold-and-reset family: IF, LIF, ALIF, StochasticLIF
# --------------------------------------------------------------------------- #
class ThresholdNeuron(SpikingNeuron):
    """Shared machinery: integrate -> (refractory) -> fire -> reset.

    Subclasses customise ``_beta`` (leak) and/or ``_fire`` (spike generation).
    """

    def __init__(
        self,
        n: int,
        v_threshold: Param = 1.0,
        v_reset: Param = 0.0,
        reset_mode: str = "hard",
        refractory: float = 0.0,
        resistance: Param = 1.0,
        **kw,
    ):
        super().__init__(n, **kw)
        if reset_mode not in ("hard", "soft"):
            raise ValueError("reset_mode must be 'hard' or 'soft'")
        _register(self, "v_threshold", v_threshold)
        _register(self, "v_reset", v_reset)
        _register(self, "resistance", resistance)
        self.reset_mode = reset_mode
        self.refractory_steps = int(round(refractory / self.dt))

    # ---- hooks ----------------------------------------------------------- #
    def _beta(self) -> Param:
        return 1.0  # no leak

    def _fire(self, v: Tensor, st: State) -> Tensor:
        return spike_fn(v - self.v_threshold, self.surrogate_slope)

    # ---- core ------------------------------------------------------------ #
    def _init_state(self, like: Tensor) -> State:
        st = {"v": torch.zeros_like(like)}
        if self.refractory_steps:
            st["ref"] = torch.zeros_like(like)
        return st

    def _step(self, current: Tensor, st: State) -> Tuple[Tensor, State]:
        inplace = not torch.is_grad_enabled()
        v = st["v"]
        nref = self.refractory_steps

        # integrate: v <- beta * v + R * I
        beta = self._beta()
        if inplace:
            if not _is_one(beta):
                v.mul_(beta)
            _axpy_(v, current, self.resistance)
        else:
            v = _mul(v, beta) + _mul(current, self.resistance)

        # refractory: clamp membrane to reset value while the counter is active
        if nref:
            ref = st["ref"]
            in_ref = ref > 0
            v = _fill(v, in_ref, self.v_reset, inplace)

        if self._keep_pre:
            st["v_pre"] = v.clone() if inplace else v
        else:
            st.pop("v_pre", None)

        # fire
        s = self._fire(v, st)
        if nref:
            s = s * (~in_ref)
        m = s.detach() > 0.5  # reset is detached from the graph

        # reset
        if self.reset_mode == "hard":
            v = _fill(v, m, self.v_reset, inplace)
        elif inplace:
            v.sub_(m.to(v.dtype) * self.v_threshold)
        else:
            v = v - m.to(v.dtype) * self.v_threshold

        if nref:
            ref.sub_(1.0).clamp_(min=0.0)
            ref.masked_fill_(m, float(nref))

        st["v"] = v
        return s, st


class IFNeuron(ThresholdNeuron):
    """Non-leaky integrate-and-fire: v <- v + R*I; spike at threshold."""


class LIF(ThresholdNeuron):
    """Leaky integrate-and-fire: v <- exp(-dt/tau_m) * v + R*I.

    ``tau_m`` may be a float or a per-neuron tensor. With ``learnable_tau`` the
    time constants become trainable (stored as log tau, per neuron).
    """

    def __init__(self, n: int, tau_m: Param = 20.0, learnable_tau: bool = False, **kw):
        super().__init__(n, **kw)
        self.learnable_tau = learnable_tau
        if learnable_tau:
            tau = torch.as_tensor(tau_m, dtype=torch.get_default_dtype()).expand(self.n)
            self.log_tau = nn.Parameter(tau.clone().log())
        else:
            if isinstance(tau_m, Tensor):
                beta: Param = torch.exp(-self.dt / tau_m)
            else:
                beta = math.exp(-self.dt / float(tau_m))
            _register(self, "_beta_const", beta)

    def _beta(self) -> Param:
        if self.learnable_tau:
            return torch.exp(-self.dt / torch.exp(self.log_tau))
        return self._beta_const


class AdaptiveLIF(LIF):
    """LIF with adaptive threshold: theta_eff = theta + strength * a,
    a <- exp(-dt/tau_adapt) * a + spike. Slow adaptation gives recurrent nets
    long memory and richer oscillatory dynamics."""

    def __init__(self, n: int, tau_adapt: float = 200.0, adapt_strength: Param = 0.2, **kw):
        super().__init__(n, **kw)
        self._rho = math.exp(-self.dt / float(tau_adapt))
        _register(self, "adapt_strength", adapt_strength)

    def _init_state(self, like: Tensor) -> State:
        st = super()._init_state(like)
        st["a"] = torch.zeros_like(like)
        return st

    def _fire(self, v: Tensor, st: State) -> Tensor:
        theta = self.v_threshold + self.adapt_strength * st["a"]
        s = spike_fn(v - theta, self.surrogate_slope)
        st["a"] = st["a"] * self._rho + s
        return s


class StochasticLIF(LIF):
    """LIF with escape noise: spikes are Bernoulli draws from a probability
    that increases with the membrane potential.

    escape='sigmoid': p = sigmoid((v - theta) / delta)
    escape='exp'    : rate = rho0 * exp((v - theta) / delta) Hz,
                      p = 1 - exp(-rate * dt)
    ``delta`` (same units as v) sets the softness; delta -> 0 recovers LIF.
    Gradients use a straight-through estimator through p.
    """

    def __init__(
        self,
        n: int,
        delta: Param = 0.1,
        escape: str = "sigmoid",
        rho0: float = 10.0,
        **kw,
    ):
        super().__init__(n, **kw)
        if escape not in ("sigmoid", "exp"):
            raise ValueError("escape must be 'sigmoid' or 'exp'")
        self.escape = escape
        self.rho0 = float(rho0)
        _register(self, "delta", delta)

    def _fire(self, v: Tensor, st: State) -> Tensor:
        x = (v - self.v_threshold) / self.delta
        if self.escape == "sigmoid":
            p = torch.sigmoid(x)
        else:
            rate = self.rho0 * torch.exp(x.clamp(max=30.0))
            p = -torch.expm1(rate * (-self.dt * 1e-3))
        bern = (torch.rand_like(p) < p).to(p.dtype)
        if p.requires_grad and torch.is_grad_enabled():
            return bern + p - p.detach()  # value = bern, gradient = dp
        return bern


# --------------------------------------------------------------------------- #
# Izhikevich
# --------------------------------------------------------------------------- #
class Izhikevich(SpikingNeuron):
    """Izhikevich (2003) model, native mV / ms units:

        dv/dt = 0.04 v^2 + 5 v + 140 - u + I
        du/dt = a (b v - u)
        if v >= v_peak:  v <- c,  u <- u + d

    ``a, b, c, d`` may be floats or per-neuron tensors (heterogeneous
    populations). ``substeps`` splits each ``dt`` into smaller Euler steps for
    v (Izhikevich's original code uses 2 x 0.5 ms for dt = 1 ms).
    """

    PRESETS = {
        "RS": (0.02, 0.20, -65.0, 8.00),    # regular spiking
        "IB": (0.02, 0.20, -55.0, 4.00),    # intrinsically bursting
        "CH": (0.02, 0.20, -50.0, 2.00),    # chattering
        "FS": (0.10, 0.20, -65.0, 2.00),    # fast spiking
        "LTS": (0.02, 0.25, -65.0, 2.00),   # low-threshold spiking
        "TC": (0.02, 0.25, -65.0, 0.05),    # thalamo-cortical
        "RZ": (0.10, 0.26, -65.0, 2.00),    # resonator
    }
    V_CLIP = 1.0e3  # guards against float overflow under extreme drive

    def __init__(
        self,
        n: int,
        a: Param = 0.02,
        b: Param = 0.20,
        c: Param = -65.0,
        d: Param = 8.0,
        v_peak: float = 30.0,
        substeps: int = 2,
        surrogate_slope: float = 0.5,  # voltage scale is mV, not ~1
        **kw,
    ):
        super().__init__(n, surrogate_slope=surrogate_slope, **kw)
        for name, val in (("a", a), ("b", b), ("c", c), ("d", d)):
            _register(self, name, val)
        self.v_peak = float(v_peak)
        self.substeps = int(substeps)

    @classmethod
    def from_preset(cls, name: str, n: int, **kw) -> "Izhikevich":
        a, b, c, d = cls.PRESETS[name]
        return cls(n, a=a, b=b, c=c, d=d, **kw)

    @classmethod
    def cortical_population(
        cls, n_exc: int, n_inh: int, seed: Optional[int] = None, **kw
    ) -> "Izhikevich":
        """Heterogeneous E/I population from Izhikevich (2003); excitatory
        neurons come first."""
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        re = torch.rand(n_exc, generator=g)
        ri = torch.rand(n_inh, generator=g)
        a = torch.cat([torch.full((n_exc,), 0.02), 0.02 + 0.08 * ri])
        b = torch.cat([torch.full((n_exc,), 0.20), 0.25 - 0.05 * ri])
        c = torch.cat([-65.0 + 15.0 * re ** 2, torch.full((n_inh,), -65.0)])
        d = torch.cat([8.0 - 6.0 * re ** 2, torch.full((n_inh,), 2.0)])
        return cls(n_exc + n_inh, a=a, b=b, c=c, d=d, **kw)

    def _init_state(self, like: Tensor) -> State:
        v = torch.zeros_like(like).add_(self.c)
        u = v * self.b
        return {"v": v, "u": u}

    def _step(self, current: Tensor, st: State) -> Tuple[Tensor, State]:
        inplace = not torch.is_grad_enabled()
        v, u = st["v"], st["u"]
        h = self.dt / self.substeps

        if inplace:
            for _ in range(self.substeps):
                dv = torch.mul(v, 0.04).add_(5.0).mul_(v).add_(140.0).sub_(u).add_(current)
                v.add_(dv, alpha=h).clamp_(max=self.V_CLIP)
            du = torch.mul(v, self.b).sub_(u)
            _axpy_(u, du, self.a * self.dt)
        else:
            for _ in range(self.substeps):
                v = (v + h * (v * (0.04 * v + 5.0) + 140.0 - u + current)).clamp(max=self.V_CLIP)
            u = u + (self.a * self.dt) * (v * self.b - u)

        if self._keep_pre:
            st["v_pre"] = (v.clone() if inplace else v).clamp(max=self.v_peak)
        else:
            st.pop("v_pre", None)

        s = spike_fn(v - self.v_peak, self.surrogate_slope)
        m = s.detach() > 0.5

        if inplace:
            _fill(v, m, self.c, True)
            u.add_(m.to(u.dtype) * self.d)
        else:
            v = torch.where(m, self.c, v)
            u = u + m.to(u.dtype) * self.d

        st["v"], st["u"] = v, u
        return s, st


# --------------------------------------------------------------------------- #
# Poisson
# --------------------------------------------------------------------------- #
class PoissonNeuron(SpikingNeuron):
    """Stochastic rate-driven neurons: the input is the firing rate in Hz and
    each step spikes with probability 1 - exp(-rate * dt). Stateless apart from
    ``membrane`` (the last spike probability). Straight-through gradient."""

    def _init_state(self, like: Tensor) -> State:
        return {"v": torch.zeros_like(like)}

    def _step(self, rate: Tensor, st: State) -> Tuple[Tensor, State]:
        p = -torch.expm1(rate.clamp(min=0.0) * (-self.dt * 1e-3))
        bern = (torch.rand_like(p) < p).to(p.dtype)
        s = bern + p - p.detach() if (p.requires_grad and torch.is_grad_enabled()) else bern
        st["v"] = p
        return s, st


# --------------------------------------------------------------------------- #
# Factory (lets a network class pick the neuron model by name)
# --------------------------------------------------------------------------- #
NEURON_TYPES = {
    "if": IFNeuron,
    "lif": LIF,
    "alif": AdaptiveLIF,
    "izhikevich": Izhikevich,
    "stochastic_lif": StochasticLIF,
    "poisson": PoissonNeuron,
}


def make_neuron(kind: str, n: int, **kwargs) -> SpikingNeuron:
    try:
        cls = NEURON_TYPES[kind.lower()]
    except KeyError:
        raise ValueError(f"unknown neuron type {kind!r}; choose from {sorted(NEURON_TYPES)}")
    return cls(n, **kwargs)


# --------------------------------------------------------------------------- #
# Self-test:  python neurons.py
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    torch.manual_seed(0)
    T, B, N = 200, 4, 16

    def drive(kind: str) -> Tensor:
        if kind == "izhikevich":
            return 10.0 + 2.0 * torch.randn(T, B, N)
        if kind == "poisson":
            return 40.0 * torch.rand(T, B, N)  # Hz
        return 0.12 + 0.05 * torch.randn(T, B, N)

    kwargs = {
        "if": dict(refractory=2.0),
        "lif": dict(tau_m=20.0, refractory=2.0),
        "alif": dict(tau_m=20.0),
        "izhikevich": dict(),
        "stochastic_lif": dict(tau_m=20.0, delta=0.1),
        "poisson": dict(),
    }

    for kind, kw in kwargs.items():
        neuron = make_neuron(kind, N, **kw)
        x = drive(kind)

        # 1. shapes + substitutability through the common interface
        with torch.no_grad():
            neuron.reset_state()
            torch.manual_seed(1)
            fast = neuron.sequence(x)
        assert fast.shape == x.shape and set(fast.unique().tolist()) <= {0.0, 1.0}, kind

        # 2. in-place (no_grad) and out-of-place (autograd) paths agree
        neuron.reset_state()
        torch.manual_seed(1)
        slow = neuron.sequence(x)
        assert torch.equal(fast, slow), f"{kind}: inplace/out-of-place mismatch"

        # 3. gradients flow back to the input
        neuron.reset_state()
        xr = x.clone().requires_grad_(True)
        neuron.sequence(xr).sum().backward()
        assert xr.grad is not None and torch.isfinite(xr.grad).all(), kind

        # 4. membrane recording + state handling
        neuron.reset_state()
        with torch.no_grad():
            _, trace = neuron.sequence(x, record_membrane=True)
        assert trace.shape == x.shape
        neuron.detach_state()

        print(f"{kind:15s} ok   mean rate = {fast.mean().item() * 1000 / neuron.dt:7.1f} Hz")

    # learnable time constants receive gradients
    lif = LIF(N, tau_m=20.0, learnable_tau=True)
    lif.sequence(drive("lif")).sum().backward()
    assert lif.log_tau.grad is not None and torch.isfinite(lif.log_tau.grad).all()

    # heterogeneous population + presets construct and run
    pop = Izhikevich.cortical_population(12, 4, seed=0)
    with torch.no_grad():
        assert pop.sequence(10.0 + torch.randn(50, 16)).shape == (50, 16)
    Izhikevich.from_preset("FS", 8)

    print("all self-tests passed")


if __name__ == "__main__":
    _selftest()

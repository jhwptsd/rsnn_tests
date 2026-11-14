"""
Recurrent spiking neural network (RSNN) with excitatory/inhibitory populations
and a family of interchangeable learning rules.

Network
-------
``RSNN`` builds a sparse recurrent network of ``n_rec`` neurons, a fraction
``exc_fraction`` of them excitatory (indices ``[0, n_exc)``) and the rest
inhibitory. Neuron models come from ``neurons.py`` and can be swapped by name,
class, factory or pre-built instance; excitatory and inhibitory neurons may use
different models. Size, sparsity (a single density or a separate one for each of
EE/EI/IE/II), Dale's law, spectral radius, synaptic filtering and readout are
all configurable.

    net = RSNN(n_in=20, n_rec=200, n_out=3, exc_fraction=0.8, connectivity=0.1,
               neuron="alif", inh_neuron="lif")
    y = net(x)                                  # x: (T, B, n_in) -> (T, B, n_out)
    y, rec = net(x, record_membrane=True)       # rec: spikes, membrane, ...

Learning rules (all operate on the same network)
------------------------------------------------
    SurrogateBPTT      backpropagation through time with surrogate gradients
                       (supervised; accepts a fixed *or learnable* loss module)
    STDP               unsupervised pair-based STDP (traces, soft/hard bounds)
    RSTDP              reward-modulated STDP: STDP eligibility x (reward - baseline)
    ThreeFactor        eligibility x post-synaptic pseudo-derivative x learning
                       signal; with the default settings this *is* e-prop
                       (random or symmetric feedback), and any externally
                       supplied modulator turns it into a generic 3-factor rule
    InhibitorySTDP     Vogels et al. 2011 rule for I->E synapses (keeps
                       excitatory neurons near a target rate)
    HomeostaticScaling multiplicative synaptic scaling toward a target rate

Local rules are applied to a *recording* of a chunk (``record=True``), and
chunks can be as short as you like, so updates can be as online as you need.
Rules can be combined, e.g. ThreeFactor for the task plus InhibitorySTDP and
HomeostaticScaling to keep the network in a stable, selective active regime.

Conventions
-----------
Weight matrices are (post, pre). Recurrent spikes arrive with a one-step delay.
With ``dale=True`` the recurrent parameter stores non-negative *magnitudes*;
the effective weight is ``sign(pre) * relu(w) * mask`` so excitatory neurons
never become inhibitory. Fixed sparsity masks are never modified by learning.
Time is in ms (see ``neurons.py``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neurons import SpikingNeuron, make_neuron

__all__ = [
    "RSNN",
    "Recording",
    "Learner",
    "SurrogateBPTT",
    "STDP",
    "RSTDP",
    "ThreeFactor",
    "InhibitorySTDP",
    "HomeostaticScaling",
    "mse_error",
    "softmax_error",
]

NeuronSpec = Union[str, type, Callable[..., SpikingNeuron], SpikingNeuron]


# --------------------------------------------------------------------------- #
# Time-filter helpers (loops over T on small (B, N) tensors; the heavy
# synapse-sized work is always a single matmul over all T*B samples)
# --------------------------------------------------------------------------- #
def _causal_filter(z: Tensor, decay: float) -> Tensor:
    """out[t] = decay * out[t-1] + z[t]."""
    if decay == 0.0:
        return z
    out = torch.empty_like(z)
    acc = torch.zeros_like(z[0])
    for t in range(z.shape[0]):
        acc = acc * decay + z[t]
        out[t] = acc
    return out


def _anticausal_filter(z: Tensor, decay: float) -> Tensor:
    """out[t] = z[t] + decay * out[t+1] (sum over the future)."""
    if decay == 0.0:
        return z
    out = torch.empty_like(z)
    acc = torch.zeros_like(z[-1])
    for t in range(z.shape[0] - 1, -1, -1):
        acc = acc * decay + z[t]
        out[t] = acc
    return out


def _flat(z: Tensor) -> Tensor:
    return z.reshape(-1, z.shape[-1])


def _signed_magnitude_update(w: Tensor, delta: Tensor, w_max: Optional[float]) -> None:
    """Change |w| by delta without ever flipping the sign of w."""
    mag = (w.abs() + delta).clamp_(min=0.0)
    if w_max is not None:
        mag.clamp_(max=w_max)
    w.copy_(torch.sign(w) * mag)


# --------------------------------------------------------------------------- #
# Recording of a forward pass (input to the local learning rules)
# --------------------------------------------------------------------------- #
@dataclass
class Recording:
    x: Tensor                  # (T, B, n_in)  inputs
    s_prev: Tensor             # (T, B, N)     recurrent pre-synaptic spikes used at each step
    spikes: Tensor             # (T, B, N)     post-synaptic spikes
    y: Tensor                  # (T, B, n_out) readout
    v: Optional[Tensor] = None  # (T, B, N)    pre-reset membrane (needs record_membrane)

    def rate_hz(self, dt: float) -> Tensor:
        """Per-neuron mean firing rate in Hz."""
        return self.spikes.mean(dim=(0, 1)) * (1000.0 / dt)


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class RSNN(nn.Module):
    """Sparse recurrent E/I spiking network with a leaky linear readout.

    Parameters
    ----------
    n_in, n_rec, n_out : sizes of input, recurrent population and readout.
    exc_fraction       : fraction of recurrent neurons that are excitatory.
    connectivity       : recurrent connection probability. A float, or a dict
                         with keys "EE", "EI", "IE", "II" (post type first).
    input_connectivity : probability of each input->recurrent connection.
    neuron, neuron_kwargs : model for the recurrent neurons ("lif", "if",
                         "alif", "izhikevich", "stochastic_lif", "poisson", a
                         SpikingNeuron subclass, a factory ``f(n, dt=...)``, or
                         a pre-built population of size n_rec / n_exc).
    inh_neuron, inh_neuron_kwargs : if given, inhibitory neurons use this model
                         (a separate population); otherwise one population is
                         used for everything.
    dt                 : time step in ms (must match any pre-built neurons).
    dale               : enforce Dale's law (E columns >= 0, I columns <= 0).
    inh_gain           : initial inhibitory/excitatory weight ratio.
    input_scale, rec_scale : initial weight scales. They are in the units the
                         neuron expects: per-step charge for IF/LIF/ALIF,
                         current for Izhikevich (use much larger values there).
    spectral_radius    : if set, rescale initial recurrent weights to it.
    w_max              : optional cap on recurrent weight magnitude.
    tau_syn            : optional synaptic current time constant (ms).
    tau_out            : readout leak time constant (ms); None/0 = no leak.
    readout_from       : "all" or "exc" neurons feed the readout.
    train_input / train_recurrent / train_readout : which weights learn.
    seed               : seed for reproducible construction.
    """

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        *,
        exc_fraction: float = 0.8,
        connectivity: Union[float, Dict[str, float]] = 0.1,
        input_connectivity: float = 1.0,
        neuron: NeuronSpec = "lif",
        neuron_kwargs: Optional[dict] = None,
        inh_neuron: Optional[NeuronSpec] = None,
        inh_neuron_kwargs: Optional[dict] = None,
        dt: float = 1.0,
        dale: bool = True,
        inh_gain: float = 2.0,
        input_scale: float = 1.0,
        rec_scale: float = 1.0,
        spectral_radius: Optional[float] = None,
        w_max: Optional[float] = None,
        tau_syn: Optional[float] = None,
        tau_out: Optional[float] = 20.0,
        readout_from: str = "all",
        readout_scale: float = 1.0,
        train_input: bool = True,
        train_recurrent: bool = True,
        train_readout: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if not 0.0 < exc_fraction <= 1.0:
            raise ValueError("exc_fraction must be in (0, 1]")
        if readout_from not in ("all", "exc"):
            raise ValueError("readout_from must be 'all' or 'exc'")
        self.n_in, self.n_rec, self.n_out = int(n_in), int(n_rec), int(n_out)
        self.dt = float(dt)
        self.dale = bool(dale)
        self.w_max = w_max
        self.train_input, self.train_recurrent, self.train_readout = (
            train_input, train_recurrent, train_readout)
        N = self.n_rec
        self.n_exc = min(N, max(1, int(round(exc_fraction * N))))
        self.n_inh = N - self.n_exc
        if inh_neuron is not None and self.n_inh == 0:
            raise ValueError("inh_neuron given but exc_fraction leaves no inhibitory neurons")

        gen = torch.Generator()
        gen.manual_seed(seed if seed is not None else int(torch.randint(0, 2**31 - 1, (1,))))

        # ---- neuron populations ---------------------------------------- #
        if inh_neuron is None:
            pops = [self._make_pop(neuron, neuron_kwargs, N)]
            self.slices = [(0, N)]
        else:
            pops = [self._make_pop(neuron, neuron_kwargs, self.n_exc),
                    self._make_pop(inh_neuron, inh_neuron_kwargs, self.n_inh)]
            self.slices = [(0, self.n_exc), (self.n_exc, N)]
        self.pops = nn.ModuleList(pops)

        # ---- cell types ------------------------------------------------- #
        is_exc = torch.zeros(N, dtype=torch.bool)
        is_exc[: self.n_exc] = True
        self.register_buffer("is_exc", is_exc)
        self.register_buffer("sign", torch.where(is_exc, 1.0, -1.0))

        # ---- fixed sparsity masks --------------------------------------- #
        P = self._probability_matrix(connectivity)
        rec_mask = (torch.rand(N, N, generator=gen) < P).float()
        rec_mask.fill_diagonal_(0.0)
        self.register_buffer("rec_mask", rec_mask)
        in_mask = (torch.rand(N, self.n_in, generator=gen) < float(input_connectivity)).float()
        self.register_buffer("in_mask", in_mask)
        out_mask = is_exc.float() if readout_from == "exc" else torch.ones(N)
        self.register_buffer("out_mask", out_mask)

        # ---- weights ----------------------------------------------------- #
        k_in = rec_mask.sum(1, keepdim=True).clamp(min=1.0)       # in-degree per neuron
        mag = torch.randn(N, N, generator=gen).abs() * rec_scale / k_in.sqrt()
        mag[:, self.n_exc:] *= inh_gain
        mag = mag * rec_mask
        if spectral_radius is not None:
            radius = torch.linalg.eigvals(mag * self.sign).abs().max()
            if radius > 0:
                mag = mag * (spectral_radius / radius)
        w_rec = mag if self.dale else mag * self.sign
        self.w_rec = nn.Parameter(w_rec, requires_grad=train_recurrent)

        n_eff = max(1.0, self.n_in * float(input_connectivity))
        w_in = torch.randn(N, self.n_in, generator=gen) * input_scale / math.sqrt(n_eff) * in_mask
        self.w_in = nn.Parameter(w_in, requires_grad=train_input)

        n_ro = max(1.0, float(out_mask.sum()))
        w_out = torch.randn(self.n_out, N, generator=gen) * readout_scale / math.sqrt(n_ro)
        self.w_out = nn.Parameter(w_out, requires_grad=train_readout)
        self.b_out = nn.Parameter(torch.zeros(self.n_out), requires_grad=train_readout)

        # ---- dynamics constants ------------------------------------------ #
        self.syn_decay = math.exp(-self.dt / tau_syn) if tau_syn else None
        self.kappa = math.exp(-self.dt / tau_out) if tau_out else 0.0   # readout leak
        self._state: Optional[Dict[str, Tensor]] = None

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _make_pop(self, spec: NeuronSpec, kwargs: Optional[dict], n: int) -> SpikingNeuron:
        kw = dict(kwargs or {})
        if isinstance(spec, SpikingNeuron):
            if spec.n != n:
                raise ValueError(f"pre-built neuron has n={spec.n}, expected {n}")
            if abs(spec.dt - self.dt) > 1e-12:
                raise ValueError(f"pre-built neuron dt={spec.dt} != network dt={self.dt}")
            return spec
        kw.setdefault("dt", self.dt)
        if isinstance(spec, str):
            return make_neuron(spec, n, **kw)
        if isinstance(spec, type) and issubclass(spec, SpikingNeuron):
            return spec(n, **kw)
        if callable(spec):
            pop = spec(n, **kw)
            if not isinstance(pop, SpikingNeuron):
                raise TypeError("neuron factory must return a SpikingNeuron")
            return pop
        raise TypeError(f"cannot build a neuron population from {spec!r}")

    def _probability_matrix(self, connectivity: Union[float, Dict[str, float]]) -> Tensor:
        if isinstance(connectivity, dict):
            try:
                p = {k: float(connectivity[k]) for k in ("EE", "EI", "IE", "II")}
            except KeyError as e:
                raise ValueError("connectivity dict needs keys EE, EI, IE, II") from e
        else:
            p = {k: float(connectivity) for k in ("EE", "EI", "IE", "II")}
        if not all(0.0 <= v <= 1.0 for v in p.values()):
            raise ValueError("connection probabilities must be in [0, 1]")
        N, e = self.n_rec, self.n_exc
        P = torch.empty(N, N)
        P[:e, :e], P[:e, e:], P[e:, :e], P[e:, e:] = p["EE"], p["EI"], p["IE"], p["II"]  # (post, pre)
        return P

    # ------------------------------------------------------------------ #
    # weights
    # ------------------------------------------------------------------ #
    def effective_weights(self) -> Tuple[Tensor, Tensor, Tensor]:
        """(W_in, W_rec, W_out), differentiable, with masks and Dale's law applied."""
        W_in = self.w_in * self.in_mask
        if self.dale:
            W_rec = torch.relu(self.w_rec) * self.rec_mask * self.sign
        else:
            W_rec = self.w_rec * self.rec_mask
        W_out = self.w_out * self.out_mask
        return W_in, W_rec, W_out

    def rec_magnitude(self) -> Tensor:
        """|W_rec| with the sparsity mask applied."""
        w = torch.relu(self.w_rec) if self.dale else self.w_rec.abs()
        return w * self.rec_mask

    @property
    def density(self) -> float:
        return float(self.rec_mask.sum() / (self.n_rec * (self.n_rec - 1)))

    # --- parameter updates used by the non-gradient learning rules -------- #
    @torch.no_grad()
    def project(self, w_max: Optional[float] = None) -> None:
        """Re-impose Dale's law / weight cap after a direct update."""
        cap = w_max if w_max is not None else self.w_max
        if self.dale:
            self.w_rec.clamp_(min=0.0, max=cap)
        elif cap is not None:
            self.w_rec.clamp_(-cap, cap)

    @torch.no_grad()
    def apply_rec_magnitude_delta(self, delta: Tensor, w_max: Optional[float] = None) -> None:
        """Strengthen (delta > 0) or weaken recurrent synapses irrespective of
        their sign. Masked entries are untouched; signs are preserved."""
        delta = delta * self.rec_mask
        if self.dale:
            self.w_rec.add_(delta)
            self.project(w_max)
        else:
            _signed_magnitude_update(self.w_rec, delta, w_max if w_max is not None else self.w_max)

    @torch.no_grad()
    def apply_rec_effective_delta(self, delta: Tensor) -> None:
        """Add ``delta`` to the *effective* recurrent weights (gradient-style)."""
        delta = delta * self.rec_mask
        self.w_rec.add_(delta * self.sign if self.dale else delta)
        self.project()

    @torch.no_grad()
    def apply_in_magnitude_delta(self, delta: Tensor, w_max: Optional[float] = None) -> None:
        _signed_magnitude_update(self.w_in, delta * self.in_mask, w_max)

    @torch.no_grad()
    def apply_in_effective_delta(self, delta: Tensor) -> None:
        self.w_in.add_(delta * self.in_mask)

    @torch.no_grad()
    def apply_out_delta(self, d_w: Tensor, d_b: Optional[Tensor] = None) -> None:
        self.w_out.add_(d_w * self.out_mask)
        if d_b is not None:
            self.b_out.add_(d_b)

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def reset_state(self) -> None:
        self._state = None
        for p in self.pops:
            p.reset_state()

    def detach_state(self) -> None:
        if self._state is not None:
            self._state = {k: v.detach() for k, v in self._state.items()}
        for p in self.pops:
            p.detach_state()

    def _get_state(self, x: Tensor) -> Dict[str, Tensor]:
        B = x.shape[1]
        st = self._state
        if (st is not None and st["s"].shape == (B, self.n_rec)
                and st["s"].dtype == x.dtype and st["s"].device == x.device):
            return st
        for p in self.pops:
            p.reset_state()
        st = {"s": x.new_zeros(B, self.n_rec), "y": x.new_zeros(B, self.n_out)}
        if self.syn_decay is not None:
            st["i"] = x.new_zeros(B, self.n_rec)
        self._state = st
        return st

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #
    def _step_pops(self, cur: Tensor) -> Tensor:
        if len(self.pops) == 1:
            return self.pops[0](cur)
        return torch.cat([p(cur[:, a:b]) for p, (a, b) in zip(self.pops, self.slices)], dim=1)

    def _membrane(self) -> Tensor:
        if len(self.pops) == 1:
            return self.pops[0].membrane
        return torch.cat([p.membrane for p in self.pops], dim=1)

    def pseudo_derivative(self, v_pre: Tensor) -> Tensor:
        """Per-neuron d(spike)/d(v) for (T, B, N) pre-reset membrane."""
        parts = [p.pseudo_derivative(v_pre[..., a:b]) for p, (a, b) in zip(self.pops, self.slices)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def forward(self, x: Tensor, record: bool = False, record_membrane: bool = False):
        """Run the network over ``x`` of shape (T, B, n_in); state carries over
        between calls until ``reset_state()``. Returns the readout (T, B, n_out),
        plus a ``Recording`` if ``record`` or ``record_membrane``."""
        if x.dim() != 3 or x.shape[-1] != self.n_in:
            raise ValueError(f"expected input (T, B, {self.n_in}), got {tuple(x.shape)}")
        T, B, _ = x.shape
        N = self.n_rec
        st = self._get_state(x)
        W_in, W_rec, W_out = self.effective_weights()
        W_rec_t = W_rec.t()

        drive = torch.matmul(x, W_in.t())               # input current for all T in one GEMM
        spikes = x.new_empty(T, B, N)
        v_hist = x.new_empty(T, B, N) if record_membrane else None
        s0 = s = st["s"]
        i_syn = st.get("i")
        if record_membrane:
            for p in self.pops:
                p.record_pre_reset = True
        try:
            for t in range(T):
                cur = drive[t] + torch.matmul(s, W_rec_t)
                if i_syn is not None:
                    i_syn = i_syn * self.syn_decay + cur
                    cur = i_syn
                s = self._step_pops(cur)
                spikes[t] = s
                if v_hist is not None:
                    v_hist[t] = self._membrane()
        finally:
            if record_membrane:
                for p in self.pops:
                    p.record_pre_reset = False

        ro = torch.matmul(spikes, W_out.t()) + self.b_out   # readout for all T in one GEMM
        if self.kappa == 0.0:
            y, y_last = ro, ro[-1]
        else:
            y = torch.empty_like(ro)
            y_last = st["y"]
            for t in range(T):
                y_last = y_last * self.kappa + ro[t]
                y[t] = y_last

        st["s"], st["y"] = s, y_last
        if i_syn is not None:
            st["i"] = i_syn

        if not (record or record_membrane):
            return y
        s_prev = torch.cat([s0.unsqueeze(0), spikes[:-1]], dim=0)
        return y, Recording(x=x, s_prev=s_prev, spikes=spikes, y=y, v=v_hist)

    def extra_repr(self) -> str:
        return (f"in={self.n_in}, rec={self.n_rec} (E={self.n_exc}, I={self.n_inh}), "
                f"out={self.n_out}, density={self.density:.3f}, dale={self.dale}")


# --------------------------------------------------------------------------- #
# Error signals for the supervised, local rules (dL/dy, per time step)
# --------------------------------------------------------------------------- #
def mse_error(y: Tensor, target: Tensor) -> Tensor:
    """dL/dy for L = 0.5 * sum (y - target)^2 (the learners divide by batch)."""
    return y - target


def softmax_error(y: Tensor, labels: Tensor) -> Tensor:
    """dL/dy for per-time-step cross-entropy. ``labels``: (B,) or (T, B) class ids."""
    T, B, K = y.shape
    if labels.dim() == 1:
        labels = labels.unsqueeze(0).expand(T, B)
    return torch.softmax(y, dim=-1) - F.one_hot(labels, K).to(y.dtype)


# --------------------------------------------------------------------------- #
# Learning rules
# --------------------------------------------------------------------------- #
class Learner(ABC):
    """Common base: a learning rule bound to one network."""

    def __init__(self, model: RSNN):
        self.model = model

    @abstractmethod
    def update(self, *args, **kwargs):
        """Apply one weight update."""


# ---- 1. surrogate gradients + backpropagation through time ----------------- #
class SurrogateBPTT(Learner):
    """Supervised training by BPTT; spikes use the neurons' surrogate gradients.

    ``loss_fn(y, target) -> scalar``. It may be an ``nn.Module`` with its own
    parameters (a *learnable objective*); those are optimised jointly. Optional
    firing-rate regularisation keeps the network out of silence / runaway
    synchrony, and ``tbptt`` truncates the backward pass to chunks of that many
    steps. Gradients are clipped to ``grad_clip``.
    """

    def __init__(
        self,
        model: RSNN,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        tbptt: Optional[int] = None,
        grad_clip: Optional[float] = 1.0,
        rate_target_hz: Optional[float] = None,
        rate_weight: float = 0.0,
        input_noise: float = 0.0,
    ):
        super().__init__(model)
        self.loss_fn = loss_fn if loss_fn is not None else F.mse_loss
        params = [p for p in model.parameters() if p.requires_grad]
        if isinstance(self.loss_fn, nn.Module):
            params += list(self.loss_fn.parameters())
        self.params = params
        self.opt = optimizer if optimizer is not None else torch.optim.Adam(params, lr=lr)
        self.tbptt = tbptt
        self.grad_clip = grad_clip
        self.rate_target_hz = rate_target_hz
        self.rate_weight = rate_weight
        self.input_noise = input_noise

    def update(self, x: Tensor, target: Tensor, carry_state: bool = False) -> float:
        """One optimisation pass over a (T, B, ...) sequence. Returns mean loss."""
        m = self.model
        m.train()
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.train()
        if not carry_state:
            m.reset_state()
        T = x.shape[0]
        K = self.tbptt or T
        use_rate = self.rate_weight > 0.0 and self.rate_target_hz is not None
        total, chunks = 0.0, 0
        for a in range(0, T, K):
            xc, tc = x[a:a + K], target[a:a + K]
            if self.input_noise > 0.0:
                xc = xc + self.input_noise * torch.randn_like(xc)
            out = m(xc, record=use_rate)
            y, rec = out if use_rate else (out, None)
            loss = self.loss_fn(y, tc)
            if use_rate:
                p_target = self.rate_target_hz * m.dt * 1e-3
                loss = loss + self.rate_weight * ((rec.spikes.mean(dim=(0, 1)) - p_target) ** 2).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
            self.opt.step()
            m.project()
            m.detach_state()
            total += float(loss.detach())
            chunks += 1
        return total / max(chunks, 1)


# ---- 2. pair-based STDP ---------------------------------------------------- #
class STDP(Learner):
    """Unsupervised pair-based STDP with exponential traces.

    LTP: a post-synaptic spike potentiates synapses from recently active
    pre-synaptic neurons; LTD: a pre-synaptic spike depresses synapses from
    recently active post-synaptic neurons. Changes act on weight *magnitude*
    (so excitatory and inhibitory synapses are both strengthened by LTP) and are
    summed over the chunk, then applied at its end. ``mu`` > 0 gives soft bounds
    (LTP ~ (w_max - w)^mu, LTD ~ w^mu; needs ``w_max``).

    targets: "rec" and/or "in" (input weights; input values act as pre-spikes).
    pre_types: restrict plasticity to "all", "exc" or "inh" pre-synaptic cells.
    """

    def __init__(
        self,
        model: RSNN,
        a_plus: float = 0.01,
        a_minus: float = 0.0105,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        lr: float = 1.0,
        w_max: Optional[float] = None,
        mu: float = 0.0,
        targets: Sequence[str] = ("rec",),
        pre_types: str = "all",
    ):
        super().__init__(model)
        if mu > 0.0 and w_max is None:
            raise ValueError("soft bounds (mu > 0) need w_max")
        if pre_types not in ("all", "exc", "inh"):
            raise ValueError("pre_types must be 'all', 'exc' or 'inh'")
        self.a_plus, self.a_minus, self.lr = a_plus, a_minus, lr
        self.d_plus = math.exp(-model.dt / tau_plus)
        self.d_minus = math.exp(-model.dt / tau_minus)
        self.w_max, self.mu = w_max, mu
        self.targets, self.pre_types = tuple(targets), pre_types

    # -- shared by RSTDP ---------------------------------------------------- #
    def _pair_terms(self, pre: Tensor, post: Tensor, post_w: Optional[Tensor] = None
                    ) -> Tuple[Tensor, Tensor]:
        """Return (P, D), each (n_post, n_pre): P = sum post(t) * pre_trace(t),
        D = sum post_trace(t-1) * pre(t); averaged over the batch."""
        B = post.shape[1]
        pre_tr = _causal_filter(pre, self.d_plus)
        post_tr = _causal_filter(post, self.d_minus)
        post_before = torch.zeros_like(post_tr)
        post_before[1:] = post_tr[:-1]
        post_ltp, post_ltd = post, post_before
        if post_w is not None:
            post_ltp, post_ltd = post * post_w, post_before * post_w
        P = _flat(post_ltp).t() @ _flat(pre_tr) / B
        D = _flat(post_ltd).t() @ _flat(pre) / B
        return P, D

    def _delta(self, P: Tensor, D: Tensor, w_mag: Tensor) -> Tensor:
        if self.mu == 0.0:
            ltp, ltd = P, D
        else:
            ltp = P * (self.w_max - w_mag).clamp(min=0.0).pow(self.mu)
            ltd = D * w_mag.pow(self.mu)
        return self.lr * (self.a_plus * ltp - self.a_minus * ltd)

    def _pre_filter(self) -> Optional[Tensor]:
        m = self.model
        if self.pre_types == "all":
            return None
        sel = m.is_exc if self.pre_types == "exc" else ~m.is_exc
        return sel.to(m.rec_mask.dtype).unsqueeze(0)       # (1, N): broadcast over post rows

    @torch.no_grad()
    def update(self, rec: Recording) -> None:
        m = self.model
        if "rec" in self.targets and m.train_recurrent:
            P, D = self._pair_terms(rec.s_prev, rec.spikes)
            delta = self._delta(P, D, m.rec_magnitude())
            sel = self._pre_filter()
            m.apply_rec_magnitude_delta(delta if sel is None else delta * sel, self.w_max)
        if "in" in self.targets and m.train_input:
            P, D = self._pair_terms(rec.x, rec.spikes)
            m.apply_in_magnitude_delta(self._delta(P, D, (m.w_in * m.in_mask).abs()), self.w_max)


# ---- 3. reward-modulated STDP --------------------------------------------- #
class RSTDP(STDP):
    """R-STDP: STDP eligibility, decayed with ``tau_elig`` toward the end of the
    chunk, multiplied by (reward - running baseline). Reward is a scalar or a
    per-sample (B,) tensor delivered after the chunk; positive reward reinforces
    the pairings that occurred, negative reward reverses them."""

    def __init__(self, model: RSNN, tau_elig: float = 200.0, baseline_decay: float = 0.95, **kw):
        super().__init__(model, **kw)
        self.elig_decay = math.exp(-model.dt / tau_elig)
        self.baseline_decay = baseline_decay
        self.baseline = 0.0

    @torch.no_grad()
    def update(self, rec: Recording, reward: Union[float, Tensor]) -> float:
        m = self.model
        T, B, _ = rec.spikes.shape
        r = torch.as_tensor(reward, dtype=rec.spikes.dtype, device=rec.spikes.device)
        r_mean = float(r.mean())
        r_eff = (r - self.baseline).reshape(1, -1, 1) if r.dim() else (r - self.baseline)
        self.baseline = self.baseline_decay * self.baseline + (1.0 - self.baseline_decay) * r_mean
        t_idx = torch.arange(T - 1, -1, -1, device=rec.spikes.device, dtype=rec.spikes.dtype)
        w_t = (self.elig_decay ** t_idx).reshape(T, 1, 1) * r_eff          # decay toward chunk end
        if "rec" in self.targets and m.train_recurrent:
            P, D = self._pair_terms(rec.s_prev, rec.spikes, w_t)
            delta = self._delta(P, D, m.rec_magnitude())
            sel = self._pre_filter()
            m.apply_rec_magnitude_delta(delta if sel is None else delta * sel, self.w_max)
        if "in" in self.targets and m.train_input:
            P, D = self._pair_terms(rec.x, rec.spikes, w_t)
            m.apply_in_magnitude_delta(self._delta(P, D, (m.w_in * m.in_mask).abs()), self.w_max)
        return r_mean


# ---- 4. three-factor rule / e-prop ---------------------------------------- #
def _membrane_decay(model: RSNN) -> float:
    """Mean membrane decay of the recurrent neurons (for eligibility traces)."""
    for p in model.pops:
        beta = getattr(p, "_beta", None)
        if callable(beta):
            b = beta()
            return float(b.mean()) if isinstance(b, Tensor) else float(b)
    return math.exp(-model.dt / 20.0)


class ThreeFactor(Learner):
    """Eligibility-trace x post-synaptic pseudo-derivative x learning signal.

        dW_ij = -lr * sum_t  L_i(t) * psi_i(t) * zbar_j(t)

    * pre-synaptic factor  zbar_j: pre-synaptic spikes low-passed with the
      membrane decay (the local "eligibility trace");
    * post-synaptic factor psi_i: pseudo-derivative of the spike function;
    * third factor L_i: either a learning signal broadcast from the output error
      through feedback weights (``feedback``: "random" or "symmetric"; this is
      e-prop), or any modulator you pass in (neuromodulator, reward prediction
      error, ...) via ``update(rec, modulator=...)``.

    The readout is trained with its exact local gradient. For the leaky
    readout, the sum over t is evaluated with an equivalent anti-causal filter
    of L, which makes the whole update one matmul per matrix; it is
    mathematically identical to accumulating the online rule across the chunk
    and applying it at the end. It ignores recurrent spike-to-spike credit
    paths and, for adaptive neurons, the threshold-adaptation eligibility, so it
    approximates BPTT (that is the e-prop trade-off) and is exact only for the
    readout. Needs ``record_membrane=True`` recordings.
    """

    def __init__(
        self,
        model: RSNN,
        lr: float = 1e-3,
        lr_out: Optional[float] = None,
        feedback: Union[str, Tensor] = "random",
        targets: Sequence[str] = ("rec", "in"),
        train_readout: bool = True,
        tau_trace: Optional[float] = None,
        kappa: Optional[float] = None,
        seed: int = 0,
    ):
        super().__init__(model)
        self.lr = lr
        self.lr_out = lr if lr_out is None else lr_out
        self.targets = tuple(targets)
        self.train_readout = train_readout
        self.alpha = math.exp(-model.dt / tau_trace) if tau_trace else None
        self.kappa = kappa
        self.feedback = feedback
        self._seed = seed
        self._B: Optional[Tensor] = None

    def _feedback(self) -> Tensor:
        """(N, n_out) matrix broadcasting the output error to each neuron."""
        m = self.model
        if isinstance(self.feedback, Tensor):
            return self.feedback.to(m.w_out.device, m.w_out.dtype)
        if self.feedback == "symmetric":
            return (m.w_out * m.out_mask).detach().t()
        if self.feedback == "random":
            if self._B is None:
                g = torch.Generator().manual_seed(self._seed)       # same matrix on any device
                self._B = torch.randn(m.n_rec, m.n_out, generator=g) / math.sqrt(m.n_out)
            if self._B.device != m.w_out.device or self._B.dtype != m.w_out.dtype:
                self._B = self._B.to(m.w_out.device, m.w_out.dtype)  # move once, then reuse
            return self._B
        raise ValueError("feedback must be 'random', 'symmetric' or a (N, n_out) tensor")

    @staticmethod
    def _as_modulator(mod: Union[float, Tensor], rec: Recording) -> Tensor:
        T, B, _ = rec.spikes.shape
        mod = torch.as_tensor(mod, dtype=rec.spikes.dtype, device=rec.spikes.device)
        if mod.dim() == 0:
            return mod.reshape(1, 1, 1).expand(T, B, 1)
        if mod.dim() == 1:                       # per-sample
            return mod.reshape(1, B, 1).expand(T, B, 1)
        if mod.dim() == 2:                       # (T, B)
            return mod.unsqueeze(-1)
        return mod                               # (T, B, 1) or (T, B, N)

    @torch.no_grad()
    def gradients(self, rec: Recording, error: Optional[Tensor] = None,
                  modulator: Optional[Union[float, Tensor]] = None) -> Dict[str, Tensor]:
        """Loss-gradient estimates (effective-weight units, (post, pre))."""
        m = self.model
        if modulator is None and error is None:
            raise ValueError("pass an output `error` (dL/dy) or an explicit `modulator`")
        T, B, N = rec.spikes.shape
        out: Dict[str, Tensor] = {}
        if self.targets:      # skipped entirely for readout-only training
            if rec.v is None:
                raise ValueError("ThreeFactor needs a recording made with record_membrane=True")
            psi = m.pseudo_derivative(rec.v.detach())
            if modulator is not None:
                L = self._as_modulator(modulator, rec)
                kappa = self.kappa if self.kappa is not None else 0.0
            else:
                L = torch.matmul(error, self._feedback().t())      # (T, B, N)
                kappa = self.kappa if self.kappa is not None else m.kappa
            alpha = self.alpha if self.alpha is not None else _membrane_decay(m)
            g = _anticausal_filter(L, kappa) * psi                  # (T, B, N)
            if "rec" in self.targets:
                out["rec"] = _flat(g).t() @ _flat(_causal_filter(rec.s_prev, alpha)) / B
            if "in" in self.targets:
                out["in"] = _flat(g).t() @ _flat(_causal_filter(rec.x, alpha)) / B
        if error is not None and self.train_readout:
            z_out = _causal_filter(rec.spikes, m.kappa)
            out["out"] = _flat(error).t() @ _flat(z_out) / B
            ones = _causal_filter(torch.ones(T, 1, 1, dtype=error.dtype, device=error.device), m.kappa)
            out["bias"] = (error * ones).sum(dim=(0, 1)) / B
        return out

    @torch.no_grad()
    def update(self, rec: Recording, error: Optional[Tensor] = None,
               modulator: Optional[Union[float, Tensor]] = None) -> Dict[str, Tensor]:
        m = self.model
        g = self.gradients(rec, error, modulator)
        if "rec" in g and m.train_recurrent:
            m.apply_rec_effective_delta(-self.lr * g["rec"])
        if "in" in g and m.train_input:
            m.apply_in_effective_delta(-self.lr * g["in"])
        if "out" in g and m.train_readout:
            m.apply_out_delta(-self.lr_out * g["out"], -self.lr_out * g["bias"])
        return g


# ---- 5. stabilising / homeostatic rules ---------------------------------- #
class InhibitorySTDP(Learner):
    """Vogels et al. (2011) inhibitory plasticity on I->E synapses.

        dw_ij = lr * (pre_trace_j * post_i + post_trace_i * pre_j - alpha * pre_j)

    with alpha set by the target rate, so inhibition grows until each
    excitatory neuron fires near ``target_rate_hz``. In an always-active
    network this is what lets the background be suppressed while strong,
    specific inputs still get through.
    """

    def __init__(self, model: RSNN, lr: float = 1e-3, target_rate_hz: float = 5.0,
                 tau: float = 20.0, w_max: Optional[float] = None):
        super().__init__(model)
        self.lr, self.w_max = lr, w_max
        self.decay = math.exp(-model.dt / tau)
        rho0 = target_rate_hz * model.dt * 1e-3                      # spikes per step
        self.alpha = 2.0 * rho0 / (1.0 - self.decay)

    @torch.no_grad()
    def update(self, rec: Recording) -> None:
        m = self.model
        if not m.train_recurrent or m.n_inh == 0:
            return
        B = rec.spikes.shape[1]
        pre, post = rec.s_prev, rec.spikes
        P = _flat(post).t() @ _flat(_causal_filter(pre, self.decay))
        Q = _flat(_causal_filter(post, self.decay)).t() @ _flat(pre)
        S = _flat(pre).sum(dim=0)                                      # (N_pre,)
        delta = self.lr * (P + Q - self.alpha * S.unsqueeze(0)) / B
        block = m.is_exc.to(delta.dtype).unsqueeze(1) * (~m.is_exc).to(delta.dtype).unsqueeze(0)
        m.apply_rec_magnitude_delta(delta * block, self.w_max)


class HomeostaticScaling(Learner):
    """Multiplicative synaptic scaling: each neuron scales its excitatory (and
    input) synapses up when it fires below ``target_rate_hz`` and down when above."""

    def __init__(self, model: RSNN, target_rate_hz: float = 10.0, eta: float = 0.1,
                 targets: Sequence[str] = ("rec", "in"), max_step: float = 2.0):
        super().__init__(model)
        self.target = target_rate_hz * model.dt * 1e-3
        self.eta = eta
        self.targets = tuple(targets)
        self.max_step = max_step

    @torch.no_grad()
    def update(self, rec: Recording) -> Tensor:
        m = self.model
        rate = rec.spikes.mean(dim=(0, 1))                                # (N,)
        err = (self.target - rate) / max(self.target, 1e-12)
        scale = torch.exp(self.eta * err).clamp(1.0 / self.max_step, self.max_step)
        if "rec" in self.targets and m.train_recurrent:
            factor = torch.ones_like(m.w_rec)
            factor[:, m.is_exc] = scale.unsqueeze(1)
            m.w_rec.mul_(factor)
            m.project()
        if "in" in self.targets and m.train_input:
            m.w_in.mul_(scale.unsqueeze(1))
        return rate


# --------------------------------------------------------------------------- #
# Self-test:  python rsnn.py
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    torch.manual_seed(0)
    T, B, n_in, N, n_out = 40, 6, 12, 80, 3

    def spikes_in(p=0.2):
        return (torch.rand(T, B, n_in) < p).float()

    # 1. structure: size, E/I ratio, density, Dale's law, no self-connections
    net = RSNN(n_in, N, n_out, exc_fraction=0.75, connectivity=0.15, seed=1)
    assert (net.n_exc, net.n_inh) == (60, 20)
    assert abs(net.density - 0.15) < 0.04, net.density
    _, W, _ = net.effective_weights()
    assert (W[:, :60] >= 0).all() and (W[:, 60:] <= 0).all() and W.diagonal().abs().sum() == 0
    y = net(spikes_in())
    assert y.shape == (T, B, n_out) and torch.isfinite(y).all()

    # 2. per-type connectivity + spectral radius
    net2 = RSNN(n_in, N, n_out, connectivity=dict(EE=0.2, EI=0.4, IE=0.4, II=0.1),
                spectral_radius=0.9, seed=2)
    _, W2, _ = net2.effective_weights()
    assert abs(torch.linalg.eigvals(W2).abs().max().item() - 0.9) < 1e-3

    # 3. neuron models are interchangeable, incl. separate E / I models
    for kind, kw, scale in [("lif", {}, 1.0), ("if", {}, 1.0), ("alif", {}, 1.0),
                            ("stochastic_lif", {}, 1.0), ("izhikevich", {}, 10.0)]:
        n = RSNN(n_in, N, n_out, neuron=kind, neuron_kwargs=kw, input_scale=scale,
                 rec_scale=scale, seed=3)
        with torch.no_grad():
            yy = n(spikes_in())
        assert yy.shape == (T, B, n_out) and torch.isfinite(yy).all(), kind
    mixed = RSNN(n_in, N, n_out, neuron="izhikevich",
                 neuron_kwargs=dict(a=0.02, b=0.2, c=-65.0, d=8.0),
                 inh_neuron="izhikevich", inh_neuron_kwargs=dict(a=0.1, b=0.2, c=-65.0, d=2.0),
                 input_scale=10.0, rec_scale=10.0, seed=4)
    with torch.no_grad():
        assert mixed(spikes_in()).shape == (T, B, n_out)

    # 4. surrogate-gradient BPTT runs and gives finite gradients/loss
    net = RSNN(n_in, N, n_out, seed=5, input_scale=3.0, rec_scale=1.0)
    x, target = spikes_in(), torch.randn(T, B, n_out)
    bptt = SurrogateBPTT(net, lr=1e-2, tbptt=20, rate_target_hz=20.0, rate_weight=1.0)
    losses = [bptt.update(x, target) for _ in range(30)]
    assert all(math.isfinite(l) for l in losses)
    print(f"BPTT loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    # ... with a learnable loss module
    class ScaledMSE(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_w = nn.Parameter(torch.zeros(()))
        def forward(self, y, t):
            return torch.exp(self.log_w) * F.mse_loss(y, t)
    assert math.isfinite(SurrogateBPTT(net, loss_fn=ScaledMSE()).update(x, target))

    # 5. local rules: run, keep Dale's law and the sparsity mask intact
    net = RSNN(n_in, N, n_out, seed=6, input_scale=3.0, w_max=2.0)
    net.reset_state()
    with torch.no_grad():
        _, rec = net(x, record_membrane=True)
    before = net.w_rec.detach().clone()
    err = mse_error(rec.y, target)
    rules = [
        ("STDP", lambda: STDP(net, lr=0.5, w_max=2.0, mu=0.5, targets=("rec", "in")).update(rec)),
        ("R-STDP", lambda: RSTDP(net, lr=0.5, w_max=2.0).update(rec, reward=torch.rand(B))),
        ("3-factor", lambda: ThreeFactor(net, lr=1e-2).update(rec, error=err)),
        ("3-factor modulator", lambda: ThreeFactor(net, lr=1e-2).update(rec, modulator=torch.rand(B))),
        ("iSTDP", lambda: InhibitorySTDP(net, lr=1e-2).update(rec)),
        ("homeostasis", lambda: HomeostaticScaling(net).update(rec)),
    ]
    for name, run in rules:
        run()
        assert (net.w_rec >= 0).all(), name                           # Dale
        assert (net.w_rec * (1 - net.rec_mask)).abs().sum() == 0, name  # mask intact
        assert (net.w_rec <= 2.0 + 1e-6).all(), name                  # w_max
    assert not torch.equal(before, net.w_rec), "local rules changed nothing"

    # 6. e-prop readout gradient equals the autograd gradient exactly
    net = RSNN(n_in, N, n_out, seed=7, input_scale=3.0)
    net.reset_state()
    y, rec = net(x, record_membrane=True)
    loss = 0.5 * ((y - target) ** 2).sum() / B
    g_w, g_b = torch.autograd.grad(loss, [net.w_out, net.b_out], retain_graph=True)
    tf = ThreeFactor(net)
    g = tf.gradients(rec, error=(y - target).detach())
    assert torch.allclose(g["out"], g_w, rtol=1e-3, atol=1e-5), "readout gradient mismatch"
    assert torch.allclose(g["bias"], g_b, rtol=1e-3, atol=1e-5), "bias gradient mismatch"
    # recurrent: compare e-prop (symmetric feedback) with the BPTT gradient
    tf_sym = ThreeFactor(net, feedback="symmetric")
    g_rec = tf_sym.gradients(rec, error=(y - target).detach())["rec"]
    (g_bptt,) = torch.autograd.grad(loss, [net.w_rec])
    g_bptt_eff = g_bptt * net.sign * net.rec_mask        # raw -> effective-weight gradient
    cos = F.cosine_similarity(g_rec.flatten(), g_bptt_eff.flatten(), dim=0)
    print(f"e-prop vs BPTT recurrent gradient cosine similarity: {cos:.3f}")

    print("all rsnn self-tests passed")


if __name__ == "__main__":
    _selftest()

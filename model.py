"""
Spiking language model: a recurrent spiking neural network that reads text one
token at a time and predicts the next token.

    tokens (B, T) -> learned embedding -> input current
                  -> recurrent E/I spiking network (rsnn.RSNN, surrogate-gradient trainable)
                  -> linear readout of the spikes -> next-token logits (B, T, V)

* The recurrent core is ``rsnn.RSNN``: sparse excitatory/inhibitory connectivity,
  Dale's law, and any neuron model from ``neurons.py`` (default: adaptive LIF,
  whose slow threshold adaptation gives the network the long memory that language
  needs). The network is *stateful*: its membrane potentials, adaptation and
  last spikes carry over between calls, which is what lets training use long
  streams of text and lets ``generate`` feed one token at a time.
* Each token drives the network for ``steps_per_token`` time steps (1 = one
  spiking update per token). More steps give a richer rate-coded response per
  token at proportionally higher cost.
* Logits are read out from binary spikes, so spike rates matter: the network is
  calibrated to a target firing rate before training (``calibrate_rate``) and the
  training loop regularises the rate (see ``train_model.py``).

Everything is vectorised over the batch and runs on the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rsnn import RSNN


@dataclass
class LMConfig:
    vocab_size: int
    d_emb: int = 128                      # token embedding size (= RSNN input size)
    n_rec: int = 1024                     # recurrent spiking neurons
    exc_fraction: float = 0.8             # fraction of excitatory neurons
    connectivity: float = 0.05            # recurrent connection density
    input_connectivity: float = 1.0
    neuron: str = "alif"                  # "if", "lif", "alif", "stochastic_lif", "izhikevich"
    neuron_kwargs: dict = field(default_factory=lambda: dict(tau_m=20.0, tau_adapt=200.0,
                                                             adapt_strength=0.2))
    inh_neuron: Optional[str] = None      # separate model for inhibitory cells (None = same)
    inh_neuron_kwargs: Optional[dict] = None
    dt: float = 1.0                       # time units per step (time constants are in these units)
    steps_per_token: int = 1
    dale: bool = True
    inh_gain: float = 2.0
    input_scale: float = 1.0
    rec_scale: float = 1.0
    spectral_radius: Optional[float] = None
    tau_syn: Optional[float] = None
    tau_out: Optional[float] = None       # readout leak; None = logits from the current spikes only
    readout_scale: float = 1.0
    dropout: float = 0.1                  # on the embedding
    seed: Optional[int] = 0


class SpikingLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_emb)
        self.drop = nn.Dropout(cfg.dropout)
        self.rsnn = RSNN(
            cfg.d_emb, cfg.n_rec, cfg.vocab_size,
            exc_fraction=cfg.exc_fraction, connectivity=cfg.connectivity,
            input_connectivity=cfg.input_connectivity,
            neuron=cfg.neuron, neuron_kwargs=cfg.neuron_kwargs,
            inh_neuron=cfg.inh_neuron, inh_neuron_kwargs=cfg.inh_neuron_kwargs,
            dt=cfg.dt, dale=cfg.dale, inh_gain=cfg.inh_gain,
            input_scale=cfg.input_scale, rec_scale=cfg.rec_scale,
            spectral_radius=cfg.spectral_radius, tau_syn=cfg.tau_syn, tau_out=cfg.tau_out,
            readout_scale=cfg.readout_scale, seed=cfg.seed,
        )

    # ------------------------------------------------------------------ #
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset_state(self) -> None:
        """Forget the network state (start of a new document / stream)."""
        self.rsnn.reset_state()

    def detach_state(self) -> None:
        """Keep the state but cut the autograd graph (truncated BPTT)."""
        self.rsnn.detach_state()

    # ------------------------------------------------------------------ #
    def forward(self, tokens: Tensor, return_spikes: bool = False):
        """tokens: (B, T) integer ids. Returns logits (B, T, V) for the next token
        at every position; with ``return_spikes`` also the spike trains
        (T * steps_per_token, B, n_rec). State carries over between calls."""
        k = self.cfg.steps_per_token
        T = tokens.shape[1]
        x = self.drop(self.embed(tokens)).transpose(0, 1)             # (T, B, d_emb)
        if k > 1:
            x = x.repeat_interleave(k, dim=0)                         # hold each token for k steps
        if return_spikes:
            y, rec = self.rsnn(x, record=True)
            spikes = rec.spikes
        else:
            y, spikes = self.rsnn(x), None
        if k > 1:
            y = y.reshape(T, k, y.shape[1], y.shape[2]).mean(dim=1)   # average the k readouts
        logits = y.transpose(0, 1)                                    # (B, T, V)
        return (logits, spikes) if return_spikes else logits

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def calibrate_rate(self, tokens: Tensor, target_rate: float = 0.05, iters: int = 10) -> float:
        """Scale input and recurrent weights by a common factor (bisection in log
        space) so the untrained network fires near ``target_rate`` spikes per
        neuron per step on ``tokens``. A network that is silent or saturated
        carries no information, and nothing downstream can fix that. Returns the
        factor applied."""
        r = self.rsnn
        w_in0, w_rec0 = r.w_in.detach().clone(), r.w_rec.detach().clone()
        lo, hi = math.log(1e-2), math.log(1e2)
        best_err, best_f = float("inf"), 1.0
        was_training = self.training
        self.eval()
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = math.exp(mid)
            r.w_in.copy_(w_in0 * f)
            r.w_rec.copy_(w_rec0 * f)
            self.reset_state()
            _, spikes = self(tokens, return_spikes=True)
            rate = float(spikes.mean())
            err = abs(math.log((rate + 1e-6) / target_rate))
            if err < best_err:
                best_err, best_f = err, f
            if rate > target_rate:
                hi = mid
            else:
                lo = mid
        r.w_in.copy_(w_in0 * best_f)
        r.w_rec.copy_(w_rec0 * best_f)
        self.reset_state()
        self.train(was_training)
        return best_f

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, prompt: Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, generator: Optional[torch.Generator] = None) -> Tensor:
        """Autoregressive sampling. ``prompt``: (P,) or (B, P) token ids (P >= 1).
        The prompt is run through the network to build up its state, then tokens
        are fed back one at a time. ``temperature`` 0 = greedy. Returns
        (B, P + max_new_tokens)."""
        was_training = self.training
        self.eval()
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        self.reset_state()
        out = [prompt]
        logits = self(prompt)[:, -1]                                  # (B, V)
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k < logits.shape[-1]:
                    kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            out.append(nxt)
            logits = self(nxt)[:, -1]
        self.reset_state()
        self.train(was_training)
        return torch.cat(out, dim=1)

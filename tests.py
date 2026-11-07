"""
Benchmark sweeps for the RSNN: one figure per axis of variation, every figure
showing task accuracy.

    python tests.py                 # full run  (CPU: roughly 1-2 hours)
    python tests.py --quick         # tiny smoke run (a few minutes)
    python tests.py --only size ei  # subset of: size ei neuron method sparsity
    python tests.py --plots-only    # re-plot from saved results.json
    python tests.py --retune        # ignore cached hyper-parameters

Task (applied, time-series, noise-rejecting)
--------------------------------------------
Event detection in a noisy always-on input stream. 20 input channels emit
Poisson background spikes. In 2 of 3 trials a short burst of elevated rate
appears in channel group A (class 0) or group B (class 1) at a random time in
the first half of the trial; in the third (class 2) there is only noise. The
network must report the class *after a delay*, so it has to ignore noise,
respond to the specific perturbation, and remember it. The objective is
cross-entropy of the readout averaged over the final window; the metric is
accuracy on a fixed held-out test set (chance = 33%).

Protocol
--------
* Every learning method trains the same network on the same task stream.
    reservoir  recurrent weights frozen, readout trained by the delta rule
    stdp       unsupervised STDP on recurrent weights + delta-rule readout
    rstdp      reward-modulated STDP (reward = +1 correct / -1 wrong) + delta-rule readout
    eprop      three-factor rule (e-prop) on input/recurrent/readout weights
    bptt       surrogate-gradient backpropagation through time, all weights
* Hyper-parameters (learning rates) are tuned BEFORE the sweeps by random
  search on a validation set at a reference configuration, separately for each
  learning method (and, for the neuron-type sweep, for each neuron model).
  Tuning seeds and the validation set are disjoint from the sweep seeds and the
  test set. Tuned values are cached in hparams.json.
* Network gain (input and recurrent weight scale) is calibrated per
  configuration so the untrained network fires near TARGET_HZ. Without this,
  changing size, density or E/I ratio would mostly change how active the
  network is rather than how well it computes.
* Each sweep point is repeated over several seeds (network + data stream).

Outputs (in --out): results.json, hparams.json, one PNG and one CSV per sweep.
The CSV is the table view of each figure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from rsnn import RSNN, RSTDP, STDP, SurrogateBPTT, ThreeFactor

# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
N_IN, T_STEPS, WINDOW, N_CLASSES = 20, 60, 10, 3
P_BG, P_SIG, BURST_LEN = 0.10, 0.22, 8
TARGET_HZ = 15.0
TASK_DESC = "3-class burst detection in noise, readout after a delay"


def make_batch(batch: int, gen: torch.Generator):
    """x: (T, B, N_IN) spikes; labels: (B,) in {0: burst A, 1: burst B, 2: noise only}."""
    labels = torch.randint(0, N_CLASSES, (batch,), generator=gen)
    onset = torch.randint(5, T_STEPS // 2 - BURST_LEN + 1, (batch,), generator=gen)
    t = torch.arange(T_STEPS).view(T_STEPS, 1, 1)
    active = ((t >= onset.view(1, batch, 1)) & (t < (onset + BURST_LEN).view(1, batch, 1))).float()
    half = N_IN // 2
    group = torch.zeros(batch, N_IN)
    group[labels == 0, :half] = 1.0
    group[labels == 1, half:] = 1.0
    p = P_BG + (P_SIG - P_BG) * active * group.unsqueeze(0)
    x = (torch.rand(T_STEPS, batch, N_IN, generator=gen) < p).float()
    return x, labels


def window_logits(y: torch.Tensor) -> torch.Tensor:
    return y[-WINDOW:].mean(dim=0)


def ce_loss(y: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the window-averaged readout; target is (T, B) labels."""
    return F.cross_entropy(window_logits(y), target[0].long())


def window_error(y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """dL/dy_t of ``ce_loss``: nonzero only inside the final window."""
    p = torch.softmax(window_logits(y), dim=-1) - F.one_hot(labels, N_CLASSES).to(y.dtype)
    err = torch.zeros_like(y)
    err[-WINDOW:] = p.unsqueeze(0) / WINDOW
    return err


# --------------------------------------------------------------------------- #
# Configuration and budgets
# --------------------------------------------------------------------------- #
REF_NEURON = "lif"
METHODS = ("reservoir", "stdp", "rstdp", "eprop", "bptt")
MAIN = ("reservoir", "eprop", "bptt")        # series shown in the numeric-axis sweeps
NEURONS = ("if", "lif", "alif", "stochastic_lif", "izhikevich")


@dataclass(frozen=True)
class Cfg:
    n_rec: int = 100
    exc_fraction: float = 0.8
    connectivity: float = 0.1
    neuron: str = REF_NEURON
    method: str = "bptt"


@dataclass
class Budget:
    quick: bool
    ref_n: int            # network size used for tuning and for non-size sweeps
    iters: int            # training iterations per run
    batch: int
    n_eval: int           # test / validation set size
    seeds: int
    tune_trials: int      # random-search trials per method (reference neuron)
    tune_iters: int
    tune_seeds: int
    neuron_tune_trials: int  # trials per (method, neuron) in the neuron sweep
    rate_weight: float


FULL = Budget(False, ref_n=100, iters=120, batch=32, n_eval=256, seeds=3,
              tune_trials=8, tune_iters=120, tune_seeds=2, neuron_tune_trials=4, rate_weight=50.0)
QUICK = Budget(True, ref_n=40, iters=8, batch=16, n_eval=64, seeds=1,
               tune_trials=2, tune_iters=8, tune_seeds=1, neuron_tune_trials=1, rate_weight=50.0)

# log-uniform search ranges (the first trial is always the geometric centre)
SPACES: Dict[str, Dict[str, tuple]] = {
    "reservoir": {"lr_out": (1e-4, 1e-1)},
    "stdp": {"lr": (1e-3, 10.0), "lr_out": (1e-4, 1e-1)},
    "rstdp": {"lr": (1e-3, 10.0), "lr_out": (1e-4, 1e-1)},
    "eprop": {"lr": (1e-5, 1e-1), "lr_out": (1e-4, 1e-1)},
    "bptt": {"lr": (1e-4, 3e-2)},
}

SWEEPS = {
    "size": dict(field="n_rec", values=[25, 50, 100, 200, 400], quick=[20, 40], kind="lines",
                 xscale="log", xlabel="Recurrent network size (neurons)",
                 title="Accuracy vs network size", file="fig_network_size.png"),
    "ei": dict(field="exc_fraction", values=[0.5, 0.65, 0.8, 0.9, 1.0], quick=[0.6, 0.9], kind="lines",
               xscale="linear", xlabel="Excitatory fraction  E / (E + I)",
               title="Accuracy vs excitatory / inhibitory ratio", file="fig_ei_ratio.png"),
    "neuron": dict(field="neuron", values=list(NEURONS), quick=["lif", "izhikevich"], kind="bars",
                   xlabel="Neuron model", title="Accuracy vs neuron type", file="fig_neuron_type.png"),
    "method": dict(field="method", values=list(METHODS), quick=list(METHODS), kind="method",
                   xlabel="Learning method", title="Accuracy vs learning method",
                   file="fig_learning_method.png"),
    "sparsity": dict(field="connectivity", values=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0], quick=[0.05, 0.5],
                     kind="lines", xscale="log", xlabel="Recurrent connection density",
                     title="Accuracy vs sparsity", file="fig_sparsity.png"),
}

METHOD_NAME = {"reservoir": "Reservoir (readout only)", "stdp": "STDP + readout",
               "rstdp": "R-STDP + readout", "eprop": "e-prop (three-factor)",
               "bptt": "Surrogate-gradient BPTT"}
METHOD_SHORT = {"reservoir": "Reservoir", "stdp": "STDP", "rstdp": "R-STDP",
                "eprop": "e-prop", "bptt": "BPTT"}
NEURON_NAME = {"if": "IF", "lif": "LIF", "alif": "Adaptive LIF",
               "stochastic_lif": "Stochastic LIF", "izhikevich": "Izhikevich"}

FAILURES: List[str] = []


# --------------------------------------------------------------------------- #
# One training run
# --------------------------------------------------------------------------- #
def build_net(cfg: Cfg, gain: float, seed: int) -> RSNN:
    return RSNN(N_IN, cfg.n_rec, N_CLASSES, exc_fraction=cfg.exc_fraction,
                connectivity=cfg.connectivity, neuron=cfg.neuron,
                input_scale=gain, rec_scale=gain, seed=seed)


_GAIN_CACHE: Dict[tuple, float] = {}


@torch.no_grad()
def calibrate_gain(cfg: Cfg, seed: int, iters: int = 10) -> float:
    """Bisection (in log space) on the weight gain so the untrained network
    fires near TARGET_HZ on the task input."""
    key = (cfg.neuron, cfg.n_rec, cfg.exc_fraction, cfg.connectivity, seed)
    if key in _GAIN_CACHE:
        return _GAIN_CACHE[key]
    gen = torch.Generator().manual_seed(10_000 + seed)
    x, _ = make_batch(32, gen)
    lo, hi = math.log(1e-2), math.log(1e3)
    best_err, best_gain = float("inf"), math.exp((lo + hi) / 2)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        net = build_net(cfg, math.exp(mid), seed)
        net.reset_state()
        _, rec = net(x, record=True)
        rate = float(rec.rate_hz(net.dt).mean())
        err = abs(math.log((rate + 1e-3) / TARGET_HZ))
        if err < best_err:
            best_err, best_gain = err, math.exp(mid)
        if rate > TARGET_HZ:
            hi = mid
        else:
            lo = mid
    _GAIN_CACHE[key] = best_gain
    return best_gain


def run(cfg: Cfg, hp: dict, bud: Budget, seed: int, split: str = "test",
        iters: Optional[int] = None) -> dict:
    """Train ``cfg.method`` and return accuracy on a fixed held-out set."""
    torch.manual_seed(seed)
    gain = calibrate_gain(cfg, seed)
    net = build_net(cfg, gain, seed)
    net.w_max = 3.0 * float(net.rec_magnitude().max())      # cap for the STDP family
    gen = torch.Generator().manual_seed(seed * 7919 + 1)
    x_eval, y_eval = make_batch(bud.n_eval, torch.Generator().manual_seed(555 if split == "val" else 999))
    iters = iters or bud.iters
    m = cfg.method

    bptt = stdp = rstdp = eprop = readout = None
    if m == "bptt":
        bptt = SurrogateBPTT(net, loss_fn=ce_loss, lr=hp["lr"], grad_clip=1.0,
                             rate_target_hz=TARGET_HZ, rate_weight=bud.rate_weight)
    elif m == "eprop":
        eprop = ThreeFactor(net, lr=hp["lr"], lr_out=hp["lr_out"], targets=("rec", "in"))
    else:
        readout = ThreeFactor(net, lr=0.0, lr_out=hp["lr_out"], targets=())   # delta rule
        if m == "stdp":
            stdp = STDP(net, lr=hp["lr"], w_max=net.w_max, mu=0.5)
        elif m == "rstdp":
            rstdp = RSTDP(net, tau_elig=100.0, lr=hp["lr"], w_max=net.w_max, mu=0.5)

    t0 = time.time()
    for _ in range(iters):
        x, labels = make_batch(bud.batch, gen)
        if bptt is not None:
            bptt.update(x, labels.unsqueeze(0).expand(T_STEPS, -1))
            continue
        net.reset_state()
        with torch.no_grad():
            if eprop is not None:
                y, rec = net(x, record_membrane=True)
            else:
                y, rec = net(x, record=True)
        err = window_error(y, labels)
        if eprop is not None:
            eprop.update(rec, error=err)
        else:
            readout.update(rec, error=err)
            if stdp is not None:
                stdp.update(rec)
            elif rstdp is not None:
                correct = (window_logits(y).argmax(dim=1) == labels).float()
                rstdp.update(rec, reward=2.0 * correct - 1.0)
    seconds = time.time() - t0

    net.reset_state()
    with torch.no_grad():
        y, rec = net(x_eval, record=True)
    logits = torch.nan_to_num(window_logits(y))
    acc = float((logits.argmax(dim=1) == y_eval).float().mean())
    return dict(acc=acc, seconds=seconds, rate_hz=float(rec.rate_hz(net.dt).mean()), gain=gain)


def safe_run(cfg: Cfg, hp: dict, bud: Budget, seed: int, **kw) -> dict:
    """A failed run is recorded as NaN (and reported at the end), so one bad
    configuration cannot throw away an hours-long sweep."""
    try:
        return run(cfg, hp, bud, seed, **kw)
    except Exception:
        traceback.print_exc()
        FAILURES.append(f"{cfg} seed={seed}")
        return dict(acc=float("nan"), seconds=float("nan"), rate_hz=float("nan"), gain=float("nan"))


# --------------------------------------------------------------------------- #
# Hyper-parameter search (done before the sweeps, cached on disk)
# --------------------------------------------------------------------------- #
def sample_hp(space: Dict[str, tuple], n: int, rng: random.Random) -> List[dict]:
    trials = [{k: math.sqrt(lo * hi) for k, (lo, hi) in space.items()}]
    while len(trials) < n:
        trials.append({k: math.exp(rng.uniform(math.log(lo), math.log(hi)))
                       for k, (lo, hi) in space.items()})
    return trials


class HPStore:
    """Tuned hyper-parameters keyed by 'method|neuron', persisted as JSON."""

    def __init__(self, path: str, retune: bool = False):
        self.path = path
        self.data: Dict[str, dict] = {}
        if os.path.exists(path) and not retune:
            with open(path) as f:
                self.data = json.load(f)

    def get(self, method: str, neuron: str, bud: Budget) -> dict:
        key = f"{method}|{neuron}"
        if key not in self.data:
            self.data[key] = self._tune(method, neuron, bud)
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        return self.data[key]["hp"]

    @staticmethod
    def _tune(method: str, neuron: str, bud: Budget) -> dict:
        cfg = Cfg(n_rec=bud.ref_n, neuron=neuron, method=method)
        n_trials = bud.tune_trials if neuron == REF_NEURON else bud.neuron_tune_trials
        rng = random.Random(f"{method}|{neuron}")
        trials = []
        for i, hp in enumerate(sample_hp(SPACES[method], n_trials, rng)):
            accs = [safe_run(cfg, hp, bud, seed=1000 + s, split="val", iters=bud.tune_iters)["acc"]
                    for s in range(bud.tune_seeds)]
            accs = [a for a in accs if not math.isnan(a)]
            val = statistics.mean(accs) if accs else float("nan")
            trials.append(dict(hp=hp, val_acc=val))
            print(f"[tune {method}|{neuron}] trial {i + 1}/{n_trials} "
                  f"{ {k: round(v, 5) for k, v in hp.items()} } val acc {val:.3f}", flush=True)
        best = max(trials, key=lambda t: -1.0 if math.isnan(t["val_acc"]) else t["val_acc"])
        return dict(hp=best["hp"], val_acc=best["val_acc"], trials=trials)


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def run_sweep(name: str, bud: Budget, store: HPStore) -> List[dict]:
    spec = SWEEPS[name]
    values = spec["quick"] if bud.quick else spec["values"]
    base = Cfg(n_rec=bud.ref_n)
    records: List[dict] = []
    for x in values:
        methods = (x,) if name == "method" else MAIN
        for m in methods:
            neuron = x if name == "neuron" else REF_NEURON
            cfg = replace(base, method=m, **({} if name == "method" else {spec["field"]: x}))
            hp = store.get(m, neuron, bud)
            for s in range(bud.seeds):
                r = safe_run(cfg, hp, bud, seed=s)
                records.append(dict(x=x, method=m, seed=s, **r))
                print(f"[{name}] x={x} method={m} seed={s}: acc={r['acc']:.3f} "
                      f"rate={r['rate_hz']:.1f}Hz  {r['seconds']:.1f}s", flush=True)
    return records


def aggregate(records: List[dict]) -> Dict[str, Dict[object, dict]]:
    """{method: {x: {acc, std, n, seconds, rate_hz}}}  (NaN runs are ignored)."""
    groups: Dict[tuple, List[dict]] = {}
    for r in records:
        groups.setdefault((r["method"], r["x"]), []).append(r)
    out: Dict[str, Dict[object, dict]] = {}
    for (m, x), rs in groups.items():
        ok = [r for r in rs if not math.isnan(r["acc"])]
        acc = [r["acc"] for r in ok]
        out.setdefault(m, {})[x] = dict(
            acc=statistics.mean(acc) if acc else float("nan"),
            std=statistics.stdev(acc) if len(acc) > 1 else 0.0,
            n=len(acc),
            seconds=statistics.mean([r["seconds"] for r in ok]) if ok else float("nan"),
            rate_hz=statistics.mean([r["rate_hz"] for r in ok]) if ok else float("nan"),
        )
    return out


def write_csv(path: str, sweep: str, agg: Dict[str, Dict[object, dict]]) -> None:
    with open(path, "w") as f:
        f.write("sweep,x,method,acc_mean,acc_std,n_seeds,train_seconds,rate_hz\n")
        for m, by_x in agg.items():
            for x, a in by_x.items():
                f.write(f"{sweep},{x},{m},{a['acc']:.4f},{a['std']:.4f},{a['n']},"
                        f"{a['seconds']:.2f},{a['rate_hz']:.2f}\n")


# --------------------------------------------------------------------------- #
# Plotting (matplotlib, imported lazily so the experiments run without it)
# --------------------------------------------------------------------------- #
# Light-mode tokens and categorical slots 1-3 of the validated reference
# palette (all-pairs CVD dE >= 9.2, normal-vision dE >= 24). Colour follows the
# method, in every figure. The aqua slot is under 3:1 contrast on the surface,
# so every figure also carries a legend, direct labels where they fit, distinct
# marker shapes, and a CSV table view.
SURFACE, PRIMARY, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
COLOR = {"reservoir": "#2a78d6", "eprop": "#eb6834", "bptt": "#1baf7a"}
MARKER = {"reservoir": "o", "eprop": "s", "bptt": "^"}
SINGLE = "#2a78d6"        # one-series bar charts use categorical slot 1
CHANCE = 1.0 / N_CLASSES
DPI = 200


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _new_axes(fig, rect, ylim, ylabel, pct=True):
    from matplotlib.ticker import FuncFormatter
    ax = fig.add_axes(rect, facecolor=SURFACE)
    ax.set_ylim(*ylim)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=8, labelcolor=SECONDARY, pad=5)
    if pct:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel(ylabel, fontsize=8.5, color=SECONDARY, labelpad=6)
    return ax


def _titles(fig, title, subtitle):
    fig.text(0.02, 0.955, title, fontsize=12, fontweight="semibold", color=PRIMARY, ha="left", va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=8, color=SECONDARY, ha="left", va="top")


def _fmt(v):
    return str(v) if isinstance(v, int) else f"{v:g}"


def _subtitle(meta) -> str:
    return (f"{TASK_DESC}.  Mean ± s.d. over {meta['seeds']} seeds, test set n={meta['n_eval']}.  "
            f"Chance = {CHANCE:.0%}.")


def plot_lines(path, spec, agg, values, meta):
    import numpy as np
    from matplotlib.lines import Line2D
    plt = _mpl()
    fig = plt.figure(figsize=(6.6, 4.6), dpi=DPI, facecolor=SURFACE)
    ax = _new_axes(fig, [0.11, 0.27, 0.80, 0.56], (0.0, 1.0), "Test accuracy")
    xs = np.array(values, dtype=float)
    if spec["xscale"] == "log":
        ax.set_xscale("log")
    if spec["xscale"] == "log":
        ax.set_xlim(xs.min() / 1.15, xs.max() * 1.9)             # room on the right for end labels
    else:
        pad = xs.max() - xs.min()
        ax.set_xlim(xs.min() - 0.03 * pad, xs.max() + 0.22 * pad)
    ax.set_xticks(xs)
    ax.set_xticklabels([_fmt(v) for v in values])
    ax.minorticks_off()
    ax.set_xlabel(spec["xlabel"], fontsize=8.5, color=SECONDARY, labelpad=6)

    ax.axhline(CHANCE, color=MUTED, linewidth=1.0, zorder=1)
    ax.text(xs.min(), CHANCE + 0.015, "chance", fontsize=7.5, color=MUTED, ha="left", va="bottom")

    ends = []
    for m in MAIN:
        by_x = agg.get(m, {})
        mean = np.array([by_x.get(v, {}).get("acc", np.nan) for v in values])
        std = np.array([by_x.get(v, {}).get("std", np.nan) for v in values])
        ax.fill_between(xs, mean - std, mean + std, color=COLOR[m], alpha=0.10, linewidth=0, zorder=2)
        ax.plot(xs, mean, color=COLOR[m], linewidth=1.8, solid_capstyle="round", solid_joinstyle="round",
                marker=MARKER[m], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        valid = np.where(~np.isnan(mean))[0]
        if len(valid):
            ends.append((mean[valid[-1]], xs[valid[-1]], m))

    # direct labels at the line ends, only when they do not collide (else the legend carries identity)
    ys = sorted(e[0] for e in ends)
    if len(ends) > 1 and min(b - a for a, b in zip(ys, ys[1:])) >= 0.07:
        for y, x, m in ends:
            ax.annotate(METHOD_SHORT[m], (x, y), xytext=(8, 0), textcoords="offset points",
                        fontsize=8, color=PRIMARY, va="center", ha="left")

    handles = [Line2D([0], [0], color=COLOR[m], linewidth=1.8, marker=MARKER[m], markersize=7,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label=METHOD_NAME[m]) for m in MAIN]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=SECONDARY, bbox_to_anchor=(0.5, 0.01), handlelength=2.2, columnspacing=1.6)
    _titles(fig, spec["title"], _subtitle(meta))
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def _round_bar(ax, x, h, w, color, r_px=8.0):
    """Bar with a rounded data-end (4 css px at 2x) and a square baseline."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    if not (h > 0):
        return
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ux, uy = (x1 - x0) / bb.width, (y1 - y0) / bb.height          # data units per device pixel
    rx, ry = r_px * ux, r_px * uy
    shrink = min(1.0, (h / 2) / ry, (w / 2) / rx)
    rx, ry = rx * shrink, ry * shrink
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none", linewidth=0,
                                zorder=3))
    ax.add_patch(Rectangle((x - w / 2, 0), w, ry, facecolor=color, edgecolor="none", linewidth=0, zorder=3))


def _bar_panel(fig, rect, cats, cat_labels, series, ylim, ylabel, pct, label_tips, xlabel):
    """series: list of (name, color, means, stds). Bars <= 24 css px wide, 2 px gaps."""
    import numpy as np
    ax = _new_axes(fig, rect, ylim, ylabel, pct=pct)
    n, k = len(cats), len(series)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(cat_labels)
    ax.set_xlabel(xlabel, fontsize=8.5, color=SECONDARY, labelpad=6)
    slot = min(0.24, 0.72 / k)
    bb = ax.get_window_extent()
    gap = 4.0 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / bb.width            # 2 css px at 2x
    span = ylim[1] - ylim[0]
    for j, (name, color, means, stds) in enumerate(series):
        for i in range(n):
            m, s = means[i], stds[i]
            if m is None or np.isnan(m):
                continue
            xc = i + (j - (k - 1) / 2) * slot
            _round_bar(ax, xc, m, slot - gap, color)
            if s and not np.isnan(s) and s > 0:
                ax.plot([xc, xc], [max(m - s, 0), m + s], color=SECONDARY, linewidth=1.0,
                        solid_capstyle="butt", zorder=4)
            if label_tips:
                txt = f"{m:.0%}" if pct else f"{m:.0f}s"
                ax.text(xc, m + (s if s and not np.isnan(s) else 0) + 0.015 * span, txt,
                        fontsize=8, color=PRIMARY, ha="center", va="bottom", zorder=5)
    return ax


def plot_bars(path, spec, agg, values, meta):
    """Neuron-type figure: grouped bars, one series per learning method."""
    import numpy as np
    from matplotlib.patches import Patch
    plt = _mpl()
    fig = plt.figure(figsize=(6.6, 4.6), dpi=DPI, facecolor=SURFACE)
    series = []
    for m in MAIN:
        by_x = agg.get(m, {})
        series.append((METHOD_NAME[m], COLOR[m],
                       [by_x.get(v, {}).get("acc", np.nan) for v in values],
                       [by_x.get(v, {}).get("std", np.nan) for v in values]))
    ax = _bar_panel(fig, [0.11, 0.27, 0.86, 0.56], values, [NEURON_NAME.get(v, v) for v in values],
                    series, (0.0, 1.0), "Test accuracy", True, False, spec["xlabel"])
    ax.axhline(CHANCE, color=MUTED, linewidth=1.0, zorder=1)
    ax.text(ax.get_xlim()[0] + 0.02, CHANCE + 0.015, "chance", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    handles = [Patch(facecolor=COLOR[m], edgecolor="none", label=METHOD_NAME[m]) for m in MAIN]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=SECONDARY, bbox_to_anchor=(0.5, 0.01), handlelength=1.2, columnspacing=1.6)
    _titles(fig, spec["title"], _subtitle(meta))
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_method(path, spec, agg, values, meta):
    """Learning-method figure: accuracy and training time as two panels (never a dual axis)."""
    import numpy as np
    plt = _mpl()
    fig = plt.figure(figsize=(8.4, 4.4), dpi=DPI, facecolor=SURFACE)
    names = [METHOD_SHORT[v] for v in values]
    acc = [agg.get(v, {}).get(v, {}).get("acc", np.nan) for v in values]
    std = [agg.get(v, {}).get(v, {}).get("std", np.nan) for v in values]
    sec = [agg.get(v, {}).get(v, {}).get("seconds", np.nan) for v in values]
    ax = _bar_panel(fig, [0.08, 0.17, 0.41, 0.63], values, names, [("Test accuracy", SINGLE, acc, std)],
                    (0.0, 1.0), "Test accuracy", True, True, spec["xlabel"])
    ax.axhline(CHANCE, color=MUTED, linewidth=1.0, zorder=1)
    ax.text(ax.get_xlim()[0] + 0.02, CHANCE + 0.015, "chance", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    top = max([s for s in sec if not np.isnan(s)] or [1.0]) * 1.25
    _bar_panel(fig, [0.58, 0.17, 0.40, 0.63], values, names, [("Training time", SINGLE, sec, [0] * len(sec))],
               (0.0, top), "Training time (s)", False, True, spec["xlabel"])
    _titles(fig, spec["title"], _subtitle(meta) + "  Right: wall-clock training time.")
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


PLOTTERS = {"lines": plot_lines, "bars": plot_bars, "method": plot_method}


def make_outputs(out: str, name: str, records: List[dict], meta: dict) -> None:
    spec = SWEEPS[name]
    values = spec["quick"] if meta["quick"] else spec["values"]
    agg = aggregate(records)
    write_csv(os.path.join(out, spec["file"].replace(".png", ".csv")), name, agg)
    PLOTTERS[spec["kind"]](os.path.join(out, spec["file"]), spec, agg, values, meta)
    print(f"wrote {spec['file']} (+ csv)", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny budget for a smoke run")
    ap.add_argument("--only", nargs="+", choices=list(SWEEPS), help="run a subset of sweeps")
    ap.add_argument("--out", default=None, help="output directory (default results / results_quick)")
    ap.add_argument("--seeds", type=int, default=None, help="override seeds per point")
    ap.add_argument("--retune", action="store_true", help="redo the hyper-parameter search")
    ap.add_argument("--force", action="store_true", help="re-run sweeps already in results.json")
    ap.add_argument("--plots-only", action="store_true", help="only re-draw from results.json")
    args = ap.parse_args(argv)

    bud = QUICK if args.quick else FULL
    if args.seeds:
        bud = replace(bud, seeds=args.seeds)
    out = args.out or ("results_quick" if args.quick else "results")
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(out, "results.json")
    results = {"meta": asdict(bud), "sweeps": {}}
    if os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)
    names = args.only or list(SWEEPS)

    if not args.plots_only:
        store = HPStore(os.path.join(out, "hparams.json"), retune=args.retune)
        results["meta"] = asdict(bud)
        for name in names:
            if name in results["sweeps"] and not args.force:
                print(f"[{name}] already in results.json, skipping (use --force to redo)")
                continue
            results["sweeps"][name] = run_sweep(name, bud, store)
            with open(res_path, "w") as f:                      # save after every sweep
                json.dump(results, f, indent=1)

    for name in names:
        if name in results["sweeps"]:
            make_outputs(out, name, results["sweeps"][name], results["meta"])

    if FAILURES:
        print(f"\n{len(FAILURES)} run(s) failed and are recorded as NaN:", file=sys.stderr)
        for f_ in FAILURES[:20]:
            print("  ", f_, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

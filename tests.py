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
appears in channel group A (class 0) or group B (class 1) at a random time
(steps 5-22 of 60); in the third (class 2) there is only noise. The
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

Language-model sweep ("steps")
------------------------------
The one sweep that is not on the burst-detection task: it trains the spiking
language model of model.py on character-level text and varies how many spiking
updates each token gets (steps_per_token = 1..8, readouts averaged). With one
update the logits come from a single step of binary spikes; more updates give a
graded, rate-coded readout and more recurrent computation per token, at k times
the compute. Everything else (data, steps, seeds, learning rate) is held fixed.
Metrics: test next-character accuracy and bits per character, plus training time.

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
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from rsnn import RSNN, RSTDP, STDP, SurrogateBPTT, ThreeFactor
from compare_models import evaluate as lm_evaluate, make_spiking, run_training   # language-model sweep
from train_model import CharTokenizer, batchify, load_text, split_ids

# Runs on the GPU when one is available. Override with --device, or by
# reassigning DEVICE (e.g. torch.device("cpu")) before calling run()/run_sweep().
# Data are generated on the CPU with seeded generators (so a seed means the same
# trials on any device) and moved over; everything else lives on DEVICE.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
N_IN, T_STEPS, WINDOW, N_CLASSES = 20, 60, 10, 3
# Calibrated with a NumPy replica of this network: an ideal observer reaches
# 100%, a frozen random reservoir with a trained readout ~60%, which leaves
# room for methods that also learn the recurrent weights. Weaker bursts
# (e.g. 0.10 / 0.22 / 8 steps) leave NO usable information in a frozen
# reservoir, so every method that cannot build memory sits at chance.
P_BG, P_SIG, BURST_LEN = 0.05, 0.40, 10
ONSET_LO, ONSET_HI = 5, 22
TARGET_HZ = 15.0
TASK_DESC = "3-class burst detection in noise, readout after a delay"


def make_batch(batch: int, gen: torch.Generator, device: Optional[torch.device] = None):
    """x: (T, B, N_IN) spikes; labels: (B,) in {0: burst A, 1: burst B, 2: noise only}.
    Generated on the CPU from ``gen`` and moved to ``device`` if given."""
    labels = torch.randint(0, N_CLASSES, (batch,), generator=gen)
    onset = torch.randint(ONSET_LO, ONSET_HI + 1, (batch,), generator=gen)
    t = torch.arange(T_STEPS).view(T_STEPS, 1, 1)
    active = ((t >= onset.view(1, batch, 1)) & (t < (onset + BURST_LEN).view(1, batch, 1))).float()
    half = N_IN // 2
    group = torch.zeros(batch, N_IN)
    group[labels == 0, :half] = 1.0
    group[labels == 1, half:] = 1.0
    p = P_BG + (P_SIG - P_BG) * active * group.unsqueeze(0)
    x = (torch.rand(T_STEPS, batch, N_IN, generator=gen) < p).float()
    if device is not None:
        x, labels = x.to(device), labels.to(device)
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


def oracle_accuracy(n_train: int = 2000, n_test: int = 256) -> float:
    """Ceiling: an ideal observer that reads the raw input spikes. Scan statistic =
    the largest sliding-window (BURST_LEN) spike count in each channel group;
    'noise only' if both are below a threshold fitted on training trials."""
    half = N_IN // 2

    def stats(x):
        c = torch.cat([torch.zeros(1, *x.shape[1:]), x.cumsum(0)])
        w = c[BURST_LEN:] - c[:-BURST_LEN]                          # (T-L+1, B, N_IN)
        return torch.stack([w[:, :, :half].sum(2).amax(0), w[:, :, half:].sum(2).amax(0)], dim=1)

    def predict(s, thr):
        top = s.argmax(dim=1)
        return torch.where(s.amax(dim=1) < thr, torch.full_like(top, N_CLASSES - 1), top)

    x, y = make_batch(n_train, torch.Generator().manual_seed(3))
    xt, yt = make_batch(n_test, torch.Generator().manual_seed(999))     # same set as the test split
    s, st = stats(x), stats(xt)
    thr = max(range(int(s.max()) + 2), key=lambda t: float((predict(s, t) == y).float().mean()))
    return float((predict(st, thr) == yt).float().mean())


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
    # language-model sweep (spiking updates per token)
    lm_data: str = "tinyshakespeare"
    lm_n_rec: int = 256
    lm_d_emb: int = 64
    lm_seq: int = 64
    lm_batch: int = 32
    lm_steps: int = 400
    lm_tune_steps: int = 120
    lm_eval_chunks: int = 40
    lm_seeds: int = 2


FULL = Budget(False, ref_n=100, iters=120, batch=32, n_eval=256, seeds=3,
              tune_trials=8, tune_iters=120, tune_seeds=2, neuron_tune_trials=4, rate_weight=50.0)
QUICK = Budget(True, ref_n=40, iters=8, batch=16, n_eval=64, seeds=1,
               tune_trials=2, tune_iters=8, tune_seeds=1, neuron_tune_trials=1, rate_weight=50.0,
               lm_data="builtin", lm_n_rec=64, lm_d_emb=16, lm_seq=16, lm_batch=8, lm_steps=15,
               lm_tune_steps=8, lm_eval_chunks=3, lm_seeds=1)

# log-uniform search ranges (the first trial is always the geometric centre)
# Readout step sizes must be large: the delta-rule gradient is small (the error is
# divided by WINDOW and the filtered spike features are ~0.3), so at lr_out <= 0.1
# the readout barely leaves its initialisation in ~100 iterations.
SPACES: Dict[str, Dict[str, tuple]] = {
    "reservoir": {"lr_out": (1e-2, 1e2)},
    "stdp": {"lr": (1e-3, 10.0), "lr_out": (1e-2, 1e2)},
    "rstdp": {"lr": (1e-3, 10.0), "lr_out": (1e-2, 1e2)},
    "eprop": {"lr": (1e-4, 1.0), "lr_out": (1e-2, 1e2)},
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
    "steps": dict(field="steps_per_token", values=list(range(1, 9)), quick=[1, 2, 4], kind="steps",
                  xlabel="Spiking updates per token", title="Language model vs spiking updates per token",
                  file="fig_steps_per_token.png"),
}

METHOD_NAME = {"reservoir": "Reservoir (readout only)", "stdp": "STDP + readout",
               "rstdp": "R-STDP + readout", "eprop": "e-prop (three-factor)",
               "bptt": "Surrogate-gradient BPTT"}
METHOD_SHORT = {"reservoir": "Reservoir", "stdp": "STDP", "rstdp": "R-STDP",
                "eprop": "e-prop", "bptt": "BPTT"}
NEURON_NAME = {"if": "IF", "lif": "LIF", "alif": "Adaptive LIF",
               "stochastic_lif": "Stochastic LIF", "izhikevich": "Izhikevich"}

FAILURES: List[str] = []

# Fingerprint of everything that changes what a result means. Cached results and
# tuned hyper-parameters from a different task/search space are discarded.
TASK_SIG = json.loads(json.dumps(dict(
    n_in=N_IN, T=T_STEPS, window=WINDOW, p_bg=P_BG, p_sig=P_SIG, burst_len=BURST_LEN,
    onset=[ONSET_LO, ONSET_HI], target_hz=TARGET_HZ, spaces=SPACES)))


# --------------------------------------------------------------------------- #
# One training run
# --------------------------------------------------------------------------- #
def build_net(cfg: Cfg, gain: float, seed: int) -> RSNN:
    net = RSNN(N_IN, cfg.n_rec, N_CLASSES, exc_fraction=cfg.exc_fraction,
               connectivity=cfg.connectivity, neuron=cfg.neuron,
               input_scale=gain, rec_scale=gain, seed=seed)
    return net.to(DEVICE)       # before any optimiser is created


_GAIN_CACHE: Dict[tuple, float] = {}


@torch.no_grad()
def calibrate_gain(cfg: Cfg, seed: int, iters: int = 10) -> float:
    """Bisection (in log space) on the weight gain so the untrained network
    fires near TARGET_HZ on the task input."""
    key = (cfg.neuron, cfg.n_rec, cfg.exc_fraction, cfg.connectivity, seed)
    if key in _GAIN_CACHE:
        return _GAIN_CACHE[key]
    gen = torch.Generator().manual_seed(10_000 + seed)
    x, _ = make_batch(32, gen, DEVICE)
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


def _sync() -> None:
    """CUDA kernels are asynchronous; wait so wall-clock timings are honest."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run(cfg: Cfg, hp: dict, bud: Budget, seed: int, split: str = "test",
        iters: Optional[int] = None) -> dict:
    """Train ``cfg.method`` and return accuracy on a fixed held-out set."""
    torch.manual_seed(seed)
    gain = calibrate_gain(cfg, seed)
    net = build_net(cfg, gain, seed)
    net.w_max = 3.0 * float(net.rec_magnitude().max())      # cap for the STDP family
    gen = torch.Generator().manual_seed(seed * 7919 + 1)
    x_eval, y_eval = make_batch(bud.n_eval, torch.Generator().manual_seed(555 if split == "val" else 999),
                                DEVICE)
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

    _sync()
    t0 = time.time()
    for _ in range(iters):
        x, labels = make_batch(bud.batch, gen, DEVICE)
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
    _sync()
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
            if self.data.get("_task") != TASK_SIG:
                print(f"{path}: tuned for a different task / search space, discarding", flush=True)
                self.data = {}

    def get(self, method: str, neuron: str, bud: Budget) -> dict:
        key = f"{method}|{neuron}"
        if key not in self.data:
            self.data[key] = self._tune(method, neuron, bud)
            self.data["_task"] = TASK_SIG
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        return self.data[key]["hp"]

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value
        self.data["_task"] = TASK_SIG
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

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
    if name == "steps":
        return run_steps_sweep(bud, store)
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
        entry = dict(
            acc=statistics.mean(acc) if acc else float("nan"),
            std=statistics.stdev(acc) if len(acc) > 1 else 0.0,
            n=len(acc),
            seconds=statistics.mean([r["seconds"] for r in ok]) if ok else float("nan"),
            rate_hz=statistics.mean([r["rate_hz"] for r in ok]) if ok else float("nan"),
        )
        bpc = [r["bpc"] for r in ok if "bpc" in r]              # language-model sweep only
        if bpc:
            entry.update(bpc=statistics.mean(bpc), bpc_std=statistics.stdev(bpc) if len(bpc) > 1 else 0.0)
        out.setdefault(m, {})[x] = entry
    return out


TABLE_COLUMNS = ("sweep", "x", "method", "acc_mean", "acc_std", "n_seeds", "train_seconds", "rate_hz")


def table_rows(sweep: str, agg: Dict[str, Dict[object, dict]]) -> List[dict]:
    """The table view of a figure (also what ends up in the CSV)."""
    rows = []
    for m, by_x in agg.items():
        for x, a in by_x.items():
            row = dict(sweep=sweep, x=x, method=m, acc_mean=round(a["acc"], 4), acc_std=round(a["std"], 4),
                       n_seeds=a["n"], train_seconds=round(a["seconds"], 2), rate_hz=round(a["rate_hz"], 2))
            if "bpc" in a:
                row.update(bpc_mean=round(a["bpc"], 4), bpc_std=round(a["bpc_std"], 4))
            rows.append(row)
    return rows


def write_csv(path: str, sweep: str, agg: Dict[str, Dict[object, dict]]) -> None:
    rows = table_rows(sweep, agg)
    cols = list(rows[0]) if rows else list(TABLE_COLUMNS)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")


# --------------------------------------------------------------------------- #
# Language-model sweep: spiking updates per token
# --------------------------------------------------------------------------- #
LM_LR_CANDIDATES = [1e-3, 2e-3, 4e-3, 8e-3]


def _lm_args(bud: Budget, k: int, seed: int) -> SimpleNamespace:
    """The settings run_training() and make_spiking() read, for one steps-per-token value."""
    return SimpleNamespace(
        seq_len=bud.lm_seq, weight_decay=0.0, warmup=50, min_lr_ratio=0.1, grad_clip=1.0,
        rate_target=0.05, rate_weight=bud.rate_weight, no_calibrate=False,
        d_emb=bud.lm_d_emb, n_rec=bud.lm_n_rec, exc_fraction=0.8, connectivity=0.1,
        neuron="alif", neuron_kwargs=None, steps_per_token=k, dropout=0.1, seed=seed)


def _lm_data(bud: Budget):
    """Character-level text split into train / validation / test streams (on DEVICE)."""
    text = load_text(bud.lm_data, "data")
    tok = CharTokenizer.from_text(text)
    train_ids, val_ids, test_ids = split_ids(torch.from_numpy(tok.encode(text)), 0.05, 0.05)
    T = bud.lm_seq
    vb, tb = max(1, min(32, len(val_ids) // (T + 1))), max(1, min(32, len(test_ids) // (T + 1)))
    streams = batchify(train_ids, bud.lm_batch, DEVICE)
    val, test = batchify(val_ids, vb, DEVICE), batchify(test_ids, tb, DEVICE)
    if min(streams.shape[1], val.shape[1], test.shape[1]) <= T + 1:
        raise ValueError("language-model text too short for lm_seq / lm_batch; lower them")
    return tok, streams, val, test


def lm_run(bud: Budget, k: int, lr: float, seed: int, data, steps: int, tune: bool = False) -> dict:
    """Train the spiking LM with ``k`` updates per token. Tuning runs return the validation
    loss; real runs return test accuracy / bits per character of the best-validation weights."""
    tok, streams, val, test = data
    args = _lm_args(bud, k, seed)
    torch.manual_seed(seed)
    model = make_spiking(args, tok.vocab_size, DEVICE, streams)      # moved to DEVICE, rate-calibrated
    hist, best = run_training(f"lm k={k}", "spiking", model, lr, steps, streams, val, args, DEVICE,
                              eval_every=steps if tune else max(1, steps // 4), log_every=10 ** 9,
                              eval_chunks=bud.lm_eval_chunks, seed=seed)
    if best is None or hist.get("diverged"):
        raise RuntimeError(f"language-model training diverged (k={k}, lr={lr:g})")
    if tune:
        return dict(val_loss=hist["val"][-1]["loss"])
    model.load_state_dict(best)
    ev = lm_evaluate("spiking", model, test, bud.lm_seq)
    return dict(acc=ev["acc"], bpc=ev["bpc"], seconds=hist["train_time"],
                rate_hz=ev["rate"] * 1000.0 / model.cfg.dt, gain=float("nan"))


def lm_learning_rate(bud: Budget, data, store: HPStore) -> float:
    """Tune the learning rate once, at k = 1, before the sweep (cached like the other searches)."""
    key = f"lm|steps|{bud.lm_data}|{bud.lm_n_rec}|{bud.lm_seq}|{bud.lm_batch}|{bud.lm_tune_steps}"
    if key in store.data:
        return store.data[key]["hp"]["lr"]
    trials = []
    for lr in ([2e-3, 4e-3] if bud.quick else LM_LR_CANDIDATES):
        try:
            v = lm_run(bud, 1, lr, 1000, data, bud.lm_tune_steps, tune=True)["val_loss"]
        except Exception:
            traceback.print_exc()
            FAILURES.append(f"lm tuning lr={lr:g}")
            v = float("inf")
        trials.append(dict(lr=lr, val_loss=v))
        print(f"[tune lm] lr {lr:g}: validation loss {v:.4f}", flush=True)
    best = min(trials, key=lambda t: t["val_loss"])
    store.put(key, dict(hp=dict(lr=best["lr"]), trials=trials))
    return best["lr"]


def run_steps_sweep(bud: Budget, store: HPStore) -> List[dict]:
    spec = SWEEPS["steps"]
    values = spec["quick"] if bud.quick else spec["values"]
    data = _lm_data(bud)
    lr = lm_learning_rate(bud, data, store)
    print(f"[steps] learning rate {lr:g}; {bud.lm_steps} steps of {bud.lm_batch * bud.lm_seq:,} tokens per run",
          flush=True)
    records: List[dict] = []
    for k in values:
        for s in range(bud.lm_seeds):
            try:
                r = lm_run(bud, k, lr, s, data, bud.lm_steps)
            except Exception:
                traceback.print_exc()
                FAILURES.append(f"steps k={k} seed={s}")
                r = dict(acc=float("nan"), bpc=float("nan"), seconds=float("nan"),
                         rate_hz=float("nan"), gain=float("nan"))
            records.append(dict(x=k, method="spiking", seed=s, **r))
            print(f"[steps] k={k} seed={s}: acc={r['acc']:.3f} bpc={r['bpc']:.3f} {r['seconds']:.0f}s", flush=True)
    return records


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
SHOW = False              # True in a notebook: draw figures inline instead of only saving


def _mpl():
    import matplotlib
    if not SHOW:
        matplotlib.use("Agg")      # headless; in a notebook keep the inline backend
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, plt, path):
    """Save to ``path`` (if given) and/or display inline (if SHOW)."""
    if path:
        fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    if SHOW:
        plt.show()
    plt.close(fig)


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


def _refs(ax, meta, x_left):
    """Reference lines: chance, and the ideal-observer ceiling when known."""
    ax.axhline(CHANCE, color=MUTED, linewidth=1.0, zorder=1)
    ax.text(x_left, CHANCE + 0.015, "chance", fontsize=7.5, color=MUTED, ha="left", va="bottom")
    ceiling = meta.get("oracle")
    if ceiling:
        ax.axhline(ceiling, color=MUTED, linewidth=1.0, zorder=1)
        ax.text(x_left, ceiling - 0.015, "ideal observer (reads the raw input)", fontsize=7.5,
                color=MUTED, ha="left", va="top")


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

    _refs(ax, meta, xs.min())

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
    _finish(fig, plt, path)


def _round_bar(ax, x, h, w, color, r_px=None):
    """Bar with a rounded data-end (4 css px) and a square baseline."""
    r_px = 4.0 * DPI / 100.0 if r_px is None else r_px
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
    gap = (2.0 * DPI / 100.0) * (ax.get_xlim()[1] - ax.get_xlim()[0]) / bb.width   # 2 css px
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
    _refs(ax, meta, ax.get_xlim()[0] + 0.02)
    handles = [Patch(facecolor=COLOR[m], edgecolor="none", label=METHOD_NAME[m]) for m in MAIN]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               labelcolor=SECONDARY, bbox_to_anchor=(0.5, 0.01), handlelength=1.2, columnspacing=1.6)
    _titles(fig, spec["title"], _subtitle(meta))
    _finish(fig, plt, path)


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
    _refs(ax, meta, ax.get_xlim()[0] + 0.02)
    top = max([s for s in sec if not np.isnan(s)] or [1.0]) * 1.25
    _bar_panel(fig, [0.58, 0.17, 0.40, 0.63], values, names, [("Training time", SINGLE, sec, [0] * len(sec))],
               (0.0, top), "Training time (s)", False, True, spec["xlabel"])
    _titles(fig, spec["title"], _subtitle(meta) + "  Right: wall-clock training time.")
    _finish(fig, plt, path)


def plot_steps(path, spec, agg, values, meta):
    """Language-model sweep: accuracy, bits per character and training time against the number
    of spiking updates per token (three panels with their own axes, never a dual axis)."""
    import numpy as np
    plt = _mpl()
    fig = plt.figure(figsize=(12.4, 4.4), dpi=DPI, facecolor=SURFACE)
    by_x = agg.get("spiking", {})
    xs = np.array(values, dtype=float)
    panels = [("acc", "std", "Test next-character accuracy", True, 0.07),
              ("bpc", "bpc_std", "Test bits / char (lower = better)", False, 0.40),
              ("seconds", None, "Training time (s)", False, 0.73)]
    for key, std_key, ylabel, pct, x0 in panels:
        mean = np.array([by_x.get(v, {}).get(key, np.nan) for v in values], dtype=float)
        std = np.array([by_x.get(v, {}).get(std_key, 0.0) if std_key else 0.0 for v in values], dtype=float)
        finite = np.isfinite(mean)
        lo = float((mean - std)[finite].min()) if finite.any() else 0.0
        hi = float((mean + std)[finite].max()) if finite.any() else 1.0
        pad = 0.08 * (hi - lo or 1.0)
        ylim = (0.0, hi + pad) if key == "seconds" else (max(0.0, lo - pad), hi + pad)
        ax = _new_axes(fig, [x0, 0.20, 0.25, 0.58], ylim, ylabel, pct=pct)
        ax.set_xlim(xs.min() - 0.4, xs.max() + 0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([_fmt(v) for v in values])
        ax.set_xlabel(spec["xlabel"], fontsize=8.5, color=SECONDARY, labelpad=6)
        ax.fill_between(xs, mean - std, mean + std, color=SINGLE, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(xs, mean, color=SINGLE, linewidth=1.8, solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    _titles(fig, spec["title"],
            f"Spiking E/I language model ({meta.get('lm_n_rec', '?')} adaptive-LIF neurons), "
            f"{meta.get('lm_steps', '?')} steps per run, mean ± s.d. over {meta.get('lm_seeds', '?')} seeds. "
            f"Each token drives the network for k updates; the k readouts are averaged.")
    _finish(fig, plt, path)


PLOTTERS = {"lines": plot_lines, "bars": plot_bars, "method": plot_method, "steps": plot_steps}


def make_outputs(out: Optional[str], name: str, records: List[dict], meta: dict) -> List[dict]:
    """Draw (and, if ``out`` is a directory, save) the figure for one sweep.
    With ``out=None`` nothing touches the disk: set SHOW=True to display it.
    Returns the table rows behind the figure."""
    spec = SWEEPS[name]
    values = spec["quick"] if meta["quick"] else spec["values"]
    agg = aggregate(records)
    png = os.path.join(out, spec["file"]) if out else None
    if out:
        write_csv(os.path.join(out, spec["file"].replace(".png", ".csv")), name, agg)
        print(f"wrote {spec['file']} (+ csv)", flush=True)
    PLOTTERS[spec["kind"]](png, spec, agg, values, meta)
    return table_rows(name, agg)


def preflight(bud: Budget, ceiling: float) -> None:
    """Cheap sanity check before the long run: can a *frozen* reservoir with a
    trained readout beat chance on this task? If not, every method that cannot
    build memory sits at chance and the comparison says nothing."""
    r = safe_run(Cfg(n_rec=bud.ref_n, method="reservoir"), {"lr_out": 1.0}, bud, seed=0, split="val")
    print(f"preflight: chance {CHANCE:.0%} | frozen reservoir {r['acc']:.0%} "
          f"at {r['rate_hz']:.1f} Hz | ideal observer {ceiling:.0%}", flush=True)
    if not r["acc"] > CHANCE + 0.08:
        print("WARNING: the frozen reservoir is at chance, so the input does not survive in "
              "its activity. Methods that cannot learn memory will show chance accuracy. "
              "Make the burst stronger (P_SIG), longer (BURST_LEN) or later (ONSET_*), or "
              "give the neurons slower dynamics.", file=sys.stderr, flush=True)
    if bud.quick:
        print("(quick budget: too few iterations for this number to be meaningful)")


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
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = ap.parse_args(argv)

    global DEVICE
    if args.device:
        DEVICE = torch.device(args.device)
    print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name(DEVICE)})" if DEVICE.type == "cuda" else ""))
    bud = QUICK if args.quick else FULL
    if args.seeds:
        bud = replace(bud, seeds=args.seeds)
    out = args.out or ("results_quick" if args.quick else "results")
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(out, "results.json")
    results = {"meta": {}, "sweeps": {}}
    if os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)
        if results.get("meta", {}).get("task") != TASK_SIG:
            if args.plots_only:
                print("warning: results.json was produced with a different task definition", file=sys.stderr)
            else:
                print(f"{res_path}: produced with a different task definition, discarding", flush=True)
                results = {"meta": {}, "sweeps": {}}
    names = args.only or list(SWEEPS)

    if not args.plots_only:
        results["meta"] = dict(asdict(bud), task=TASK_SIG, oracle=oracle_accuracy(n_test=bud.n_eval))
        preflight(bud, results["meta"]["oracle"])
        store = HPStore(os.path.join(out, "hparams.json"), retune=args.retune)
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

"""
Spiking language model vs a standard transformer on the same task.

    python compare_models.py                  # Tiny Shakespeare, spiking vs 2 transformers
    python compare_models.py --quick          # tiny smoke test (a few minutes)
    python compare_models.py --models spiking transformer_eff
    python compare_models.py --plots-only     # redraw from saved history.json

Models
------
spiking             the E/I recurrent spiking network of model.py (surrogate-gradient BPTT,
                    stateful truncated BPTT across chunks)
transformer_eff     standard GPT-style transformer (transformer_model.py), backprop, sized to
                    the spiking model's TRAINABLE parameters: the weights its sparsity masks
                    leave connected. This is the number of degrees of freedom the optimiser has.
transformer_stored  the same architecture sized to the spiking model's STORED parameters: every
                    element of its tensors, including the ~95% of the recurrent matrix that is
                    masked to zero and never changes.

Both transformer sizes are shown because "same number of parameters" is ambiguous for a
sparse model; the stored count is ~5x the trainable count here.

Fairness
--------
* Identical data, splits, chunking and chunk order (same seed), batch size, sequence length,
  optimiser (AdamW), warm-up + cosine schedule, gradient clipping and number of steps/tokens.
* The learning rate is tuned separately for each model (short validation runs) before the
  real runs; skip with --no-tune or fix with --lr-spiking / --lr-transformer.
* Every model is trained on --seeds different seeds (default 3: different initialisation, dropout
  and data order; the data order for a given seed is the same for every model, so runs are paired).
  The learning rate (and the spiking model's updates per token) are chosen once, at the base seed,
  and reused. Curves show the mean with a +-1 s.d. band; bars show the mean, the s.d. and each seed's
  result; the table reports the mean, s.d. and a 95% t-interval. With only 3 seeds that interval is
  wide (about 2.5 s.d.), which is the honest size of the uncertainty. Models can be added seed by
  seed: re-running with a larger --seeds only trains the missing runs.
* The best-validation weights of each run are evaluated on held-out TEST text.
* Training time is wall-clock time spent in training steps only (CUDA-synchronised; evaluation
  time is excluded).
* The spiking model's updates per token (--steps-per-token) are chosen automatically
  ("auto", the default): short validation runs of k = 1..--spt-max, same data and budget,
  and the k with the lowest validation loss wins (ties within --spt-tolerance go to the
  smaller k, since cost grows with k). The learning rate is then tuned at that k. The
  trials (loss and time per k) are saved in history.json, so the cost/benefit is visible.
  Pass an integer to fix k instead.
Caveat: the spiking network is stateful (its state carries across chunks, so its effective
context is unbounded); the transformer sees windows of --seq-len tokens, evaluated chunk by
chunk exactly as it is trained. That is the standard transformer setting, but it means the two
see different amounts of context.

Outputs (in --out): history.json, comparison.csv / comparison.md (the table view), and
fig_loss_vs_time.png, fig_final_comparison.png.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from model import LMConfig, SpikingLM
from train_model import CharTokenizer, batchify, load_text, lr_at, sample_text, split_ids
from transformer_model import TransformerConfig, TransformerLM, match_transformer, param_count

ORDER = ("spiking", "transformer_eff", "transformer_stored")
LABEL = {"spiking": "Spiking RSNN",
         "transformer_eff": "Transformer (matched trainable params)",
         "transformer_stored": "Transformer (matched stored params)"}
SHORT = {"spiking": "Spiking\nRSNN", "transformer_eff": "Transformer\n(trainable)",
         "transformer_stored": "Transformer\n(stored)"}
END_LABEL = {"spiking": "Spiking", "transformer_eff": "Transf. (trainable)",
             "transformer_stored": "Transf. (stored)"}      # short enough to fit beside a line end
# Categorical slots 1-3 of the validated palette; colour follows the model in every figure.
COLOR = {"spiking": "#2a78d6", "transformer_eff": "#eb6834", "transformer_stored": "#1baf7a"}
MARKER = {"spiking": "o", "transformer_eff": "s", "transformer_stored": "^"}
LR_CANDIDATES = {"spiking": [5e-4, 1e-3, 2e-3, 4e-3, 8e-3],
                 "transformer": [3e-4, 1e-3, 3e-3, 6e-3, 1e-2]}


# --------------------------------------------------------------------------- #
# Evaluation and training (shared by both model types)
# --------------------------------------------------------------------------- #
def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(kind: str, model, streams: torch.Tensor, T: int, max_chunks: Optional[int] = None) -> dict:
    """Held-out loss, bits per character, perplexity and next-character top-1 accuracy.
    The spiking model carries its state across chunks; the transformer sees each chunk alone."""
    model.eval()
    model.reset_state()
    losses, accs, rates = [], [], []
    for n, i in enumerate(range(0, streams.shape[1] - T, T)):
        if max_chunks is not None and n >= max_chunks:
            break
        x, y = streams[:, i:i + T], streams[:, i + 1:i + T + 1]
        if kind == "spiking":
            logits, spikes = model(x, return_spikes=True)
            rates.append(spikes.mean())
        else:
            logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))
        accs.append((logits.argmax(dim=-1) == y).float().mean())
    model.reset_state()
    model.train()
    loss = float(torch.stack(losses).mean())
    out = dict(loss=loss, bpc=loss / math.log(2), ppl=math.exp(min(loss, 20.0)),
               acc=float(torch.stack(accs).mean()))
    if rates:
        out["rate"] = float(torch.stack(rates).mean())
    return out


def run_training(name: str, kind: str, model, lr: float, total_steps: int, streams, val_streams, args,
                 device, *, eval_every: int, log_every: int, eval_chunks: Optional[int], seed: int):
    """One training run. Returns (history, best_state_dict). Identical data order for every
    model that uses the same seed."""
    T, B = args.seq_len, streams.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    warmup = min(args.warmup, max(1, total_steps // 10))
    rng = random.Random(seed)
    hist = dict(name=name, kind=kind, lr=lr, train=[], val=[])
    best_loss, best_state = float("inf"), None
    step, train_time, bad = 0, 0.0, 0
    run_ce, n_run = torch.zeros((), device=device), 0
    model.train()
    _sync(device)
    t_seg = time.time()
    done = False
    while not done:
        model.reset_state()
        data = streams[:, rng.randrange(T):]                      # same offsets for every model
        for i in range(0, data.shape[1] - T, T):
            if step >= total_steps:
                done = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, lr, warmup, args.min_lr_ratio)
            x, y = data[:, i:i + T], data[:, i + 1:i + T + 1]
            if kind == "spiking":
                logits, spikes = model(x, return_spikes=True)
                ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                rate = spikes.mean(dim=(0, 1))
                loss = ce + args.rate_weight * ((rate - args.rate_target) ** 2).mean()
            else:
                logits = model(x)
                ce = loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            if not torch.isfinite(loss):
                bad += 1
                print(f"[{name}] step {step}: non-finite loss, skipping ({bad} in a row)", file=sys.stderr)
                if bad >= 20:
                    hist["diverged"] = True
                    return hist, best_state
                opt.zero_grad(set_to_none=True)
                model.reset_state()
                continue
            bad = 0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if kind == "spiking":
                model.rsnn.project()                              # keep Dale's law
                model.detach_state()                              # truncate BPTT, keep the state
            step += 1
            run_ce += ce.detach()
            n_run += 1

            if step % log_every == 0:
                _sync(device)
                ce_m = float(run_ce) / n_run
                hist["train"].append(dict(step=step, tokens=step * B * T, time=train_time + time.time() - t_seg,
                                          loss=ce_m, bpc=ce_m / math.log(2)))
                print(f"[{name}] step {step:6d}/{total_steps} | train bpc {ce_m / math.log(2):.3f} | "
                      f"{hist['train'][-1]['time']:.0f}s", flush=True)
                run_ce, n_run = torch.zeros((), device=device), 0

            if step % eval_every == 0 or step == total_steps:
                _sync(device)
                train_time += time.time() - t_seg                 # the clock stops during evaluation
                ev = evaluate(kind, model, val_streams, T, eval_chunks)
                hist["val"].append(dict(step=step, tokens=step * B * T, time=train_time, **ev))
                print(f"[{name}]   validation: bpc {ev['bpc']:.3f} | acc {ev['acc']:.3f} | ppl {ev['ppl']:.2f}",
                      flush=True)
                if ev["loss"] < best_loss:
                    best_loss = ev["loss"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _sync(device)
                t_seg = time.time()
    _sync(device)
    train_time += time.time() - t_seg
    hist["train_time"] = train_time
    hist["tokens"] = step * B * T
    hist["tokens_per_s"] = hist["tokens"] / max(train_time, 1e-9)
    return hist, best_state


# --------------------------------------------------------------------------- #
# Model construction, parameter matching
# --------------------------------------------------------------------------- #
def make_spiking(args, vocab: int, device, streams=None) -> SpikingLM:
    cfg = LMConfig(vocab_size=vocab, d_emb=args.d_emb, n_rec=args.n_rec, exc_fraction=args.exc_fraction,
                   connectivity=args.connectivity, neuron=args.neuron, steps_per_token=args.steps_per_token,
                   dropout=args.dropout, seed=args.seed)
    if args.neuron_kwargs:
        cfg.neuron_kwargs = json.loads(args.neuron_kwargs)
    elif args.neuron != "alif":
        cfg.neuron_kwargs = {}
    model = SpikingLM(cfg).to(device)                             # move BEFORE building the optimiser
    if streams is not None and not args.no_calibrate:
        model.calibrate_rate(streams[:, :args.seq_len], args.rate_target)
    return model


def transformer_config(args, vocab: int, target: int) -> TransformerConfig:
    return match_transformer(target, vocab, args.seq_len, args.n_layer, args.n_head, args.dropout)


def _spt_arg(text: str):
    """argparse type for --steps-per-token: 'auto' or a positive integer."""
    if text == "auto":
        return text
    try:
        k = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected 'auto' or a positive integer")
    if k < 1:
        raise argparse.ArgumentTypeError("steps per token must be at least 1")
    return k


# everything that changes which k wins; a saved selection is only reused if all of it matches
_SPT_SIGNATURE_FIELDS = ("data", "n_rec", "d_emb", "exc_fraction", "connectivity", "neuron", "neuron_kwargs",
                         "seq_len", "batch_size", "dropout", "rate_target", "rate_weight", "seed", "spt_max",
                         "spt_steps", "tune_steps", "spt_seeds", "spt_lr", "spt_tolerance", "warmup",
                         "min_lr_ratio", "weight_decay", "grad_clip")


def spt_signature(args) -> dict:
    return {f: getattr(args, f, None) for f in _SPT_SIGNATURE_FIELDS}


def pick_steps_per_token(trials: List[dict], tolerance: float = 0.0) -> int:
    """The k with the lowest validation loss; any k within ``tolerance`` (relative) of the best
    loses to a smaller k, because each extra update per token costs proportionally more compute."""
    ok = [t for t in trials if math.isfinite(t["val_loss"])]
    if not ok:
        raise RuntimeError("every steps-per-token trial diverged; lower --spt-lr")
    best = min(t["val_loss"] for t in ok)
    return min(t["k"] for t in ok if t["val_loss"] <= best * (1.0 + tolerance))


def select_steps_per_token(args, vocab: int, device, streams, val_streams) -> dict:
    """Short validation runs for k = 1..spt_max (identical data and step budget; the learning rate
    is fixed at --spt-lr, then tuned afterwards at the chosen k). Returns the trials and the choice."""
    candidates = list(range(1, args.spt_max + 1))
    steps = args.spt_steps or args.tune_steps
    trials = []
    for k in candidates:
        losses, secs = [], []
        for sd in range(args.spt_seeds):
            a = copy.copy(args)
            a.steps_per_token, a.seed = k, args.seed + sd
            torch.manual_seed(a.seed)
            model = make_spiking(a, vocab, device, streams)
            h, _ = run_training(f"steps/token k={k}", "spiking", model, args.spt_lr, steps, streams, val_streams,
                                a, device, eval_every=steps, log_every=10 ** 9, eval_chunks=args.eval_chunks,
                                seed=a.seed)
            diverged = h.get("diverged") or not h["val"]
            losses.append(float("inf") if diverged else h["val"][-1]["loss"])
            secs.append(h.get("train_time", float("nan")))
            del model
        loss = sum(losses) / len(losses)
        trials.append(dict(k=k, val_loss=loss, val_bpc=loss / math.log(2), seconds=sum(secs) / len(secs)))
        print(f"[steps/token] k={k}: validation bpc {loss / math.log(2):.3f} | {trials[-1]['seconds']:.0f}s",
              flush=True)
    k = pick_steps_per_token(trials, args.spt_tolerance)
    print(f"[steps/token] selected k = {k}", flush=True)
    return dict(k=k, trials=trials, signature=spt_signature(args))


# --------------------------------------------------------------------------- #
# Figures (shared style with tests.py: validated palette, legend + direct labels, table view)
# --------------------------------------------------------------------------- #
# two-sided 95% critical values of Student's t by degrees of freedom (n seeds -> df = n - 1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def _stats(values) -> tuple:
    """(mean, sample s.d., 95% half-width, n) ignoring missing / non-finite values. With one value the
    s.d. is 0 and the interval is undefined (NaN)."""
    v = [x for x in values if x is not None and math.isfinite(x)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = sum(v) / n
    if n == 1:
        return mean, 0.0, float("nan"), 1
    sd = statistics.stdev(v)
    df = n - 1
    t = _T95[max(k for k in _T95 if k <= df)] if df <= 30 else 1.96
    return mean, sd, t * sd / math.sqrt(n), n


def _runs(entry: dict, default_seed: int = 0) -> List[dict]:
    """The per-seed runs of a model entry. Also reads the older single-run layout."""
    if "runs" in entry:
        return entry["runs"]
    return [dict(seed=entry.get("seed", default_seed), hist=entry["hist"], final=entry["final"],
                 trainable_params=entry["trainable_params"], stored_params=entry["stored_params"])]


def plan_seeds(entry: Optional[dict], seeds: List[int], default_seed: int = 0) -> tuple:
    """(existing runs, seeds still to train). Lets a bigger --seeds extend a finished model."""
    runs = _runs(entry, default_seed) if entry else []
    done = {r["seed"] for r in runs}
    return runs, [sd for sd in seeds if sd not in done]


def _curve(entry: dict, which: str) -> Optional[dict]:
    """Mean and s.d. across seeds of a logged curve ('val' or 'train'). Every seed logs on the same step
    grid, so curves are aligned by index (times are averaged too)."""
    import numpy as np
    curves = [r["hist"][which] for r in _runs(entry) if r["hist"].get(which)]
    if not curves:
        return None
    n = min(len(c) for c in curves)
    arr = lambda key: np.array([[e[key] for e in c[:n]] for c in curves], dtype=float)
    bpc = arr("bpc")
    return dict(time=arr("time").mean(0), tokens=arr("tokens").mean(0), bpc=bpc.mean(0),
                std=bpc.std(0, ddof=1) if len(curves) > 1 else np.zeros(n), n=len(curves))


def _seed_text(history: dict, models: List[str]) -> str:
    ns = sorted({len(_runs(history["models"][m])) for m in models})
    return "/".join(str(n) for n in ns) + (" seed" if ns == [1] else " seeds")


def _vis():
    import tests                                                  # plotting helpers live there
    return tests


def _line_panel(fig, rect, vis, curves, xlabel, ylabel, xlog=False, end_labels=False):
    """curves: [(model, xs, ys, with_markers[, std])]; a std draws a +-1 s.d. band. One axis per panel
    (never dual-axis)."""
    import numpy as np
    curves = [tuple(c) + (None,) * (5 - len(c)) for c in curves]
    ys_all = [y for _, _, ys, _, sd in curves
              for y in (list(np.asarray(ys) - np.asarray(sd)) + list(np.asarray(ys) + np.asarray(sd))
                        if sd is not None else ys) if np.isfinite(y)]
    lo, hi = (min(ys_all), max(ys_all)) if ys_all else (0.0, 1.0)
    pad = 0.06 * (hi - lo or 1.0)
    ax = vis._new_axes(fig, rect, (max(0.0, lo - pad), hi + pad), ylabel, pct=False)
    xs_all = [x for _, xs, _, _, _ in curves for x in xs if np.isfinite(x) and x > 0]
    if xlog:
        ax.set_xscale("log")
    for name, xs, ys, markers, sd in curves:
        if sd is not None and np.any(np.asarray(sd) > 0):
            ax.fill_between(xs, np.asarray(ys) - np.asarray(sd), np.asarray(ys) + np.asarray(sd),
                            color=COLOR[name], alpha=0.10, linewidth=0, zorder=2)
        ax.plot(xs, ys, color=COLOR[name], linewidth=1.8, solid_capstyle="round", solid_joinstyle="round",
                marker=MARKER[name] if markers else None, markersize=7, markeredgecolor=vis.SURFACE,
                markeredgewidth=1.5, zorder=3)
    if xs_all:
        x_min, x_max = min(xs_all), max(xs_all)
        if xlog:
            ax.set_xlim(x_min / 1.1, x_max * 1.15)
        else:                                          # room on the right for the end labels
            ax.set_xlim(0.0, x_max + (0.5 if end_labels else 0.05) * (x_max - x_min))
    ends = sorted((ys[-1], name, xs[-1]) for name, xs, ys, _, _ in curves if len(ys))
    if end_labels and len(ends) > 1 and min(b[0] - a[0] for a, b in zip(ends, ends[1:])) >= 0.1 * (hi - lo or 1.0):
        for y, name, x in ends:
            ax.annotate(END_LABEL[name], (x, y), xytext=(8, 0), textcoords="offset points",
                        fontsize=7.5, color=vis.PRIMARY, va="center", ha="left")
    ax.set_xlabel(xlabel, fontsize=8.5, color=vis.SECONDARY, labelpad=6)
    return ax


def _label(history: dict, m: str) -> str:
    k = history["models"][m].get("steps_per_token")
    return f"{LABEL[m]}, {k} updates/token" if k else LABEL[m]


def _short(history: dict, m: str) -> str:
    k = history["models"][m].get("steps_per_token")
    return SHORT[m].replace("RSNN", f"RSNN (k={k})") if k else SHORT[m]


def plot_loss_curves(history: dict, path: Optional[str]) -> None:
    from matplotlib.lines import Line2D
    vis = _vis()
    plt = vis._mpl()
    models = [m for m in ORDER if m in history["models"]]
    fig = plt.figure(figsize=(13.2, 4.7), dpi=vis.DPI, facecolor=vis.SURFACE)
    val = {m: _curve(history["models"][m], "val") for m in models}
    trn = {m: _curve(history["models"][m], "train") for m in models}
    _line_panel(fig, [0.055, 0.25, 0.27, 0.56], vis,
                [(m, val[m]["time"], val[m]["bpc"], True, val[m]["std"]) for m in models if val[m]],
                "Training time (s, log scale)", "Validation bits / char (lower = better)", xlog=True)
    _line_panel(fig, [0.385, 0.25, 0.27, 0.56], vis,
                [(m, val[m]["tokens"] / 1e6, val[m]["bpc"], True, val[m]["std"]) for m in models if val[m]],
                "Training tokens seen (millions)", "Validation bits / char", end_labels=True)
    _line_panel(fig, [0.75, 0.25, 0.235, 0.56], vis,
                [(m, trn[m]["time"], trn[m]["bpc"], False, trn[m]["std"]) for m in models if trn[m]],
                "Training time (s, log scale)", "Training bits / char", xlog=True)
    handles = [Line2D([0], [0], color=COLOR[m], linewidth=1.8, marker=MARKER[m], markersize=7,
                      markeredgecolor=vis.SURFACE, markeredgewidth=1.5, label=_label(history, m)) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), frameon=False, fontsize=8.5,
               labelcolor=vis.SECONDARY, bbox_to_anchor=(0.5, 0.01), handlelength=2.2, columnspacing=2.0)
    a = history["args"]
    vis._titles(fig, "Spiking RSNN vs transformer: loss against training time and data",
                f"{a['data']}, same optimiser and schedule for every model; {a['seq_len']}-token chunks, "
                f"batch {a['batch_size']}. Line = mean over {_seed_text(history, models)}, band = +-1 s.d., "
                f"markers = validation checkpoints.")
    vis._finish(fig, plt, path)


def _bar_panel(fig, rect, vis, models, values, ylim, ylabel, fmt, pct=False, names=None, stds=None,
               points=None):
    import numpy as np
    ax = vis._new_axes(fig, rect, ylim, ylabel, pct=pct)
    n = len(models)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names or [SHORT[m] for m in models])
    bb = ax.get_window_extent()
    width = (24.0 * vis.DPI / 100.0) * (n + 0.2) / bb.width       # bars are at most 24 css px wide
    span = ylim[1] - ylim[0]
    for i, (m, v) in enumerate(zip(models, values)):
        if v is None or not np.isfinite(v):
            continue
        vis._round_bar(ax, i, v, width, COLOR[m])
        top = v
        sd = stds[i] if stds is not None else None
        if sd is not None and np.isfinite(sd) and sd > 0:                 # +-1 s.d. across seeds
            ax.plot([i, i], [max(v - sd, 0.0), v + sd], color=vis.SECONDARY, linewidth=1.0,
                    solid_capstyle="butt", zorder=4)
            top = v + sd
        pts = [x for x in (points[i] if points is not None else []) if x is not None and np.isfinite(x)]
        if len(pts) > 1:                                                  # one dot per seed (the raw data)
            xs = i + (np.arange(len(pts)) - (len(pts) - 1) / 2) * (width * 0.28)
            ax.plot(xs, pts, linestyle="none", marker="o", markersize=5, markerfacecolor=vis.SURFACE,
                    markeredgecolor=vis.PRIMARY, markeredgewidth=0.9, zorder=6)
            top = max(top, max(pts))
        ax.text(i, top + 0.015 * span, fmt(v), fontsize=8, color=vis.PRIMARY, ha="center", va="bottom", zorder=5)
    return ax


def plot_final(history: dict, path: Optional[str]) -> None:
    vis = _vis()
    plt = vis._mpl()
    models = [m for m in ORDER if m in history["models"]]
    fig = plt.figure(figsize=(11.6, 4.5), dpi=vis.DPI, facecolor=vis.SURFACE)
    names = [_short(history, m) for m in models]
    runs = {m: _runs(history["models"][m]) for m in models}
    per_run = {"acc": [[r["final"]["test"]["acc"] for r in runs[m]] for m in models],
               "bpc": [[r["final"]["test"]["bpc"] for r in runs[m]] for m in models],
               "sec": [[r["hist"].get("train_time") for r in runs[m]] for m in models]}
    stat = {k: [_stats(v) for v in vals] for k, vals in per_run.items()}

    def top(key):                                   # axis headroom for the mean + s.d. and every dot
        highs = [max([mean + (sd if math.isfinite(sd) else 0.0)] + [x for x in pts if x is not None])
                 for (mean, sd, _, _), pts in zip(stat[key], per_run[key]) if math.isfinite(mean)]
        return max(highs) * 1.22

    for key, rect, ylim, ylabel, fmt, pct in (
            ("acc", [0.06, 0.2, 0.26, 0.58], (0.0, min(1.0, top("acc"))), "Test next-character accuracy",
             lambda v: f"{v:.1%}", True),
            ("bpc", [0.40, 0.2, 0.26, 0.58], (0.0, top("bpc")), "Test bits / char (lower = better)",
             lambda v: f"{v:.2f}", False),
            ("sec", [0.74, 0.2, 0.24, 0.58], (0.0, top("sec")), "Total training time (s)",
             lambda v: f"{v:.0f}s", False)):
        _bar_panel(fig, rect, vis, models, [m_ for m_, _, _, _ in stat[key]], ylim, ylabel, fmt, pct=pct,
                   names=names, stds=[sd for _, sd, _, _ in stat[key]], points=per_run[key])
    p = history["params"]
    vis._titles(fig, "Final comparison on held-out text",
                f"Trainable parameters: spiking {p['spiking_trainable']:,} (stored {p['spiking_stored']:,}); "
                f"transformers {p.get('transformer_eff', 0):,} / {p.get('transformer_stored', 0):,}. "
                f"Chance accuracy = 1/{history['vocab_size']}. Bars = mean over {_seed_text(history, models)}, "
                f"whisker = +-1 s.d., dots = individual seeds.")
    vis._finish(fig, plt, path)


def _r(x, nd: int):
    return round(x, nd) if x is not None and math.isfinite(x) else "-"


def comparison_rows(history: dict) -> List[dict]:
    """One row per model, aggregated over its seeds (mean, s.d. and 95% t-interval half-width)."""
    rows = []
    for m in [m for m in ORDER if m in history["models"]]:
        e = history["models"][m]
        runs = _runs(e)
        col = lambda f: [f(r) for r in runs]
        acc = _stats(col(lambda r: r["final"]["test"]["acc"]))
        bpc = _stats(col(lambda r: r["final"]["test"]["bpc"]))
        secs = _stats(col(lambda r: r["hist"].get("train_time")))
        rows.append(dict(
            model=LABEL[m], steps_per_token=e.get("steps_per_token") or "-", lr=e["lr"], n_seeds=len(runs),
            trainable_params=round(_stats(col(lambda r: r["trainable_params"]))[0]),
            stored_params=round(_stats(col(lambda r: r["stored_params"]))[0]),
            tokens=runs[0]["hist"].get("tokens"),
            train_seconds=_r(secs[0], 1), train_seconds_std=_r(secs[1], 1),
            tokens_per_s=_r(_stats(col(lambda r: r["hist"].get("tokens_per_s")))[0], 0),
            best_val_bpc=_r(_stats(col(lambda r: r["final"]["val"]["bpc"]))[0], 4),
            test_bpc=_r(bpc[0], 4), test_bpc_std=_r(bpc[1], 4), test_bpc_ci95=_r(bpc[2], 4),
            test_perplexity=_r(_stats(col(lambda r: r["final"]["test"]["ppl"]))[0], 3),
            test_accuracy=_r(acc[0], 4), test_accuracy_std=_r(acc[1], 4), test_accuracy_ci95=_r(acc[2], 4)))
    return rows


def write_table(history: dict, out: str) -> List[dict]:
    rows = comparison_rows(history)
    if not rows:
        return rows
    with open(os.path.join(out, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    with open(os.path.join(out, "comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md) + "\n")
    return rows


def make_outputs(history: dict, out: Optional[str]) -> None:
    if not history["models"]:
        print("no results to plot")
        return
    if out:
        write_table(history, out)
    plot_loss_curves(history, os.path.join(out, "fig_loss_vs_time.png") if out else None)
    plot_final(history, os.path.join(out, "fig_final_comparison.png") if out else None)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="tinyshakespeare", help="builtin | tinyshakespeare | wikitext2 | path")
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--models", nargs="+", choices=ORDER, default=list(ORDER))
    ap.add_argument("--out", default="results_compare")
    s = ap.add_argument_group("spiking model (same defaults as train_model.py)")
    s.add_argument("--n-rec", type=int, default=1024)
    s.add_argument("--d-emb", type=int, default=128)
    s.add_argument("--exc-fraction", type=float, default=0.8)
    s.add_argument("--connectivity", type=float, default=0.05)
    s.add_argument("--neuron", default="alif", choices=["if", "lif", "alif", "stochastic_lif", "izhikevich"])
    s.add_argument("--neuron-kwargs", default=None)
    s.add_argument("--steps-per-token", type=_spt_arg, default="auto",
                   help="spiking updates per token: an integer, or 'auto' (default) to pick the best 1..--spt-max")
    s.add_argument("--spt-max", type=int, default=8, help="largest k tried by 'auto'")
    s.add_argument("--spt-steps", type=int, default=None, help="length of each trial (default: --tune-steps)")
    s.add_argument("--spt-seeds", type=int, default=1, help="seeds per k (more = less noisy choice, slower)")
    s.add_argument("--spt-lr", type=float, default=2e-3, help="learning rate used while choosing k")
    s.add_argument("--spt-tolerance", type=float, default=0.0,
                   help="prefer a smaller k whose validation loss is within this relative margin of the best")
    s.add_argument("--rate-target", type=float, default=0.05)
    s.add_argument("--rate-weight", type=float, default=50.0)
    s.add_argument("--no-calibrate", action="store_true")
    t = ap.add_argument_group("transformer")
    t.add_argument("--n-layer", type=int, default=4)
    t.add_argument("--n-head", type=int, default=4)
    o = ap.add_argument_group("optimisation (identical for every model)")
    o.add_argument("--seq-len", type=int, default=128)
    o.add_argument("--batch-size", type=int, default=64)
    o.add_argument("--epochs", type=int, default=10)
    o.add_argument("--max-steps", type=int, default=None)
    o.add_argument("--dropout", type=float, default=0.1)
    o.add_argument("--warmup", type=int, default=100)
    o.add_argument("--min-lr-ratio", type=float, default=0.1)
    o.add_argument("--weight-decay", type=float, default=0.0)
    o.add_argument("--grad-clip", type=float, default=1.0)
    o.add_argument("--lr-spiking", type=float, default=None, help="skip tuning, use this learning rate")
    o.add_argument("--lr-transformer", type=float, default=None)
    o.add_argument("--no-tune", action="store_true", help="use default learning rates (2e-3 / 3e-3)")
    o.add_argument("--tune-steps", type=int, default=150, help="length of each learning-rate trial")
    r = ap.add_argument_group("running")
    r.add_argument("--eval-every", type=int, default=100)
    r.add_argument("--eval-batch", type=int, default=32)
    r.add_argument("--eval-chunks", type=int, default=40)
    r.add_argument("--log-every", type=int, default=25)
    r.add_argument("--seed", type=int, default=0, help="base seed; runs use seed, seed+1, ...")
    r.add_argument("--seeds", type=int, default=3, help="training runs per model (different seeds)")
    r.add_argument("--device", default=None)
    r.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--quick", action="store_true")
    r.add_argument("--plots-only", action="store_true")
    r.add_argument("--force", action="store_true", help="retrain models already in history.json")
    s2 = ap.add_argument_group("samples")
    s2.add_argument("--prompt", default="\n")
    s2.add_argument("--sample-len", type=int, default=240)
    s2.add_argument("--temperature", type=float, default=0.8)
    s2.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args(argv)
    if args.quick:
        args.data, args.n_rec, args.d_emb, args.seq_len, args.batch_size = "builtin", 128, 32, 32, 16
        args.max_steps, args.eval_every, args.eval_chunks, args.log_every = 30, 10, 3, 10
        args.tune_steps, args.warmup, args.sample_len = 10, 5, 80
        args.spt_max, args.spt_steps, args.seeds = 2, 10, min(args.seeds, 2)
        LR_CANDIDATES["spiking"], LR_CANDIDATES["transformer"] = [1e-3, 4e-3], [1e-3, 6e-3]
        args.out = args.out if args.out != "results_compare" else "results_compare_quick"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    hist_path = os.path.join(args.out, "history.json")
    os.makedirs(args.out, exist_ok=True)
    history = {"models": {}}
    if os.path.exists(hist_path):
        history = json.load(open(hist_path))
    if args.plots_only:
        make_outputs(history, args.out)
        return 0

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    # ---- data (identical for every model) ---------------------------------- #
    text = load_text(args.data, args.cache_dir)
    tok = CharTokenizer.from_text(text)
    train_ids, val_ids, test_ids = split_ids(torch.from_numpy(tok.encode(text)), 0.05, 0.05)
    T, B = args.seq_len, args.batch_size
    streams = batchify(train_ids, B, device)
    val_b = max(1, min(args.eval_batch, len(val_ids) // (T + 1)))
    test_b = max(1, min(args.eval_batch, len(test_ids) // (T + 1)))
    val_streams, test_streams = batchify(val_ids, val_b, device), batchify(test_ids, test_b, device)
    if min(streams.shape[1], val_streams.shape[1], test_streams.shape[1]) <= T + 1:
        raise SystemExit("text too short for this --seq-len / --batch-size; lower them")
    total_steps = args.max_steps or args.epochs * len(range(0, streams.shape[1] - T, T))
    V = tok.vocab_size
    print(f"data: {len(text):,} chars, vocab {V} | {total_steps} steps x {B * T:,} tokens "
          f"= {total_steps * B * T / 1e6:.1f}M training tokens per model")

    # ---- parameter matching --------------------------------------------------- #
    # ---- steps per token: fixed, or chosen by validation ---------------------------------- #
    spiking_pending = "spiking" in args.models and ("spiking" not in history["models"] or args.force)
    if args.steps_per_token == "auto":
        if spiking_pending:
            sel = history.get("spt_selection")
            if sel is None or sel.get("signature") != spt_signature(args) or args.force:
                sel = select_steps_per_token(args, V, device, streams, val_streams)
                history["spt_selection"] = sel
                json.dump(history, open(hist_path, "w"), indent=1)      # resume point
            else:
                print(f"[steps/token] reusing saved selection: k = {sel['k']}")
            args.steps_per_token = sel["k"]
        else:                                                           # spiking already trained
            args.steps_per_token = history["models"].get("spiking", {}).get("steps_per_token") or 1
    print(f"spiking updates per token: {args.steps_per_token}")

    torch.manual_seed(args.seed)
    ref = make_spiking(args, V, torch.device("cpu"))              # only for the counts
    p_train, p_stored = ref.num_effective_parameters, ref.num_parameters
    del ref
    tcfg = {"transformer_eff": transformer_config(args, V, p_train),
            "transformer_stored": transformer_config(args, V, p_stored)}
    history.update(args={k: v for k, v in vars(args).items()}, vocab_size=V,
                   params=dict(spiking_trainable=p_train, spiking_stored=p_stored,
                               **{k: param_count(c) for k, c in tcfg.items()}),
                   transformer_cfg={k: asdict(c) for k, c in tcfg.items()})
    print(f"\nspiking: {p_train:,} trainable / {p_stored:,} stored parameters (sparsity mask keeps "
          f"{100 * p_train / p_stored:.0f}%)")
    for k, c in tcfg.items():
        target = p_train if k == "transformer_eff" else p_stored
        print(f"{k}: d_model {c.d_model}, {c.n_layer} layers, {c.n_head} heads, d_ff {c.d_ff}: "
              f"{param_count(c):,} parameters ({100 * (param_count(c) - target) / target:+.2f}% vs target)")

    def builder(name, seed=None):
        seed = args.seed if seed is None else seed
        if name == "spiking":
            a = copy.copy(args)
            a.seed = seed                                         # network masks / initial weights
            return lambda: make_spiking(a, V, device, streams)
        return lambda: TransformerLM(tcfg[name]).to(device)       # initialised from the global seed

    seeds = [args.seed + i for i in range(args.seeds)]

    # ---- train each model on every seed ---------------------------------------------- #
    for name in [m for m in ORDER if m in args.models]:
        kind = "spiking" if name == "spiking" else "transformer"
        entry = None if args.force else history["models"].get(name)
        runs, todo = plan_seeds(entry, seeds, history.get("args", {}).get("seed", args.seed))
        if entry is not None:
            if not todo:
                print(f"[{name}] already has seeds {seeds} in history.json, skipping (--force to retrain)")
                continue
            entry["runs"] = runs                                  # moves a legacy single run into 'runs'
            for legacy in ("hist", "final"):
                entry.pop(legacy, None)
            lr = entry["lr"]
            print(f"[{name}] adding seeds {todo} to {len(runs)} existing run(s), lr {lr:g}")
        else:
            lr = args.lr_spiking if kind == "spiking" else args.lr_transformer
            tuning = None
            if lr is None and args.no_tune:
                lr = 2e-3 if kind == "spiking" else 3e-3
            elif lr is None:
                tuning = []
                for cand in LR_CANDIDATES[kind]:
                    torch.manual_seed(args.seed)
                    h, _ = run_training(f"tune {name} lr={cand:g}", kind, builder(name)(), cand, args.tune_steps,
                                        streams, val_streams, args, device, eval_every=args.tune_steps,
                                        log_every=10 ** 9, eval_chunks=args.eval_chunks, seed=args.seed)
                    v = h["val"][-1]["loss"] if h["val"] and not h.get("diverged") else float("inf")
                    tuning.append(dict(lr=cand, val_loss=v))
                    print(f"[tune {name}] lr {cand:g}: validation loss {v:.4f}", flush=True)
                lr = min(tuning, key=lambda r: r["val_loss"])["lr"]
                print(f"[{name}] selected lr {lr:g}", flush=True)
            entry = dict(lr=lr, tuning=tuning, runs=[], sample=None,
                         steps_per_token=args.steps_per_token if name == "spiking" else None)

        for sd in todo:
            torch.manual_seed(sd)
            model = builder(name, sd)()
            trainable = model.num_effective_parameters if name == "spiking" else model.num_parameters
            stored = model.num_parameters
            hist, best = run_training(f"{name} seed {sd}", kind, model, lr, total_steps, streams, val_streams,
                                      args, device, eval_every=args.eval_every, log_every=args.log_every,
                                      eval_chunks=args.eval_chunks, seed=sd)
            if best is None:
                print(f"[{name}] seed {sd} produced no usable checkpoint (diverged?)", file=sys.stderr)
                continue
            model.load_state_dict(best)                           # best-validation weights
            final = dict(val=evaluate(kind, model, val_streams, T), test=evaluate(kind, model, test_streams, T))
            print(f"[{name}] seed {sd} TEST: bpc {final['test']['bpc']:.3f} | ppl {final['test']['ppl']:.2f} | "
                  f"next-char accuracy {final['test']['acc']:.1%}")
            if entry.get("sample") is None:
                entry["sample"] = sample_text(model, tok, args.prompt, args.sample_len, args.temperature,
                                              args.top_k, device)
            entry["runs"].append(dict(seed=sd, hist=hist, final=final, trainable_params=trainable,
                                      stored_params=stored))
            entry["trainable_params"], entry["stored_params"] = entry["runs"][0]["trainable_params"], \
                entry["runs"][0]["stored_params"]
            history["models"][name] = entry
            json.dump(history, open(hist_path, "w"), indent=1)    # resume point after every run
            del model, best
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for name, e in history["models"].items():
        print(f"\n--- {LABEL[name]} sample (first seed) ---\n{e.get('sample', '')}")
    make_outputs(history, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

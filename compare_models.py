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
* The best-validation weights of each model are evaluated on held-out TEST text.
* Training time is wall-clock time spent in training steps only (CUDA-synchronised; evaluation
  time is excluded).
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


# --------------------------------------------------------------------------- #
# Figures (shared style with tests.py: validated palette, legend + direct labels, table view)
# --------------------------------------------------------------------------- #
def _vis():
    import tests                                                  # plotting helpers live there
    return tests


def _line_panel(fig, rect, vis, curves, xlabel, ylabel, xlog=False, end_labels=False):
    """curves: [(model, xs, ys, with_markers)]. One axis per panel (never dual-axis)."""
    import numpy as np
    ys_all = [y for _, _, ys, _ in curves for y in ys if np.isfinite(y)]
    lo, hi = (min(ys_all), max(ys_all)) if ys_all else (0.0, 1.0)
    pad = 0.06 * (hi - lo or 1.0)
    ax = vis._new_axes(fig, rect, (max(0.0, lo - pad), hi + pad), ylabel, pct=False)
    xs_all = [x for _, xs, _, _ in curves for x in xs if np.isfinite(x) and x > 0]
    if xlog:
        ax.set_xscale("log")
    for name, xs, ys, markers in curves:
        ax.plot(xs, ys, color=COLOR[name], linewidth=1.8, solid_capstyle="round", solid_joinstyle="round",
                marker=MARKER[name] if markers else None, markersize=7, markeredgecolor=vis.SURFACE,
                markeredgewidth=1.5, zorder=3)
    if xs_all:
        x_min, x_max = min(xs_all), max(xs_all)
        if xlog:
            ax.set_xlim(x_min / 1.1, x_max * 1.15)
        else:                                          # room on the right for the end labels
            ax.set_xlim(0.0, x_max + 0.25 * (x_max - x_min))
    ends = sorted((ys[-1], name, xs[-1]) for name, xs, ys, _ in curves if len(ys))
    if end_labels and len(ends) > 1 and min(b[0] - a[0] for a, b in zip(ends, ends[1:])) >= 0.1 * (hi - lo or 1.0):
        for y, name, x in ends:
            ax.annotate(SHORT[name].replace("\n", " "), (x, y), xytext=(8, 0), textcoords="offset points",
                        fontsize=7.5, color=vis.PRIMARY, va="center", ha="left")
    ax.set_xlabel(xlabel, fontsize=8.5, color=vis.SECONDARY, labelpad=6)
    return ax


def plot_loss_curves(history: dict, path: Optional[str]) -> None:
    from matplotlib.lines import Line2D
    vis = _vis()
    plt = vis._mpl()
    models = [m for m in ORDER if m in history["models"]]
    fig = plt.figure(figsize=(13.2, 4.7), dpi=vis.DPI, facecolor=vis.SURFACE)
    val = {m: history["models"][m]["hist"]["val"] for m in models}
    trn = {m: history["models"][m]["hist"]["train"] for m in models}
    _line_panel(fig, [0.055, 0.25, 0.27, 0.56], vis,
                [(m, [e["time"] for e in val[m]], [e["bpc"] for e in val[m]], True) for m in models],
                "Training time (s, log scale)", "Validation bits / char (lower = better)", xlog=True)
    _line_panel(fig, [0.385, 0.25, 0.27, 0.56], vis,
                [(m, [e["tokens"] / 1e6 for e in val[m]], [e["bpc"] for e in val[m]], True) for m in models],
                "Training tokens seen (millions)", "Validation bits / char", end_labels=True)
    _line_panel(fig, [0.715, 0.25, 0.27, 0.56], vis,
                [(m, [e["time"] for e in trn[m]], [e["bpc"] for e in trn[m]], False) for m in models],
                "Training time (s, log scale)", "Training bits / char", xlog=True)
    handles = [Line2D([0], [0], color=COLOR[m], linewidth=1.8, marker=MARKER[m], markersize=7,
                      markeredgecolor=vis.SURFACE, markeredgewidth=1.5, label=LABEL[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), frameon=False, fontsize=8.5,
               labelcolor=vis.SECONDARY, bbox_to_anchor=(0.5, 0.01), handlelength=2.2, columnspacing=2.0)
    a = history["args"]
    vis._titles(fig, "Spiking RSNN vs transformer: loss against training time and data",
                f"{a['data']}, same data order, optimiser and schedule for every model; "
                f"{a['seq_len']}-token chunks, batch {a['batch_size']}. Markers = validation checkpoints.")
    vis._finish(fig, plt, path)


def _bar_panel(fig, rect, vis, models, values, ylim, ylabel, fmt, pct=False):
    import numpy as np
    ax = vis._new_axes(fig, rect, ylim, ylabel, pct=pct)
    n = len(models)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(range(n))
    ax.set_xticklabels([SHORT[m] for m in models])
    bb = ax.get_window_extent()
    width = (24.0 * vis.DPI / 100.0) * (n + 0.2) / bb.width       # bars are at most 24 css px wide
    span = ylim[1] - ylim[0]
    for i, (m, v) in enumerate(zip(models, values)):
        if v is None or not np.isfinite(v):
            continue
        vis._round_bar(ax, i, v, width, COLOR[m])
        ax.text(i, v + 0.015 * span, fmt(v), fontsize=8, color=vis.PRIMARY, ha="center", va="bottom", zorder=5)
    return ax


def plot_final(history: dict, path: Optional[str]) -> None:
    vis = _vis()
    plt = vis._mpl()
    models = [m for m in ORDER if m in history["models"]]
    fin = {m: history["models"][m]["final"] for m in models}
    fig = plt.figure(figsize=(11.6, 4.5), dpi=vis.DPI, facecolor=vis.SURFACE)
    acc = [fin[m]["test"]["acc"] for m in models]
    bpc = [fin[m]["test"]["bpc"] for m in models]
    sec = [history["models"][m]["hist"].get("train_time") for m in models]
    _bar_panel(fig, [0.06, 0.2, 0.26, 0.58], vis, models, acc, (0.0, min(1.0, max(acc) * 1.25)),
               "Test next-character accuracy", lambda v: f"{v:.1%}", pct=True)
    _bar_panel(fig, [0.40, 0.2, 0.26, 0.58], vis, models, bpc, (0.0, max(bpc) * 1.2),
               "Test bits / char (lower = better)", lambda v: f"{v:.2f}")
    _bar_panel(fig, [0.74, 0.2, 0.24, 0.58], vis, models, sec, (0.0, max(s for s in sec if s) * 1.2),
               "Total training time (s)", lambda v: f"{v:.0f}s")
    p = history["params"]
    vis._titles(fig, "Final comparison on held-out text",
                f"Trainable parameters: spiking {p['spiking_trainable']:,} (stored {p['spiking_stored']:,}); "
                f"transformers {p.get('transformer_eff', 0):,} / {p.get('transformer_stored', 0):,}. "
                f"Chance accuracy = 1/{history['vocab_size']}.")
    vis._finish(fig, plt, path)


def comparison_rows(history: dict) -> List[dict]:
    rows = []
    for m in [m for m in ORDER if m in history["models"]]:
        e = history["models"][m]
        h, f = e["hist"], e["final"]
        rows.append(dict(
            model=LABEL[m], lr=e["lr"], trainable_params=e["trainable_params"], stored_params=e["stored_params"],
            tokens=h.get("tokens"), train_seconds=round(h.get("train_time", float("nan")), 1),
            tokens_per_s=round(h.get("tokens_per_s", float("nan"))),
            best_val_bpc=round(f["val"]["bpc"], 4), test_bpc=round(f["test"]["bpc"], 4),
            test_perplexity=round(f["test"]["ppl"], 3), test_accuracy=round(f["test"]["acc"], 4)))
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
    s.add_argument("--steps-per-token", type=int, default=1)
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
    r.add_argument("--seed", type=int, default=0)
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

    def builder(name):
        if name == "spiking":
            return lambda: make_spiking(args, V, device, streams)
        return lambda: TransformerLM(tcfg[name]).to(device)

    # ---- train each model ----------------------------------------------------- #
    for name in [m for m in ORDER if m in args.models]:
        if name in history["models"] and not args.force:
            print(f"[{name}] already in history.json, skipping (--force to retrain)")
            continue
        kind = "spiking" if name == "spiking" else "transformer"
        lr = args.lr_spiking if kind == "spiking" else args.lr_transformer
        tuning = None
        if lr is None and args.no_tune:
            lr = 2e-3 if kind == "spiking" else 3e-3
        elif lr is None:
            tuning = []
            for cand in LR_CANDIDATES[kind]:
                torch.manual_seed(args.seed)
                h, _ = run_training(f"tune {name} lr={cand:g}", "spiking" if kind == "spiking" else "transformer",
                                    builder(name)(), cand, args.tune_steps, streams, val_streams, args, device,
                                    eval_every=args.tune_steps, log_every=10 ** 9,
                                    eval_chunks=args.eval_chunks, seed=args.seed)
                v = h["val"][-1]["loss"] if h["val"] and not h.get("diverged") else float("inf")
                tuning.append(dict(lr=cand, val_loss=v))
                print(f"[tune {name}] lr {cand:g}: validation loss {v:.4f}", flush=True)
            lr = min(tuning, key=lambda r: r["val_loss"])["lr"]
            print(f"[{name}] selected lr {lr:g}", flush=True)

        torch.manual_seed(args.seed)
        model = builder(name)()
        trainable = model.num_effective_parameters if name == "spiking" else model.num_parameters
        stored = model.num_parameters
        hist, best = run_training(name, kind, model, lr, total_steps, streams, val_streams, args, device,
                                  eval_every=args.eval_every, log_every=args.log_every,
                                  eval_chunks=args.eval_chunks, seed=args.seed)
        if best is None:
            print(f"[{name}] produced no usable checkpoint (diverged?)", file=sys.stderr)
            continue
        model.load_state_dict(best)                               # best-validation weights
        final = dict(val=evaluate(kind, model, val_streams, T), test=evaluate(kind, model, test_streams, T))
        print(f"[{name}] TEST: bpc {final['test']['bpc']:.3f} | ppl {final['test']['ppl']:.2f} | "
              f"next-char accuracy {final['test']['acc']:.1%}")
        sample = sample_text(model, tok, args.prompt, args.sample_len, args.temperature, args.top_k, device)
        history["models"][name] = dict(lr=lr, tuning=tuning, trainable_params=trainable, stored_params=stored,
                                       hist=hist, final=final, sample=sample)
        json.dump(history, open(hist_path, "w"), indent=1)       # resume point
        del model, best
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name, e in history["models"].items():
        print(f"\n--- {LABEL[name]} sample ---\n{e['sample']}")
    make_outputs(history, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

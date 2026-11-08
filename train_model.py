"""
Train the spiking language model (model.py) as a character-level text generator.

    python train_model.py                          # Tiny Shakespeare (~1 MB), 20 epochs
    python train_model.py --data wikitext2         # ~11 MB of Wikipedia text (needs `datasets`)
    python train_model.py --data my_corpus.txt     # any UTF-8 text file
    python train_model.py --quick                  # tiny smoke test on a built-in snippet
    python train_model.py --sample --resume checkpoints/best.pt --prompt "ROMEO:"

Method
------
Surrogate-gradient backpropagation through time, i.e. the learning method that
trained the recurrent spiking network successfully in the benchmark sweeps (the
local rules need a recurrent memory the network had to be able to learn; BPTT
learns it). The recurrent core is an adaptive-LIF E/I network (override with
--neuron). All weights train: embedding, input, recurrent (Dale's law and the
sparsity mask preserved) and readout.

* Stateful truncated BPTT. The training text is cut into ``batch-size``
  contiguous streams; each step processes ``seq-len`` tokens and the network state
  carries into the next step (gradients stop at the chunk boundary), so the
  network learns to use long-range context exactly as it will when generating.
* Loss = next-token cross-entropy + a firing-rate regulariser that keeps every
  neuron near ``--rate-target`` spikes per step (prevents silence or runaway
  synchrony, which would starve the surrogate gradient).
* Before training the weight gain is calibrated so the untrained network already
  fires near the target rate.
* AdamW, linear warm-up then cosine decay, gradient clipping. Runs on the GPU.

Metrics are bits per character (lower is better; ~8 = random bytes, ~1.0-1.5 is
what small recurrent nets reach on Shakespeare) and perplexity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model import LMConfig, SpikingLM

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
URLS = {
    "tinyshakespeare":
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}

_BUILTIN = (
    "A spiking neuron integrates its input and fires only when a threshold is crossed. "
    "Between spikes the membrane potential leaks back toward rest. "
    "Networks of such neurons communicate with brief, discrete events instead of continuous values. "
    "Excitatory cells push their neighbours toward threshold, while inhibitory cells hold them back. "
    "The balance between the two keeps activity from dying out or running away. "
    "Learning changes the strength of the connections, so the pattern of spikes changes with experience. "
    "A recurrent network can remember what it has seen, because its own activity feeds back into itself. "
    "Slow adaptation lets a neuron carry information over many time steps. "
    "To predict the next letter, the network must keep track of the words that came before. "
) * 80            # ~ 60 KB: only for the --quick smoke test


def load_text(spec: str, cache_dir: str = "data") -> str:
    """'builtin', 'tinyshakespeare', 'wikitext2', or a path to a UTF-8 text file."""
    if spec == "builtin":
        return _BUILTIN
    cache = Path(cache_dir)
    if spec in URLS:
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"{spec}.txt"
        if not path.exists():
            print(f"downloading {spec} ...", flush=True)
            urllib.request.urlretrieve(URLS[spec], path)
        return path.read_text(encoding="utf-8", errors="replace")
    if spec == "wikitext2":
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / "wikitext2_train.txt"
        if not path.exists():
            try:
                from datasets import load_dataset
            except ImportError as e:
                raise SystemExit("wikitext2 needs the `datasets` package: pip install datasets") from e
            print("downloading wikitext-2 ...", flush=True)
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            path.write_text("".join(ds["text"]), encoding="utf-8")
        return path.read_text(encoding="utf-8", errors="replace")
    path = Path(spec)
    if not path.is_file():
        raise SystemExit(f"--data: '{spec}' is not a known dataset name or an existing file")
    return path.read_text(encoding="utf-8", errors="replace")


class CharTokenizer:
    """Character-level vocabulary."""

    def __init__(self, chars: List[str]):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self._lut = np.full(max(map(ord, self.chars)) + 1, -1, dtype=np.int64)
        for c, i in self.stoi.items():
            self._lut[ord(c)] = i

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        """Text -> int64 ids. Characters outside the vocabulary are dropped."""
        if not text:
            return np.zeros(0, dtype=np.int64)
        cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
        ids = np.full(cps.shape, -1, dtype=np.int64)
        ok = cps < len(self._lut)
        ids[ok] = self._lut[cps[ok]]
        return ids[ids >= 0]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def split_ids(ids, val_frac: float = 0.05, test_frac: float = 0.05):
    """Sequential train / validation / test split (no shuffling: text is a stream)."""
    n = len(ids)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    return ids[: n - n_val - n_test], ids[n - n_val - n_test: n - n_test], ids[n - n_test:]


def batchify(ids: torch.Tensor, batch: int, device) -> torch.Tensor:
    """(N,) -> (batch, N // batch): ``batch`` contiguous parallel streams."""
    length = ids.numel() // batch
    return ids[: length * batch].view(batch, length).to(device)


# --------------------------------------------------------------------------- #
# Schedule, evaluation
# --------------------------------------------------------------------------- #
def lr_at(step: int, total: int, base: float, warmup: int, min_ratio: float) -> float:
    """Linear warm-up, then cosine decay to base * min_ratio."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: SpikingLM, streams: torch.Tensor, seq_len: int, max_chunks: Optional[int] = None) -> dict:
    """Mean next-token loss over a stream of held-out text (state carried across chunks)."""
    model.eval()
    model.reset_state()
    losses, rates = [], []
    for n, i in enumerate(range(0, streams.shape[1] - seq_len, seq_len)):
        if max_chunks is not None and n >= max_chunks:
            break
        x, y = streams[:, i:i + seq_len], streams[:, i + 1:i + seq_len + 1]
        logits, spikes = model(x, return_spikes=True)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))
        rates.append(spikes.mean())
    model.reset_state()
    model.train()
    loss = float(torch.stack(losses).mean())
    return dict(loss=loss, bpc=loss / math.log(2), ppl=math.exp(min(loss, 20.0)),
                rate=float(torch.stack(rates).mean()))


def sample_text(model: SpikingLM, tok: CharTokenizer, prompt: str, n: int, temperature: float,
                top_k: Optional[int], device) -> str:
    ids = tok.encode(prompt)
    if len(ids) == 0:                                  # nothing usable in the prompt: start from the first symbol
        ids = np.array([0], dtype=np.int64)
    out = model.generate(torch.from_numpy(ids).to(device), n, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model, opt, step, best_val, tok, args) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict() if opt is not None else None, step=step,
                    best_val=best_val, cfg=asdict(model.cfg), chars=tok.chars, args=vars(args)), path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = SpikingLM(LMConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    return model, CharTokenizer(ck["chars"]), ck


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument_group("data")
    d.add_argument("--data", default="tinyshakespeare", help="builtin | tinyshakespeare | wikitext2 | path")
    d.add_argument("--cache-dir", default="data")
    d.add_argument("--val-frac", type=float, default=0.05)
    d.add_argument("--test-frac", type=float, default=0.05)
    m = ap.add_argument_group("model")
    m.add_argument("--n-rec", type=int, default=1024, help="recurrent spiking neurons")
    m.add_argument("--d-emb", type=int, default=128)
    m.add_argument("--exc-fraction", type=float, default=0.8)
    m.add_argument("--connectivity", type=float, default=0.05)
    m.add_argument("--neuron", default="alif", choices=["if", "lif", "alif", "stochastic_lif", "izhikevich"])
    m.add_argument("--neuron-kwargs", default=None, help='JSON, e.g. \'{"tau_m": 20, "tau_adapt": 200}\'')
    m.add_argument("--steps-per-token", type=int, default=1)
    m.add_argument("--dropout", type=float, default=0.1)
    m.add_argument("--seed", type=int, default=0)
    o = ap.add_argument_group("optimisation")
    o.add_argument("--seq-len", type=int, default=128, help="truncated-BPTT length")
    o.add_argument("--batch-size", type=int, default=64)
    o.add_argument("--epochs", type=int, default=20)
    o.add_argument("--max-steps", type=int, default=None, help="overrides --epochs")
    o.add_argument("--lr", type=float, default=2e-3)
    o.add_argument("--warmup", type=int, default=100)
    o.add_argument("--min-lr-ratio", type=float, default=0.1)
    o.add_argument("--weight-decay", type=float, default=0.0)
    o.add_argument("--grad-clip", type=float, default=1.0)
    o.add_argument("--rate-target", type=float, default=0.05, help="target spikes / neuron / step")
    o.add_argument("--rate-weight", type=float, default=50.0)
    o.add_argument("--no-calibrate", action="store_true", help="skip the initial rate calibration")
    o.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                   help="TensorFloat-32 matmuls on Ampere+ GPUs (faster, slightly less precise)")
    r = ap.add_argument_group("running")
    r.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    r.add_argument("--out", default="checkpoints")
    r.add_argument("--resume", default=None, help="checkpoint to continue from (or sample from)")
    r.add_argument("--log-every", type=int, default=25)
    r.add_argument("--eval-every", type=int, default=250)
    r.add_argument("--eval-batch", type=int, default=32)
    r.add_argument("--eval-chunks", type=int, default=40, help="validation chunks per evaluation")
    r.add_argument("--quick", action="store_true", help="tiny smoke test (built-in text, a few steps)")
    s = ap.add_argument_group("sampling")
    s.add_argument("--sample", action="store_true", help="only generate text from --resume")
    s.add_argument("--prompt", default="\n")
    s.add_argument("--sample-len", type=int, default=300)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args(argv)
    if args.quick:
        args.data, args.n_rec, args.d_emb, args.seq_len, args.batch_size = "builtin", 128, 32, 32, 16
        args.max_steps, args.eval_every, args.eval_chunks, args.log_every = 30, 15, 3, 5
        args.sample_len, args.warmup = 80, 5
        args.out = args.out if args.out != "checkpoints" else "checkpoints_quick"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- sample-only mode ------------------------------------------------ #
    if args.sample:
        if not args.resume:
            raise SystemExit("--sample needs --resume <checkpoint>")
        model, tok, _ = load_checkpoint(args.resume, device)
        print(sample_text(model, tok, args.prompt, args.sample_len, args.temperature, args.top_k, device))
        return 0

    # ---- data -------------------------------------------------------------- #
    text = load_text(args.data, args.cache_dir)
    if args.resume:
        model, tok, ck = load_checkpoint(args.resume, device)
    else:
        tok, ck, model = CharTokenizer.from_text(text), None, None
    ids = torch.from_numpy(tok.encode(text))
    train_ids, val_ids, test_ids = split_ids(ids, args.val_frac, args.test_frac)
    T, B = args.seq_len, args.batch_size
    streams = batchify(train_ids, B, device)
    eval_b = max(1, min(args.eval_batch, len(val_ids) // (T + 1)))
    val_streams = batchify(val_ids, eval_b, device)
    if streams.shape[1] <= T + 1 or val_streams.shape[1] <= T + 1:
        raise SystemExit("text too short for this --seq-len / --batch-size; lower them")
    steps_per_epoch = len(range(0, streams.shape[1] - T, T))
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    print(f"data: {len(text):,} chars | vocab {tok.vocab_size} | train {len(train_ids):,} "
          f"val {len(val_ids):,} test {len(test_ids):,} | {steps_per_epoch} steps/epoch, {total_steps} steps")

    # ---- model --------------------------------------------------------------- #
    if model is None:
        kwargs = json.loads(args.neuron_kwargs) if args.neuron_kwargs else None
        cfg = LMConfig(vocab_size=tok.vocab_size, d_emb=args.d_emb, n_rec=args.n_rec,
                       exc_fraction=args.exc_fraction, connectivity=args.connectivity,
                       neuron=args.neuron, steps_per_token=args.steps_per_token,
                       dropout=args.dropout, seed=args.seed)
        if kwargs is not None:
            cfg.neuron_kwargs = kwargs
        elif args.neuron != "alif":
            cfg.neuron_kwargs = {}                   # the ALIF defaults do not apply to other models
        model = SpikingLM(cfg).to(device)            # move BEFORE building the optimiser
        if not args.no_calibrate:
            f = model.calibrate_rate(streams[:, :T], args.rate_target)
            print(f"calibrated weight gain x{f:.3f} for a target rate of {args.rate_target}")
    n_params = model.num_parameters
    print(f"model: {model.cfg.neuron} E/I network, {model.cfg.n_rec} neurons "
          f"({model.rsnn.n_exc} E / {model.rsnn.n_inh} I), density {model.rsnn.density:.3f}, "
          f"{n_params / 1e6:.2f}M parameters")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    step, best_val = 0, float("inf")
    if ck is not None:
        if ck.get("opt"):
            opt.load_state_dict(ck["opt"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"resumed from step {step} (best val loss {best_val:.4f})")

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")
    last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")

    # ---- training loop ------------------------------------------------------- #
    model.train()
    run_ce = run_rate = run_gn = torch.zeros((), device=device)
    n_run, bad, t_log = 0, 0, time.time()
    done = False
    while not done:
        model.reset_state()                                      # new pass over the text
        off = random.randrange(T)                                # vary where chunks start
        data = streams[:, off:]
        for i in range(0, data.shape[1] - T, T):
            if step >= total_steps:
                done = True
                break
            lr = lr_at(step, total_steps, args.lr, args.warmup, args.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = data[:, i:i + T], data[:, i + 1:i + T + 1]
            logits, spikes = model(x, return_spikes=True)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            rate = spikes.mean(dim=(0, 1))                       # per-neuron spikes per step
            loss = ce + args.rate_weight * ((rate - args.rate_target) ** 2).mean()
            if not torch.isfinite(loss):
                bad += 1
                print(f"step {step}: non-finite loss, skipping update ({bad} in a row)", file=sys.stderr)
                if bad >= 20:
                    raise SystemExit("training diverged: lower --lr or --rate-weight")
                opt.zero_grad(set_to_none=True)
                model.reset_state()
                continue
            bad = 0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            model.rsnn.project()                                 # keep Dale's law / weight caps
            model.detach_state()                                 # truncate BPTT, keep the state
            step += 1
            run_ce, run_rate, run_gn = run_ce + ce.detach(), run_rate + rate.mean().detach(), run_gn + gn.detach()
            n_run += 1

            if step % args.log_every == 0:
                ce_m, rate_m, gn_m = float(run_ce) / n_run, float(run_rate) / n_run, float(run_gn) / n_run
                tps = n_run * B * T / (time.time() - t_log)
                print(f"step {step:6d}/{total_steps} | loss {ce_m:.3f} | bpc {ce_m / math.log(2):.3f} | "
                      f"rate {rate_m:.3f} | |g| {gn_m:.2f} | lr {lr:.2e} | {tps:,.0f} tok/s", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(dict(step=step, loss=ce_m, rate=rate_m, grad_norm=gn_m, lr=lr)) + "\n")
                run_ce = run_rate = run_gn = torch.zeros((), device=device)
                n_run, t_log = 0, time.time()

            if step % args.eval_every == 0 or step == total_steps:
                ev = evaluate(model, val_streams, T, args.eval_chunks)
                print(f"  >> validation: loss {ev['loss']:.4f} | bpc {ev['bpc']:.3f} | ppl {ev['ppl']:.2f} "
                      f"| rate {ev['rate']:.3f}", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps(dict(step=step, val=ev)) + "\n")
                if ev["loss"] < best_val:
                    best_val = ev["loss"]
                    save_checkpoint(best_path, model, opt, step, best_val, tok, args)
                save_checkpoint(last_path, model, opt, step, best_val, tok, args)
                print("  >> sample:", repr(sample_text(model, tok, args.prompt, min(args.sample_len, 200),
                                                       args.temperature, args.top_k, device)), flush=True)
                model.reset_state()                              # evaluation / sampling replaced the state
                t_log = time.time()
                n_run = 0
                run_ce = run_rate = run_gn = torch.zeros((), device=device)

    # ---- final test + samples ------------------------------------------------ #
    if os.path.exists(best_path):
        model, _, _ = load_checkpoint(best_path, device)
    test_b = max(1, min(args.eval_batch, len(test_ids) // (T + 1)))
    test_streams = batchify(test_ids, test_b, device)
    if test_streams.shape[1] > T + 1:
        ev = evaluate(model, test_streams, T)
        print(f"\nTEST: loss {ev['loss']:.4f} | bpc {ev['bpc']:.3f} | perplexity {ev['ppl']:.2f}")
    print(f"\n--- samples (temperature {args.temperature}, top-k {args.top_k}) ---")
    print(sample_text(model, tok, args.prompt, args.sample_len, args.temperature, args.top_k, device))
    return 0


if __name__ == "__main__":
    sys.exit(main())

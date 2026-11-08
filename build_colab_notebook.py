"""
Build ``biolearning_benchmarks.ipynb`` (a self-contained Google Colab notebook)
from the project's source files, so the notebook never drifts from the code:

    python build_colab_notebook.py

* neurons.py and rsnn.py are embedded verbatim with ``%%writefile`` cells, so
  the notebook needs nothing but a Colab runtime (no repository checkout).
* tests.py is split into one cell per section (task, budgets, training run,
  hyper-parameter search, sweeps, plotting); its command-line ``main()`` is
  replaced by notebook cells that run each sweep and display its figure and
  table inline.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "biolearning_benchmarks.ipynb"

BANNER = re.compile(r"^# -{40,} #$")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def split_tests(src: str):
    """-> (docstring, header_code, [(title, code), ...]) for everything before 'Main'."""
    doc = re.match(r'\s*"""(.*?)"""', src, re.S).group(1).strip("\n")
    body = src[src.index('"""', 3) + 3:]
    body = body.replace("from __future__ import annotations\n", "")
    lines = body.splitlines()
    starts = [i for i in range(len(lines) - 2)
              if BANNER.match(lines[i]) and BANNER.match(lines[i + 2]) and lines[i + 1].startswith("# ")]
    titles = [lines[i + 1][2:].strip() for i in starts]
    header = "\n".join(lines[: starts[0]])
    sections = []
    for k, i in enumerate(starts):
        if titles[k] == "Main":          # the CLI entry point is replaced by notebook cells
            break
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        sections.append((titles[k], "\n".join(lines[i:end])))
    return doc, header, sections


INTRO = """
# Spiking-network benchmarks (GPU)

Benchmarks a **recurrent spiking neural network** with excitatory and inhibitory
neurons on a noisy burst-detection task, sweeping five things: **network size,
E/I ratio, neuron type, learning method and sparsity**. Learning rates are tuned
beforehand by random search on a validation set. Every figure is drawn inline
and followed by the table behind it; nothing is downloaded.

**How to use**
1. *Runtime > Change runtime type > GPU* (a T4 is enough).
2. Leave **`QUICK_RUN = True`** for a first pass (a few minutes) to check that
   everything works, then set it to `False` and re-run for the real result.
3. *Runtime > Run all*, or run the cells top to bottom. Each sweep has its own
   cell, so you can re-run one on its own.

**About GPU speed.** All tensors, weights, traces and learning-rule updates live on
the GPU, and the input current and the readout are single large matrix products
over all time steps. The recurrent update itself is sequential in time, so for
small networks it is limited by kernel-launch overhead rather than arithmetic;
the GPU pays off most for the large networks and densities in the sweeps. Set
`DEVICE_OVERRIDE = "cpu"` to compare.
"""

SETTINGS = '''
#@title Run settings { display-mode: "form" }
QUICK_RUN = True            #@param {type:"boolean"}
DEVICE_OVERRIDE = "auto"    #@param ["auto", "cuda", "cpu"]
SEEDS_OVERRIDE = 0          #@param {type:"integer"}
SAVE_TO_DRIVE = False       #@param {type:"boolean"}
# QUICK_RUN: tiny budget, a few minutes: checks the pipeline, numbers are not meaningful.
# SEEDS_OVERRIDE: 0 keeps the budget default (3 seeds full, 1 quick).
# SAVE_TO_DRIVE: keep results.json / hparams.json (and PNG + CSV copies) in Google Drive so a
#                disconnected session can resume. Off: nothing is written outside this VM.
'''

GPU_CHECK = '''
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found. Use Runtime > Change runtime type > GPU, or continue on CPU (slower).")
'''

SELFTEST = '''
# Optional sanity checks of the neuron models and the network/learning rules (CPU, ~1 minute).
# Includes an exact check that the e-prop readout gradient equals the autograd gradient.
!python neurons.py
!python rsnn.py
'''

DRIVER = '''
import json, os
from dataclasses import asdict, replace

import pandas as pd
from IPython.display import display

SHOW = True        # draw figures inline (no files are written for the figures)
DPI = 110          # screen resolution; the command-line default of 200 is for saved files

if DEVICE_OVERRIDE != "auto":
    DEVICE = torch.device(DEVICE_OVERRIDE)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on:", DEVICE, f"({torch.cuda.get_device_name(DEVICE)})" if DEVICE.type == "cuda" else "")

BUD = QUICK if QUICK_RUN else FULL
if SEEDS_OVERRIDE:
    BUD = replace(BUD, seeds=SEEDS_OVERRIDE)

tag = "_quick" if QUICK_RUN else ""
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = f"/content/drive/MyDrive/biolearning_results{tag}"     # PNG + CSV copies are saved here too
else:
    OUT = None                                                   # display only
WORK = OUT or f"/content/biolearning_results{tag}"
os.makedirs(WORK, exist_ok=True)
RES_PATH = os.path.join(WORK, "results.json")

RESULTS = {"meta": {}, "sweeps": {}}
if os.path.exists(RES_PATH):
    RESULTS = json.load(open(RES_PATH))
    if RESULTS.get("meta", {}).get("task") != TASK_SIG:
        print("saved results belong to a different task definition: discarding them")
        RESULTS = {"meta": {}, "sweeps": {}}

ORACLE = oracle_accuracy(n_test=BUD.n_eval)
RESULTS["meta"] = dict(asdict(BUD), task=TASK_SIG, oracle=ORACLE, device=str(DEVICE))
STORE = HPStore(os.path.join(WORK, "hparams.json"))
FIG_ROWS = {}


def run_and_show(name, force=False):
    """Run one sweep (skipped if already done), draw its figure inline, show its table."""
    if force or name not in RESULTS["sweeps"]:
        RESULTS["sweeps"][name] = run_sweep(name, BUD, STORE)
        json.dump(RESULTS, open(RES_PATH, "w"), indent=1)       # resume point
    FIG_ROWS[name] = make_outputs(OUT, name, RESULTS["sweeps"][name], RESULTS["meta"])
    display(pd.DataFrame(FIG_ROWS[name]).drop(columns="sweep"))
    if FAILURES:
        print(f"{len(FAILURES)} run(s) failed so far and are recorded as NaN; see the tracebacks above")


preflight(BUD, ORACLE)
'''

SWEEP_TEXT = {
    "size": ("Sweep 1: network size",
             "Number of recurrent neurons, 25 to 400. Larger networks give more room for memory and "
             "benefit most from the GPU."),
    "ei": ("Sweep 2: excitatory / inhibitory ratio",
           "Fraction of neurons that are excitatory (50% to 100%). 100% means no inhibition."),
    "neuron": ("Sweep 3: neuron type",
               "IF, LIF, adaptive LIF, stochastic LIF and Izhikevich neurons; learning rates are "
               "tuned separately for each model."),
    "method": ("Sweep 4: learning method",
               "Reservoir (readout only), STDP, R-STDP, e-prop and surrogate-gradient BPTT. "
               "Accuracy on the left, wall-clock training time on the right."),
    "sparsity": ("Sweep 5: sparsity",
                 "Recurrent connection density from 2% to 100%."),
}

SECTION_MD = {
    "Task": "## The task\nBurst detection in noise, plus the ideal-observer ceiling.",
    "Configuration and budgets": "## Configuration\nBudgets (`QUICK` / `FULL`), search spaces and the sweeps.",
    "One training run": "## One training run\nBuilds a network, calibrates its gain, trains with the chosen "
                        "method and returns held-out accuracy. Everything runs on `DEVICE`.",
    "Hyper-parameter search (done before the sweeps, cached on disk)":
        "## Hyper-parameter search\nRandom search for learning rates on a validation set, done before "
        "the sweeps and cached.",
    "Sweeps": "## Sweep machinery",
    "Plotting (matplotlib, imported lazily so the experiments run without it)":
        "## Plotting\nLight-mode validated palette, legend + direct labels + table view.",
}

SUMMARY = '''
# Everything in one table, plus the tuned hyper-parameters.
all_rows = [r for rows in FIG_ROWS.values() for r in rows]
if all_rows:
    display(pd.DataFrame(all_rows))
print("tuned hyper-parameters:")
display(pd.DataFrame({k: v["hp"] for k, v in STORE.data.items() if k != "_task"}).T)
if FAILURES:
    print(f"\\n{len(FAILURES)} run(s) failed (recorded as NaN):")
    for f in FAILURES[:20]:
        print("  ", f)
'''


def main() -> None:
    neurons = (ROOT / "neurons.py").read_text()
    rsnn = (ROOT / "rsnn.py").read_text()
    doc, header, sections = split_tests((ROOT / "tests.py").read_text())

    cells = [md(INTRO), code(GPU_CHECK), code(SETTINGS),
             md("## Library code\nThe neuron models and the network are written to files so they can be "
                "imported. They are embedded verbatim from `neurons.py` and `rsnn.py`."),
             code("%%writefile neurons.py\n" + neurons), code("%%writefile rsnn.py\n" + rsnn),
             md("### Optional self-tests"), code(SELFTEST),
             md("## Benchmark protocol\n```text\n" + doc + "\n```"),
             code(header)]
    for title, body in sections:
        cells += [md(SECTION_MD.get(title, f"## {title}")), code(body)]
    cells += [md("## Setup\nPicks the device and budget, computes the ideal-observer ceiling, and runs a "
                 "quick preflight check (a frozen reservoir should beat chance)."),
              code(DRIVER)]
    for name, (title, text) in SWEEP_TEXT.items():
        cells += [md(f"## {title}\n{text}"), code(f'run_and_show("{name}")')]
    cells += [md("## Summary"), code(SUMMARY)]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "biolearning_benchmarks.ipynb", "provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # nbformat 4.5 requires an id on every cell
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell{i:03d}"
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT.name}: {len(cells)} cells")


if __name__ == "__main__":
    main()

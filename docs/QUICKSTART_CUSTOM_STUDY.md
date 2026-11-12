# Quick start: custom dataset and custom objectives

This guide walks through using COMPASS with __your own
dataset__ and __your own definitions__ of \(f_P\), \(f_S\), and \(f_A\). 

---

## What is COMPASS?

COMPASS stands for *constrained objective multi-criteria performance, accountability, stability and saliency*.

When you train many model configurations in a hyperparameter sweep, picking the
single best accuracy often hides problems: an imprecise performance estimate,
saliency maps that cannot be inspected, or failure on clinically important metrics.

**COMPASS** scores every configuration on three scalar objectives and then runs a
**fixed constrained-Lagrangian analysis** to answer:

- Which configurations are *feasible* at your chosen stability and accountability thresholds (\(K_S\), \(K_A\))?
- What is the best achievable performance on that feasible set?
- What is the *shadow price* of tightening each constraint?

| Objective | Symbol | What it measures (default ECG) |
|-----------|--------|-------------------------------|
| __Performance__ | \(f_P\) | Bootstrap classification metrics (acc, precision, recall, specificity) |
| __Stability__ | \(f_S\) | Precision of that estimate under test-set resampling (higher = tighter) |
| __Accountability__ | \(f_A\) | Saliency continuity, compactness and contrastivity |

__You define__ how \(f_P\), \(f_S\), \(f_A\) are computed (YAML, pandas, or custom CSVs).
__COMPASS fixes__ the Lagrangian frontier and shadow-price machinery in `compass_analysis.py`.

### What you get at the end

| Output | Where | Purpose |
|--------|-------|---------|
| `analysis/three_objective_dataset.csv` | `COMPASS_DATA_ROOT` | Joined \(f_P, f_S, f_A\) per sweep config |
| Feasibility frontier plot | `notebooks/02_feasibility_frontier.ipynb` | Trade-off between stability and accountability thresholds |
| Shadow prices | same notebook | Cost of tightening \(f_S\) or \(f_A\) constraints in \(f_P\) |
| Recommended config | same notebook | Best \(f_P\) on the feasible set for your task |

### Colour key (architecture figure)

| Colour | Meaning |
|--------|---------|
| __Teal__ | Fixed COMPASS core — Lagrangian frontier, shadow prices (`compass_analysis.py`) |
| __Amber__ | User-configurable — objective formulas (`objectives:` in YAML) |
| __Gray__ | Swappable — dataset builder and ECG training/XAI (optional) |

---

## 0. Install and clone

```bash
git clone https://github.com/Ja5ya/compass.git
cd compass
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Layer 1 — Dataset

### What you need

A run directory with pickle files per [DATA_CONTRACT.md](DATA_CONTRACT.md):

```sh
<dataset_root>/
  shared/recs_normal.pkl
  datasets/<task>_vs_normal/
    recs_psych.pkl
```

Each record: `{"signal": ndarray, "label": 0|1, ...}`.

### Configure

```bash
cp configs/dataset.template.yaml configs/my_study.yaml
```

Edit paths, task codes, and ECG geometry:

```yaml
name: my_study
output_root: /path/to/writable/run

data:
  mode: prebuilt
  prebuilt_root: /path/to/dataset_root

diagnoses:
  codes: [MY_TASK]

ecg:
  n_leads: 12
  signal_len: 5000
```

Set environment variables:

```bash
export COMPASS_CONFIG=configs/my_study.yaml
export COMPASS_DATA_ROOT=/path/to/writable/run
export COMPASS_PREBUILT_ROOT=/path/to/dataset_root
```

__Build mode alternative:__ if you have manifest + `.npy` files instead of pickles,
set `data.mode: build` and run `python src/build_datasets.py --config $COMPASS_CONFIG`.
See `configs/dataset.template.yaml` for an example.

__Reference builders:__ `examples/dataset_builders/` (MIMIC, ecgpsych).

---

## Layer 2 — Modality analysis (ECG reference)

This layer is **optional**. Skip it entirely if you supply artefact CSVs directly
([Level 3 objectives](#level-3--supply-your-own-artefact-csvs) below).

### Stage 2 — Train hyperparameter sweep

```bash
# Smoke test: one sweep task
PYTHONPATH=src python src/train_multi_f_array.py --config $COMPASS_CONFIG --task-id 0

# Full sweep on Slurm
bash slurm/submit_train.sh
```

Outputs per config: `hyper_sweep/<task>/<arch>/lr*_dr*/test_predictions.npz`,
`metrics.json`.

### Stage 3 — Saliency scoring (f_A components)

```bash
PYTHONPATH=src python src/run_xai.py --config $COMPASS_CONFIG --code MY_TASK --hyper_sweep_fa
PYTHONPATH=src python src/compass_analysis.py --config $COMPASS_CONFIG merge-fa
```

Outputs: `xai_results/hyper_sweep_fa/sweep_fa_<task>.csv` with component columns
(`continuity_valid`, `compactness_valid`, `contrastivity_valid`, …).

---

## Layer 3 — COMPASS core

### Stage 4 — Bootstrap (f_P, f_S inputs)

```bash
PYTHONPATH=src python src/compass_analysis.py --config $COMPASS_CONFIG bootstrap
```

Outputs:

| File | Contents |
|------|----------|
| `hyper_sweep/bootstrap_results.csv` | Raw bootstrap mean/std per metric |
| `hyper_sweep/bootstrap_results_with_scores.csv` | + `f_P`, `f_S` from YAML |

Default bootstrap metrics: `acc`, `precision`, `recall`, `specificity`. Custom
metrics require [Level 3](#level-3--supply-your-own-artefact-csvs).

### Stage 5 — Feasibility frontier

```bash
jupyter notebook notebooks/02_feasibility_frontier.ipynb
```

The notebook calls `load_three_objective_dataset()`, which joins bootstrap + XAI,
applies your YAML objectives, and writes `analysis/three_objective_dataset.csv`.

Verify pipeline status:

```bash
PYTHONPATH=src python src/compass_analysis.py --config $COMPASS_CONFIG status
```

---

## Custom objectives — three extension levels

### Level 1 — Edit YAML (no code)

Add or edit the `objectives:` and `analysis:` blocks in your study YAML
(`configs/dataset.template.yaml` has a full example):

```yaml
objectives:
  f_P:
    metrics: [test_recall_mean, test_specificity_mean]
    weights: [0.7, 0.3]
    normalize: per_task_minmax
  f_S:
    method: inv_std          # inv_std | cv | recip_clipped
  f_A:
    components: [continuity_valid, compactness_valid, contrastivity_valid]
    weights: [0.5, 0.25, 0.25]

analysis:
  task_column: task_id       # optional alias for psych_code
  recall_min: 0.30           # set null to disable
  recall_metric: test_recall_mean
```

__Important:__ when XAI component columns are present, `f_A` is __recomputed__ from
YAML weights via `apply_fa_scores()` — the equal 1/3 value written by `run_xai.py`
is ignored.

Re-open the frontier notebook (or re-run `load_three_objective_dataset()`) after
changing objectives. Re-run bootstrap only if f_P/f_S _input columns_ changed.

| Objective | YAML keys | Input columns (default ECG) |
|-----------|-----------|----------------------------|
| \(f_P\) | `objectives.f_P.metrics`, `.weights`, `.normalize` | `test_*_mean` from bootstrap |
| \(f_S\) | `objectives.f_S.method`, `.metrics` | `test_*_std` from bootstrap |
| \(f_A\) | `objectives.f_A.components`, `.weights` | XAI component columns |

---

### Level 2 — Post-process CSV (any formula)

For non-linear or domain-specific formulas, use the joined table:

```python
import pandas as pd

df = pd.read_csv("analysis/objective_inputs.csv")

df["f_P"] = 0.5 * df["test_recall_mean"] + 0.5 * df["test_specificity_mean"]
df["f_S"] = 1.0 - df["test_recall_std"]
df["f_A"] = df[["continuity_valid", "contrastivity_valid"]].mean(axis=1)

df.to_csv("analysis/my_objectives.csv", index=False)
```

In the frontier notebook, load `my_objectives.csv` and pass it to
`compute_feasibility_frontier()` and `reliable_shadow_price()`. Those functions
only require scalar `f_P`, `f_S`, `f_A` columns grouped by task.

---

### Level 3 — Supply your own artefact CSVs

Skip Layers 1–2. Place files under `COMPASS_DATA_ROOT`:

```sh
hyper_sweep/bootstrap_results.csv
xai_results/hyper_sweep_fa/sweep_fa_<task>.csv   # or sweep_fa_summary.csv
```

**Required join keys** (one row per hyperparameter configuration):

| Column | Example |
|--------|---------|
| `psych_code` or `task_id` | `MY_TASK` |
| `architecture` | `resnet1d` |
| `learning_rate` | `0.0001` |
| `dropout_rate` | `0.1` |

__Bootstrap CSV__ — include columns referenced by your `objectives.f_P` / `f_S`
config (e.g. `test_recall_mean`, `test_recall_std`, …).

__XAI CSV__ — include columns referenced by `objectives.f_A.components`, or a
precomputed `f_A` column if no components are listed.

Then score and join:

```python
import os
os.environ["COMPASS_CONFIG"] = "configs/my_study.yaml"
os.environ["COMPASS_DATA_ROOT"] = "/path/to/run/with/csvs"

from compass_analysis import load_analysis_config, load_three_objective_dataset

cfg = load_analysis_config()
df = load_three_objective_dataset(cfg)
print(df[["f_P", "f_S", "f_A"]].head())
```

Training and XAI scripts are not required at this level.

---

## End-to-end checklist

| Step | Command / file | Layer |
|------|----------------|-------|
| 1. Copy config | `cp configs/dataset.template.yaml configs/my_study.yaml` | — |
| 2. Prepare pickles | external builder or `build_datasets.py` | 1 |
| 3. Train sweep | `train_multi_f_array.py` or Slurm | 2 |
| 4. Bootstrap | `compass_analysis.py bootstrap` | 3 |
| 5. XAI + merge | `run_xai.py` + `compass_analysis.py merge-fa` | 2 |
| 6. Custom objectives | edit YAML (L1) or pandas (L2) or BYO CSV (L3) | 3 |
| 7. Frontier | `notebooks/02_feasibility_frontier.ipynb` | 3 |

---

## Worked examples in this repository

| Study | Config | Notes |
|-------|--------|-------|
| MIMIC F32.9 | `configs/mimic_f329.yaml` | Prebuilt pickles, full sweep |
| Psychiatry ECG transfer | `configs/ecgpsych.yaml` | Single task SCZ |

---

## Further reading

| Document | Contents |
|----------|----------|
| [DATA_CONTRACT.md](DATA_CONTRACT.md) | Pickle and artefact schemas, config reference |
| [REPOSITORY_SCOPE.md](REPOSITORY_SCOPE.md) | What is in-repo vs user-supplied |
| [SOFTWAREX_CHECKLIST.md](SOFTWAREX_CHECKLIST.md) | GitHub / SoftwareX readiness |
| [README.md](../README.md) | Install, Slurm reference run, environment variables |

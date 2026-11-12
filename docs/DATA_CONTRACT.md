# Data contract between external dataset builders and the COMPASS pipeline

COMPASS is **modality-agnostic at the core**. Users
prepare datasets externally, then run modality-specific analysis (ECG in this
repository), then apply COMPASS scoring and frontier analysis on standard artifacts.

## Three layers

| Layer | Owner | This repo |
|-------|-------|-----------|
| 1. Dataset build | User / reference scripts | Optional `build_datasets.py` (build mode) |
| 2. Modality analysis | ECG module | `train_multi_f_array.py`, `run_xai.py` |
| 3. COMPASS core | Framework | `compass_analysis.py`, `src/compass/` |

## Configuration reference

YAML study configs (`configs/dataset.template.yaml` is the starting point):

| Section | Controls |
|---------|----------|
| `data.mode` | `prebuilt` (external pickles) or `build` (in-repo builder) |
| `data.prebuilt_root` | Read-only dataset root when mode=prebuilt |
| `output_root` | Writable run root for sweep / XAI / analysis |
| `ecg.*` | Lead count and signal length |
| `diagnoses.codes` | Task labels (e.g. ICD-10 codes, `SCZ`, …) |
| `training.*` | Sweep grid (architectures, learning rates, dropout) |
| `objectives.*` | \(f_P\), \(f_S\), \(f_A\) formulas (column lists + weights) |
| `analysis.*` | Task column alias (`task_id`), recall filter for frontier dataset |

Example configs: `configs/mimic_f329.yaml`, `configs/ecgpsych.yaml`, `configs/dataset.template.yaml`.

## Layer 1 — Dataset pickles (input contract)

Users bring a run directory with this layout:

```
<dataset_root>/
  shared/recs_normal.pkl
  datasets/<task>_vs_normal/
    recs_psych.pkl
    recs_normal.pkl -> ../../shared/recs_normal.pkl   # symlink optional
```

Each record in `recs_psych.pkl` / `recs_normal.pkl` must be a dict:

| Key | Type | Required | Notes |
|-----|------|----------|-------|
| `signal` | `np.ndarray` | yes | Shape `(n_channels, n_samples)`; ECG default `(12, 5000)` |
| `label` | `0` or `1` | yes | Binary case vs control |
| `gender` | str | no | Used for demographic matching |
| `age` | float | no | Used for demographic matching |
| `subject_id` | int/str | no | Record identifier |

Validate locally:

```bash
PYTHONPATH=src python -c "
from pathlib import Path
from compass.contracts import validate_dataset_layout
print(validate_dataset_layout(Path('datasets/icd10_F329_vs_normal'), n_leads=12))
"
```

### Config modes

**Prebuilt** (recommended for MIMIC and custom external builders):

```yaml
data:
  mode: prebuilt
  prebuilt_root: /path/to/dataset_root
output_root: /path/to/writable/run   # hyper_sweep, xai, analysis
```

**Build** (optional in-repo example for manifest + `.npy` cohorts):

```yaml
data:
  mode: build
output_root: /path/to/run
```

## Layer 2 — Training artifacts

Per hyperparameter config (`hyper_sweep/<task>/<arch>/lr*_dr*/`):

| File | Contents |
|------|----------|
| `test_predictions.npz` | `y_true`, `y_score` |
| `metrics.json` | Point estimates on val/test |

## Layer 3 — COMPASS artifact tables

| File | Contents | Used for |
|------|----------|----------|
| `hyper_sweep/bootstrap_results.csv` | Bootstrap mean/std per metric | \(f_P\), \(f_S\) inputs |
| `xai_results/hyper_sweep_fa/sweep_fa_<task>.csv` | Saliency component scores | \(f_A\) inputs |
| `analysis/objective_inputs.csv` | Joined bootstrap + **all** saliency columns (`f_A`, `*_valid`, per-method, diagnostics) before \(f_P\)/\(f_S\) scoring | Custom formulas / reweighting |
| `analysis/three_objective_dataset.csv` | Joined + default \(f_P,f_S,f_A\) | Frontier notebooks |

### Saliency / \(f_A\) columns and exclusion rules

Primary columns in `sweep_fa_<task>.csv` (and the merged `sweep_fa_summary.csv`):

| Column | Role |
|--------|------|
| `code`, `sweep_arch`, `sweep_run` | Join keys (parsed to task / architecture / lr / dropout) |
| `f_A` | Equal mean of the three components below |
| `continuity_valid` | Pooled continuity (SG, GC, non-degenerate SHAP, LIME) |
| `compactness_valid` | Pooled compactness (SG, GC, non-degenerate SHAP; **LIME excluded**) |
| `contrastivity_valid` | Pooled contrastivity (SG, GC, non-degenerate SHAP, LIME) |

**Degenerate SHAP.** A SHAP attribution is treated as degenerate when its raw
maximum absolute value is below `0.001` (`SHAP_DEGEN_MAX` in `run_xai.py`).
Degenerate samples are dropped from **all** \(f_A\) components before pooling.

**LIME and compactness.** LIME attributions are retained for continuity and
contrastivity but excluded from compactness because the fixed 200-segment
structure yields a near-constant compactness (\(\approx 0.525\)) across codes.
The per-config diagnostic column is `compactness_lime_ref`.

### Custom objective formulas

Objectives are applied by `src/compass/objectives.py` from YAML (`objectives.f_P`,
`objectives.f_S`, `objectives.f_A`). The pipeline **recomputes** `f_A` from
component columns using configured weights — it does not trust the pre-written
`f_A` value in XAI CSVs when components are present.

See [Objective extension levels](#objective-extension-levels) below and
[QUICKSTART_CUSTOM_STUDY.md](QUICKSTART_CUSTOM_STUDY.md) for a full walkthrough.

## Objective extension levels

The COMPASS core separates a **fixed Lagrangian frontier shell** from **user-defined
scalar objectives** \(f_P\), \(f_S\), \(f_A\). Three extension levels are supported:

### Level 1 — Edit YAML (no code)

Configure weighted combinations of existing artifact columns under `objectives:` in
your study YAML. Example — recall-only performance and continuity-heavy accountability.

```yaml
objectives:
  f_P:
    metrics: [test_recall_mean, test_specificity_mean]
    weights: [0.7, 0.3]
    normalize: per_task_minmax
    task_column: task_id      # optional; default psych_code
  f_S:
    method: inv_std
  f_A:
    components: [continuity_valid, compactness_valid, contrastivity_valid]
    weights: [0.6, 0.2, 0.2]

analysis:
  task_column: task_id        # alias for psych_code in joined tables
  recall_min: 0.30            # set null to disable
  recall_metric: test_recall_mean
```

Re-run `compass_analysis.py bootstrap` (if f_P/f_S inputs changed) and reload the
frontier notebook. `f_A` is recomputed on merge via `apply_fa_scores()`.

### Level 2 — Post-process `objective_inputs.csv` (any formula)

`analysis/objective_inputs.csv` joins bootstrap metrics with **all** XAI component
and diagnostic columns. Use pandas (or another tool) for non-linear or domain-specific
formulas, then pass the resulting `f_P`, `f_S`, `f_A` columns to the frontier notebook:

```python
import pandas as pd

df = pd.read_csv("analysis/objective_inputs.csv")
df["f_P"] = 0.5 * df["test_recall_mean"] + 0.5 * df["test_specificity_mean"]
df["f_S"] = 1.0 - df["test_recall_std"]
df["f_A"] = df[["continuity_valid", "contrastivity_valid"]].mean(axis=1)
df.to_csv("analysis/my_objectives.csv", index=False)
```

The Lagrangian functions in `compass_analysis.py` (`compute_feasibility_frontier`,
`reliable_shadow_price`) only require scalar `f_P`, `f_S`, `f_A` columns.

### Level 3 — Supply your own artifact CSVs (any modality)

Bring custom bootstrap and saliency tables that share the join keys:

| Column | Role |
|--------|------|
| `psych_code` or `task_id` (see `analysis.task_column`) | Task / cohort label |
| `architecture` | Model family |
| `learning_rate` | Sweep hyperparameter |
| `dropout_rate` | Sweep hyperparameter |

Place bootstrap metrics in `hyper_sweep/bootstrap_results.csv` and saliency
components in `xai_results/hyper_sweep_fa/sweep_fa_<task>.csv` (or merge to
`sweep_fa_summary.csv`). The COMPASS core scores and joins them; training/XAI scripts
are optional.

**Task column alias.** Training artifacts still write `psych_code` internally for
backward compatibility. Set `analysis.task_column: task_id` to rename on read when
assembling frontier datasets.

## Examples in this repository

| Example | Config | Data mode |
|---------|--------|-----------|
| MIMIC F32.9 (public) | `configs/mimic_f329.yaml` | prebuilt |
| MIMIC six-code test grid | `configs/mimic.yaml` | prebuilt |
| Psychiatry-ECG transfer | `configs/ecgpsych.yaml` | prebuilt |
| Custom build-mode study | `configs/dataset.template.yaml` | build |

Reference builders: `examples/dataset_builders/`. See [REPOSITORY_SCOPE.md](REPOSITORY_SCOPE.md).

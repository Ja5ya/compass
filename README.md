# COMPASS

**Constrained objective multi-criteria performance, accountability, stability and saliency.**

COMPASS chooses among trained models on more than predictive accuracy. It scores
every candidate configuration on performance (\(f_P\)), stability (\(f_S\)) and
accountability (\(f_A\)), then selects the best-performing configuration that
meets user-set minimum thresholds on stability and accountability (\(K_S\), \(K_A\)).
Shadow prices, estimated by finite differences, report how much performance each
tighter threshold costs. You define the three scores; the constrained-Lagrangian
shell is fixed.

| Objective | Symbol | Default meaning (ECG reference) |
|-----------|--------|--------------------------------|
| Performance | \(f_P\) | Equal-weight mean of bootstrap accuracy, precision, recall and specificity, min-max normalised per task |
| Stability | \(f_S\) | One minus the min-max normalised mean bootstrap standard deviation (higher = tighter estimate) |
| Accountability | \(f_A\) | Equal-weight mean of saliency continuity, compactness and contrastivity |

Saliency is the attribution layer: the maps from which every component of \(f_A\) is computed.
Organisational accountability is recorded by the artefact contract and is not scored.

---

## Install

```bash
git clone https://github.com/Ja5ya/compass.git
cd compass
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, TensorFlow 2.15+ (`requirements.txt`) for the ECG training / XAI path.

### Laptop demo (reviewers: start here)

Exercises the **real Layer-3 core** (bootstrap → join → scoring → constraints → shadow prices → frontier) on **fabricated** Layer-2 tables. No GPU, TensorFlow, PhysioNet or Kaggle. About **8–10 seconds**.

```bash
pip install -r examples/synthetic/requirements.txt   # deliberately no TensorFlow
bash examples/synthetic/run.sh
# or: make demo
```

The six runs are written by `make_synthetic.py`, **not** by training or `run_xai.py`. Printed recalls/accuracies are synthetic. The demo covers the join and Lagrangian core (including a binding \(\lambda_A\) and a cliff); it does **not** cover training or attribution scoring. Details: [examples/synthetic/README.md](examples/synthetic/README.md).

MIMIC (credentialed) and the open [Psychiatry ECG](https://www.kaggle.com/datasets/buraktaci/psychiatry-ecg) path remain optional full ECG runs; they need TensorFlow and, for a useful sweep, a GPU or Slurm.

---

## Documentation

| Document | When to read it |
|----------|-----------------|
| **[examples/synthetic/](examples/synthetic/)** | **Reviewer demo** — one-command core run, no ECG download |
| **[docs/QUICKSTART_CUSTOM_STUDY.md](docs/QUICKSTART_CUSTOM_STUDY.md)** | New dataset, custom \(f_P/f_S/f_A\), full walkthrough |
| [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) | Pickle / CSV schemas, config reference, objective extension levels |
| [docs/REPOSITORY_SCOPE.md](docs/REPOSITORY_SCOPE.md) | What is in-repo vs user-supplied (SoftwareX) |
| [docs/SOFTWAREX_CHECKLIST.md](docs/SOFTWAREX_CHECKLIST.md) | Pre-upload GitHub / SoftwareX checklist |
| [examples/dataset_builders/](examples/dataset_builders/) | MIMIC / ecgpsych dataset builders |

Architecture overview:

![COMPASS three-layer architecture](mof_architecture.png)

Teal = fixed Lagrangian core · Amber = configurable \(f_P\), \(f_S\), \(f_A\) · Gray = swappable dataset / ECG use case.  
Regenerate with `python scripts/make_figure.py` if you edit the figure script.

---

## Three layers

| Layer | What | In this repo |
|-------|------|--------------|
| 1. Dataset | User-built pickles or optional builders | `configs/*.yaml` → `data.mode` |
| 2. ECG use case | Train sweep + saliency scoring *(swappable)* | `train_multi_f_array.py`, `run_xai.py` |
| 3. COMPASS core | Bootstrap + objectives + frontier *(fixed Lagrangian)* | `compass_analysis.py`, `src/compass/` |

**New study?** → [docs/QUICKSTART_CUSTOM_STUDY.md](docs/QUICKSTART_CUSTOM_STUDY.md)

---

## Pipeline (five stages)

| Stage | Script | Main output |
|-------|--------|-------------|
| 1. Dataset | `build_datasets.py` *(optional)* | Pickles under `datasets/` |
| 2. Train sweep | `train_multi_f_array.py` | `hyper_sweep/.../test_predictions.npz` |
| 3. Saliency | `run_xai.py` | `xai_results/.../sweep_fa_<task>.csv` |
| 3b. Merge | `compass_analysis.py merge-fa` | `sweep_fa_summary.csv` |
| 4. Bootstrap | `compass_analysis.py bootstrap` | `bootstrap_results.csv` |
| 5. Frontier | `notebooks/02_feasibility_frontier.ipynb` | `analysis/three_objective_dataset.csv` |

All outputs live under `COMPASS_DATA_ROOT` / `output_root` in your YAML config.

---

## Reference run — MIMIC on Slurm

```bash
cp configs/env.mimic.example configs/env.mimic   # edit COMPASS_CONFIG, COMPASS_DATA_ROOT, COMPASS_PREBUILT_ROOT
source configs/env.mimic

bash slurm/submit_train.sh    # stages 2 + 4 (training + bootstrap)
# wait for completion
bash slurm/submit_xai.sh      # stages 3 + 3b (saliency + merge-fa)

PYTHONPATH=src python src/compass_analysis.py --config $COMPASS_CONFIG status
jupyter notebook notebooks/02_feasibility_frontier.ipynb
```

**Run order:** train → bootstrap → saliency → merge-fa → frontier. Do not submit saliency scoring until training finishes.

**Local smoke test** (one task, no Slurm):

```bash
source configs/env.mimic
PYTHONPATH=src python src/train_multi_f_array.py --config $COMPASS_CONFIG --task-id 0
N_BOOTSTRAP=100 PYTHONPATH=src python src/compass_analysis.py --config $COMPASS_CONFIG bootstrap
PYTHONPATH=src python src/run_xai.py --config $COMPASS_CONFIG --code F329 --hyper_sweep_fa
PYTHONPATH=src python src/compass_analysis.py --config $COMPASS_CONFIG merge-fa
```

Example configs: `configs/mimic.yaml` (6-task test grid), `configs/mimic_f329.yaml` (full single-code sweep), `configs/ecgpsych.yaml` (transfer study), `configs/dataset.template.yaml` (build-mode starting point).

Sweep size: `n_codes × n_architectures × n_learning_rates × n_dropout_rates` — check with `PYTHONPATH=src python src/pipeline_config.py --config $COMPASS_CONFIG sweep-count`.

---

## Environment variables

Environment variables use the `COMPASS_` prefix. The core package is `src/compass/`.

| Variable | Meaning |
|----------|---------|
| `COMPASS_CONFIG` | Path to study YAML |
| `COMPASS_DATA_ROOT` | Writable run directory |
| `COMPASS_PREBUILT_ROOT` | Prebuilt pickle root (`data.mode: prebuilt`) |
| `COMPASS_AUTO_BOOTSTRAP` | `1` (default): chain bootstrap after `submit_train.sh` |
| `COMPASS_AUTO_MERGE_FA` | `1` (default): chain merge-fa after `submit_xai.sh` |
| `N_BOOTSTRAP` | Bootstrap resamples (default `1000`; use `100` for tests) |
| `XAI_MODE` | `sweep` (default) or `pareto` |

Full list: see [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md#configuration-reference).

---

## Custom objectives

Edit `objectives:` and `analysis:` in your study YAML, or post-process
`analysis/objective_inputs.csv`. Three extension levels (YAML / pandas / BYO CSVs):
[docs/DATA_CONTRACT.md#objective-extension-levels](docs/DATA_CONTRACT.md#objective-extension-levels).

Walkthrough: [docs/QUICKSTART_CUSTOM_STUDY.md](docs/QUICKSTART_CUSTOM_STUDY.md).

---

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

---

## Related work

Extends the NeurIPS multi-objective trustworthiness framework. This package
generalises the pipeline: bring your own dataset → run ECG module → run COMPASS core.

## License

MIT — see [LICENSE](LICENSE).

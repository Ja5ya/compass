# Synthetic COMPASS core demo

Laptop-scale demo of **Layer 3 only** (bootstrap, join, scoring, constraints, shadow prices, frontier).

## What is fabricated

The six sweep runs are **fabricated** by `make_synthetic.py`:

- `test_predictions.npz` / `metrics.json` are **not** from `train_multi_f_array.py`
- `sweep_fa_DEMO.csv` is **not** from `run_xai.py`
- Printed recalls / accuracies / \(f_P\) values are **synthetic**, not ECG or clinical results

The demo then runs the **real** `compass_analysis` core on those tables. It validates the artefact contract and the constrained-Lagrangian shell. It does **not** cover training, attribution scoring, or any scientific finding.

## What it does cover

| Covered | Not covered |
|---------|-------------|
| merge-fa, bootstrap, join | `train_multi_f_array.py` |
| \(f_P\)/\(f_S\)/\(f_A\) scoring | `run_xai.py` |
| recall viability filter | TensorFlow / GPU |
| binding shadow price (\(\lambda_A>0\)) | Real ECG waveforms |
| cliff (\(\lambda=\infty\)) | PhysioNet / Kaggle download |
| staircase frontier steps | MIMIC / Psychiatry ECG science |

Default operating points on the fabricated DEMO grid:

- **Binding:** \((K_S,K_A)=(0.20,0.55)\) → non-zero \(\lambda_A\)
- **Cliff:** \((K_S,K_A)=(0.20,0.70)\) → undefined finite difference after one adaptive step (`lambda_A: null` in `shadow_prices.json`; strict JSON, never `Infinity`)

## Run

From the repository root:

```bash
pip install -r examples/synthetic/requirements.txt   # no TensorFlow
bash examples/synthetic/run.sh
# or: make demo
```

Typical wall-clock on a laptop / login node: **about 8–10 seconds** (`N_BOOTSTRAP=200`).

Bootstrap is seeded (`SEED=42`). The check against `expected/` uses a numeric tolerance so small NumPy/CPU differences are not false failures.

## Refresh expected outputs

```bash
UPDATE_EXPECTED=1 bash examples/synthetic/run.sh
```

## Optional: real ECG paths (not required for review)

| Path | Why it is secondary |
|------|---------------------|
| MIMIC-IV-ECG | PhysioNet credentialing + large pickles + GPU/Slurm |
| [Psychiatry ECG (Kaggle)](https://www.kaggle.com/datasets/buraktaci/psychiatry-ecg) | Open download, but still needs TensorFlow training; see `configs/ecgpsych.yaml` |

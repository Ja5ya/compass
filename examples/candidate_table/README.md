# Candidate-table example

Apply the COMPASS selection core to a CSV you already have: one row per
candidate with scored $f_P$, $f_S$ and $f_A$ (auditability). No training,
saliency scoring, GPU or patient data.

## Run

```bash
# same four packages as examples/synthetic/
pip install -r ../synthetic/requirements.txt
python examples/candidate_table/run_candidate_table.py
```

From the repository root. Default thresholds are $(K_S,K_A)=(0.65,0.54)$.

## Files

| File | Role |
|------|------|
| `candidates.csv` | Eight fabricated candidates (`model`, `f_P`, `f_S`, `f_A`) |
| `run_candidate_table.py` | Constrained selection, shadow prices, frontier summary |
| `expected_output.txt` | Reference stdout (selected model **B**, $\lambda_A\approx 6.923$) |

## What to expect

At the default thresholds the highest-$f_P$ feasible candidate is model **B**.
Tightening $K_A$ by one adaptive step moves the optimum to a lower-$f_P$
candidate, so $\lambda_A$ is about $6.923$. Values are synthetic; they are not
ECG or clinical results.

$f_A$ here is **auditability**: continuity, compactness and contrastivity of
attribution maps. It does not establish faithfulness to the model or agreement
with clinical knowledge.

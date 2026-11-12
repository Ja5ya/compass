#!/usr/bin/env bash
# One-command COMPASS core demo (no GPU / TensorFlow / ECG download).
# Fabricated Layer-2 tables → real Layer-3 analysis.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export COMPASS_CONFIG="${COMPASS_CONFIG:-$ROOT/configs/synthetic_demo.yaml}"
export COMPASS_DATA_ROOT="${COMPASS_DATA_ROOT:-$ROOT/examples/synthetic/.run}"
N_BOOTSTRAP="${N_BOOTSTRAP:-200}"
SEED="${SEED:-42}"
EXPECTED_DIR="$ROOT/examples/synthetic/expected"
UPDATE_EXPECTED="${UPDATE_EXPECTED:-0}"

echo "COMPASS synthetic demo (fabricated inputs → real Layer-3 core)"
echo "  config:    $COMPASS_CONFIG"
echo "  data_root: $COMPASS_DATA_ROOT"
echo "  bootstrap: N=$N_BOOTSTRAP  seed=$SEED"
echo "  note:      metrics below are synthetic, not ECG results"

START_TS=$(date +%s)

python3 examples/synthetic/make_synthetic.py \
  --config "$COMPASS_CONFIG" \
  --data-root "$COMPASS_DATA_ROOT" \
  --seed "$SEED"

PYTHONPATH=src python3 examples/synthetic/run_core.py \
  --config "$COMPASS_CONFIG" \
  --data-root "$COMPASS_DATA_ROOT" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo
echo "Wall-clock runtime: ${ELAPSED}s"

ANALYSIS="$COMPASS_DATA_ROOT/analysis"
mkdir -p "$EXPECTED_DIR"

if [[ "$UPDATE_EXPECTED" == "1" ]]; then
  echo "Updating committed expected outputs..."
  python3 - "$ANALYSIS" "$EXPECTED_DIR" "$ELAPSED" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

analysis, expected = Path(sys.argv[1]), Path(sys.argv[2])
elapsed = int(sys.argv[3])
df = pd.read_csv(analysis / "three_objective_dataset.csv")
keep = [
    "psych_code", "architecture", "learning_rate", "dropout_rate",
    "f_P", "f_S", "f_A",
    "test_acc_mean", "test_precision_mean", "test_recall_mean", "test_specificity_mean",
    "test_acc_std", "test_precision_std", "test_recall_std", "test_specificity_std",
    "continuity_valid", "compactness_valid", "contrastivity_valid",
]
cols = [c for c in keep if c in df.columns]
out = df[cols].sort_values(["learning_rate", "dropout_rate"]).reset_index(drop=True)
for c in out.select_dtypes(include="number").columns:
    out[c] = out[c].round(6)
out.to_csv(expected / "three_objective_dataset.csv", index=False)

frontier = pd.read_csv(analysis / "frontier_points.csv")
for c in frontier.select_dtypes(include="number").columns:
    frontier[c] = frontier[c].round(6)
frontier.to_csv(expected / "frontier_points.csv", index=False)

summary = json.loads((analysis / "demo_summary.json").read_text())
summary.pop("artefacts", None)
summary["runtime_seconds_last_update"] = elapsed
(expected / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

shadow = json.loads((analysis / "shadow_prices.json").read_text())
(expected / "shadow_prices.json").write_text(json.dumps(shadow, indent=2) + "\n")
print(f"Wrote expected tables under {expected}")
PY
fi

echo
echo "Comparing against expected/ ..."
python3 - "$ANALYSIS" "$EXPECTED_DIR" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

analysis, expected = Path(sys.argv[1]), Path(sys.argv[2])
exp = expected / "three_objective_dataset.csv"
got = analysis / "three_objective_dataset.csv"
if not exp.is_file():
    print(f"No expected file yet at {exp}")
    print("Re-run with UPDATE_EXPECTED=1 after inspecting outputs.")
    sys.exit(1)

a = pd.read_csv(exp).sort_values(["learning_rate", "dropout_rate"]).reset_index(drop=True)
b = pd.read_csv(got).sort_values(["learning_rate", "dropout_rate"]).reset_index(drop=True)
cols = [c for c in a.columns if c in b.columns]

# f_A / join keys are deterministic fabrications → tight tol.
# Bootstrap means/stds / derived f_P,f_S are seeded but may drift slightly
# across NumPy builds → looser tol.
tight = {
    "continuity_valid", "compactness_valid", "contrastivity_valid", "f_A",
    "learning_rate", "dropout_rate",
}
ok = True
for c in cols:
    if pd.api.types.is_numeric_dtype(a[c]):
        rtol = atol = (1e-5 if c in tight else 5e-3)
        if not np.allclose(
            a[c].to_numpy(dtype=float),
            b[c].to_numpy(dtype=float),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ):
            print(f"MISMATCH column {c} (rtol={rtol}, atol={atol})")
            print("  expected:", a[c].tolist())
            print("  got:     ", b[c].tolist())
            ok = False
    else:
        if list(a[c].astype(str)) != list(b[c].astype(str)):
            print(f"MISMATCH column {c}")
            ok = False
if not ok:
    sys.exit(1)
print(f"OK: {got.name} matches expected/ ({len(a)} rows; seeded bootstrap, tolerant check)")
print(f"Frontier / shadow prices: {analysis}/")
PY

echo
echo "Done in ${ELAPSED}s. Inspect:"
echo "  $ANALYSIS/three_objective_dataset.csv"
echo "  $ANALYSIS/frontier_points.csv"
echo "  $ANALYSIS/shadow_prices.json"

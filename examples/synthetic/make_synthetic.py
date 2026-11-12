#!/usr/bin/env python3
"""Generate fabricated Layer-2 artefacts for the COMPASS core laptop demo.

Writes, under COMPASS_DATA_ROOT (default: examples/synthetic/.run):

  hyper_sweep/DEMO/st_cnn/<lr>_<dr>/test_predictions.npz
  hyper_sweep/DEMO/st_cnn/<lr>_<dr>/metrics.json
  xai_results/hyper_sweep_fa/sweep_fa_DEMO.csv

These six runs are fabricated by this script — not produced by
``train_multi_f_array.py`` or ``run_xai.py``. No TensorFlow, GPU or ECG data.
Outputs are deterministic given --seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from compass_analysis import compute_metrics, dr_tag, lr_tag  # noqa: E402
from pipeline_config import load_config  # noqa: E402

CODE = "DEMO"
ARCH = "st_cnn"
N_TEST = 80

# Controlled quality per (lr, dr).
# target_acc / margin shape bootstrap means and SDs; contin/compac/contra set f_A.
# Design intent after scoring (approx.):
#   lr5e-4_dr0.3  high f_P (~1.0),  mid f_A (0.50)   — performance winner
#   lr1e-4_dr0.1  mid-high f_P,     f_A 0.60         — staircase step
#   lr5e-4_dr0.1  lower f_P (~0.8), high f_A (0.72)  — accountability winner
#   others        weaker; one row fails recall_min
PROFILE: dict[tuple[float, float], dict[str, float]] = {
    (1e-4, 0.1): {
        "target_acc": 0.88,
        "margin": 0.22,
        "contin": 0.66,
        "compac": 0.62,
        "contra": 0.52,  # f_A = 0.60
    },
    (1e-4, 0.3): {
        "target_acc": 0.78,
        "margin": 0.14,
        "contin": 0.55,
        "compac": 0.58,
        "contra": 0.46,  # f_A = 0.53
    },
    (1e-4, 0.5): {
        "target_acc": 0.68,
        "margin": 0.08,
        "contin": 0.42,
        "compac": 0.48,
        "contra": 0.36,  # f_A = 0.42
    },
    # Highest accountability, deliberately weaker performance (~0.8 after minmax)
    # Larger margin keeps bootstrap SD (and thus f_S) high enough to stay feasible
    # at the demo's binding K_S.
    (5e-4, 0.1): {
        "target_acc": 0.82,
        "margin": 0.28,
        "contin": 0.80,
        "compac": 0.74,
        "contra": 0.62,  # f_A = 0.72
    },
    # Best performance, mid accountability — binding trade-off vs row above
    (5e-4, 0.3): {
        "target_acc": 0.95,
        "margin": 0.32,
        "contin": 0.54,
        "compac": 0.52,
        "contra": 0.44,  # f_A = 0.50
    },
    # Intentionally weak recall → dropped by analysis.recall_min
    (5e-4, 0.5): {
        "target_acc": 0.50,
        "margin": 0.10,
        "contin": 0.30,
        "compac": 0.40,
        "contra": 0.20,
        "force_low_recall": 1.0,
    },
}


def _predictions(
    rng: np.random.Generator,
    target_acc: float,
    margin: float,
    force_low_recall: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fabricate scores with roughly ``target_acc`` at threshold 0.5.

    Larger ``margin`` → scores farther from 0.5 → smaller bootstrap SD.
    """
    y_true = np.array([0] * (N_TEST // 2) + [1] * (N_TEST // 2), dtype=int)
    n_err = int(round((1.0 - target_acc) * N_TEST))
    n_err = max(0, min(N_TEST, n_err))
    err_idx = rng.choice(N_TEST, size=n_err, replace=False) if n_err else np.array([], dtype=int)
    correct = np.ones(N_TEST, dtype=bool)
    correct[err_idx] = False

    y_score = np.empty(N_TEST, dtype=np.float64)
    for i in range(N_TEST):
        jitter = 0.04 * rng.standard_normal()
        if correct[i]:
            # Correct side of the threshold, distance ≈ margin
            if y_true[i] == 1:
                y_score[i] = 0.5 + margin + jitter
            else:
                y_score[i] = 0.5 - margin + jitter
        else:
            # Wrong side
            if y_true[i] == 1:
                y_score[i] = 0.5 - margin + jitter
            else:
                y_score[i] = 0.5 + margin + jitter
    y_score = np.clip(y_score, 0.02, 0.98)

    if force_low_recall:
        y_score = y_score.copy()
        y_score[y_true == 1] = rng.uniform(0.05, 0.35, size=int((y_true == 1).sum()))
        y_score[y_true == 0] = rng.uniform(0.05, 0.45, size=int((y_true == 0).sum()))
    return y_true, y_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_REPO / "configs" / "synthetic_demo.yaml"),
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Writable root (default: COMPASS_DATA_ROOT or examples/synthetic/.run)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = (
        args.data_root
        or os.environ.get("COMPASS_DATA_ROOT")
        or str(_REPO / "examples" / "synthetic" / ".run")
    )
    os.environ["COMPASS_DATA_ROOT"] = data_root
    cfg = load_config(args.config)
    cfg.output_root = data_root

    root = Path(data_root)
    sweep_root = root / "hyper_sweep"
    fa_dir = root / "xai_results" / "hyper_sweep_fa"
    fa_dir.mkdir(parents=True, exist_ok=True)

    print("Fabricating six synthetic Layer-2 runs (not trained / not from run_xai.py)")
    fa_rows: list[dict] = []
    for entry in cfg.build_sweep_grid():
        lr = float(entry["learning_rate"])
        dr = float(entry["dropout_rate"])
        key = (lr, dr)
        if key not in PROFILE:
            raise KeyError(f"No synthetic profile for lr={lr}, dr={dr}")
        prof = PROFILE[key]
        # Per-config RNG so grid edits do not reshuffle other rows.
        rng = np.random.default_rng(args.seed + int(lr * 1e7) + int(dr * 1000))
        y_true, y_score = _predictions(
            rng,
            target_acc=prof["target_acc"],
            margin=prof["margin"],
            force_low_recall=bool(prof.get("force_low_recall")),
        )
        test_m = compute_metrics(y_true, y_score, threshold=cfg.training.threshold)

        run_dir = sweep_root / CODE / ARCH / f"{lr_tag(lr)}_{dr_tag(dr)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run_dir / "test_predictions.npz", y_true=y_true, y_score=y_score)

        metrics = {
            "task_id": 0,
            "psych_code": CODE,
            "architecture": ARCH,
            "learning_rate": lr,
            "dropout_rate": dr,
            "epochs_ran": 1,
            "threshold": cfg.training.threshold,
            "random_state": args.seed,
            "splits": {"train": 160, "val": 40, "test": int(len(y_true))},
            "val": {k: float(v) for k, v in test_m.items()},
            "test": {k: float(v) for k, v in test_m.items()},
            "synthetic": True,
        }
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        cont = float(prof["contin"])
        comp = float(prof["compac"])
        contra = float(prof["contra"])
        f_a = (cont + comp + contra) / 3.0
        sweep_run = f"{lr_tag(lr)}_{dr_tag(dr)}"
        fa_rows.append(
            {
                "code": CODE,
                "sweep_arch": ARCH,
                "sweep_run": sweep_run,
                "f_A": f_a,
                "continuity_valid": cont,
                "compactness_valid": comp,
                "contrastivity_valid": contra,
                "continuity_sg": cont,
                "continuity_gc": cont,
                "continuity_sh_nondegen": cont,
                "continuity_lm": cont,
                "compactness_sg": comp,
                "compactness_gc": comp,
                "compactness_sh_nondegen": comp,
                "compactness_lime_ref": 0.525,
                "contrastivity_sg": contra,
                "contrastivity_gc": contra,
                "contrastivity_sh_nondegen": contra,
                "contrastivity_lm": contra,
                "n_shap_degenerate": 0,
                "n_shap_degenerate_pct": 0.0,
            }
        )
        print(
            f"  [fabricated] {sweep_run}: "
            f"point_recall={test_m['recall']:.3f} "
            f"point_acc={test_m['acc']:.3f} f_A={f_a:.3f}"
        )

    fa_path = fa_dir / f"sweep_fa_{CODE}.csv"
    pd.DataFrame(fa_rows).to_csv(fa_path, index=False)
    print(f"Wrote {len(fa_rows)} fabricated runs under {sweep_root}")
    print(f"Wrote fabricated {fa_path}")


if __name__ == "__main__":
    main()

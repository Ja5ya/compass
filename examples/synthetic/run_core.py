#!/usr/bin/env python3
"""Run COMPASS core stages 4–5 on fabricated synthetic artefacts.

Reports a binding shadow-price point and a near-cliff point so the demo
exercises non-zero λ and an infeasible step (cliff).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from compass_analysis import (  # noqa: E402
    compute_feasibility_frontier,
    load_analysis_config,
    load_three_objective_dataset,
    merge_sweep_fa_csvs,
    reliable_shadow_price,
    resolve_paths,
    run_bootstrap_pipeline,
)

# Default operating points chosen for the fabricated DEMO grid:
#   binding: K_A above the performance-winner's f_A (0.50) so accountability binds;
#            tightening drops from the f_A=0.60 step to the f_A=0.72 / lower-f_P step
#   cliff:   K_A just below the unique max f_A (0.72); one adaptive step empties F
DEFAULT_BINDING = (0.20, 0.55)
DEFAULT_CLIFF = (0.20, 0.70)


def _fmt_lambda(v: float) -> str:
    """Human-readable λ; cliffs are undefined finite differences."""
    if isinstance(v, float) and v == float("inf"):
        return "null(cliff)"
    return str(v)


def _jsonable_shadow(shadow: dict | None) -> dict | None:
    """Strict-JSON shadow payload: cliffs are null, never Infinity/NaN."""
    if shadow is None:
        return None
    clean: dict = {}
    for k, v in shadow.items():
        if k == "best":
            continue
        if isinstance(v, float) and (v == float("inf") or v != v):
            # null = undefined finite difference (cliff) or NaN
            clean[k] = None
        elif isinstance(v, (float, int, bool)):
            clean[k] = float(v) if isinstance(v, float) else v
        else:
            clean[k] = str(v)
    return clean


def _print_shadow(label: str, k_s: float, k_a: float, shadow: dict | None) -> None:
    if shadow is None:
        print(f"{label} @ ({k_s}, {k_a}): infeasible (empty set)")
        return
    print(
        f"{label} @ ({k_s}, {k_a}): "
        f"fP*={shadow['fP_star']:.4f}  "
        f"λ_S={_fmt_lambda(shadow['lambda_S'])}  "
        f"λ_A={_fmt_lambda(shadow['lambda_A'])}  "
        f"δ_A={shadow['delta_A']:.4g}  n_feas={shadow['n_feasible']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_REPO / "configs" / "synthetic_demo.yaml"),
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed")
    parser.add_argument("--k-s", type=float, default=DEFAULT_BINDING[0])
    parser.add_argument("--k-a", type=float, default=DEFAULT_BINDING[1])
    parser.add_argument("--cliff-k-s", type=float, default=DEFAULT_CLIFF[0])
    parser.add_argument("--cliff-k-a", type=float, default=DEFAULT_CLIFF[1])
    args = parser.parse_args()

    data_root = (
        args.data_root
        or os.environ.get("COMPASS_DATA_ROOT")
        or str(_REPO / "examples" / "synthetic" / ".run")
    )
    os.environ["COMPASS_DATA_ROOT"] = data_root

    cfg = load_analysis_config(args.config, data_root)
    cfg.output_root = data_root
    paths = resolve_paths(cfg)

    print("== merge-fa ==")
    merge_sweep_fa_csvs(cfg)

    print(f"== bootstrap (N={args.n_bootstrap}, seed={args.seed}) ==")
    run_bootstrap_pipeline(
        cfg,
        n_bootstrap=args.n_bootstrap,
        random_state=args.seed,
        progress_every=1,
    )

    print("== three-objective dataset (fabricated inputs) ==")
    df = load_three_objective_dataset(cfg)
    print(
        df[
            [
                "psych_code",
                "architecture",
                "learning_rate",
                "dropout_rate",
                "f_P",
                "f_S",
                "f_A",
                "test_recall_mean",
            ]
        ].to_string(index=False)
    )

    code = cfg.diagnoses.codes[0]
    df_code = df[df["psych_code"] == code].copy()
    frontier_df, pareto = compute_feasibility_frontier(df_code, n_grid=40)
    shadow_bind = reliable_shadow_price(df_code, args.k_s, args.k_a)
    shadow_cliff = reliable_shadow_price(df_code, args.cliff_k_s, args.cliff_k_a)

    out_dir = paths["analysis_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    three_path = paths["three_obj_csv"]
    frontier_path = out_dir / "frontier_points.csv"
    shadow_path = out_dir / "shadow_prices.json"
    summary_path = out_dir / "demo_summary.json"

    pareto.to_csv(frontier_path, index=False)

    shadow_payload = {
        "psych_code": code,
        "binding": {
            "K_S": args.k_s,
            "K_A": args.k_a,
            "result": _jsonable_shadow(shadow_bind),
        },
        "cliff": {
            "K_S": args.cliff_k_s,
            "K_A": args.cliff_k_a,
            "result": _jsonable_shadow(shadow_cliff),
        },
        "note": (
            "Inputs are fabricated by make_synthetic.py; metrics are not "
            "empirical ECG results. lambda_S/lambda_A are JSON null when the "
            "finite difference is undefined (cliff)."
        ),
    }
    with open(shadow_path, "w", encoding="utf-8") as f:
        json.dump(shadow_payload, f, indent=2)

    unique_fa = sorted(float(x) for x in df_code["f_A"].dropna().unique())
    summary = {
        "n_configs_scored": int(len(df)),
        "n_frontier_points": int(len(pareto)),
        "n_feasible_grid": int(frontier_df["feasible"].sum()),
        "unique_f_A": unique_fa,
        "f_P_range": [float(df["f_P"].min()), float(df["f_P"].max())],
        "f_S_range": [float(df["f_S"].min()), float(df["f_S"].max())],
        "f_A_range": [float(df["f_A"].min()), float(df["f_A"].max())],
        "shadow_prices": shadow_payload,
        "artefacts": {
            "three_objective_dataset": str(three_path),
            "frontier_points": str(frontier_path),
            "shadow_prices": str(shadow_path),
            "objective_inputs": str(paths["objective_inputs_csv"]),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {three_path}")
    print(f"Wrote {frontier_path}")
    print(f"Wrote {shadow_path}")
    print(f"Wrote {summary_path}")
    _print_shadow("Binding", args.k_s, args.k_a, shadow_bind)
    _print_shadow("Cliff", args.cliff_k_s, args.cliff_k_a, shadow_cliff)
    print(f"Unique fabricated f_A levels (staircase steps): {unique_fa}")


if __name__ == "__main__":
    main()

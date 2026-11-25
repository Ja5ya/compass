#!/usr/bin/env python3
"""Apply the COMPASS selection core to a plain candidate table.

Reads ``candidates.csv`` (one row per model with ``f_P``, ``f_S``, ``f_A``),
selects the best-performing feasible configuration at ``(K_S, K_A)``, and
reports shadow prices and a short feasibility frontier summary.

Fabricated inputs only — no training, attributions or patient data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from compass_analysis import (  # noqa: E402
    compute_feasibility_frontier,
    reliable_shadow_price,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "candidates.csv"
DEFAULT_KS = 0.65
DEFAULT_KA = 0.54


def _fmt_lambda(v: float) -> str:
    if isinstance(v, float) and v == float("inf"):
        return "null(cliff)"
    return f"{v:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--k-s", type=float, default=DEFAULT_KS)
    parser.add_argument("--k-a", type=float, default=DEFAULT_KA)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"model", "f_P", "f_S", "f_A"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"candidates.csv missing columns: {sorted(missing)}")

    feas = df[(df["f_S"] >= args.k_s) & (df["f_A"] >= args.k_a)]
    print(f"Candidates: {len(df)}  |  Feasible at (K_S={args.k_s}, K_A={args.k_a}): {len(feas)}")
    if len(feas) == 0:
        print("No feasible configuration.")
        return

    best = feas.loc[feas["f_P"].idxmax()]
    print(
        f"Selected model: {best['model']}  "
        f"(f_P={best['f_P']:.4g}, f_S={best['f_S']:.4g}, f_A={best['f_A']:.4g})"
    )
    print(
        "Note: f_A is auditability (continuity, compactness, contrastivity of "
        "attribution maps); it does not establish faithfulness."
    )

    shadow = reliable_shadow_price(df, args.k_s, args.k_a)
    if shadow is None:
        print("Shadow prices: infeasible")
    else:
        print(
            f"Shadow prices: λ_S={_fmt_lambda(shadow['lambda_S'])}  "
            f"λ_A={_fmt_lambda(shadow['lambda_A'])}  "
            f"δ_A={shadow['delta_A']:.4g}  n_feas={shadow['n_feasible']}"
        )

    frontier_df, pareto = compute_feasibility_frontier(df, n_grid=40)
    n_front = int(pareto["on_frontier"].sum()) if len(pareto) else 0
    print(
        f"Feasibility frontier: {n_front} staircase points "
        f"(grid size {len(frontier_df)}; fabricated table only)."
    )


if __name__ == "__main__":
    main()

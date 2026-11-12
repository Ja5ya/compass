#!/bin/bash
# Merge per-code sweep_fa_<CODE>.csv into sweep_fa_summary.csv (after XAI).
#
#SBATCH --job-name=compass_merge_fa
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/merge_fa_%j.out
#SBATCH --error=slurm_logs/merge_fa_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit with: sbatch --chdir=\$REPO_ROOT slurm/merge_fa.slurm}"
SRC_ROOT="${REPO_ROOT}/src"

# shellcheck source=common.sh
source "${REPO_ROOT}/slurm/common.sh"
compass_source_env_if_needed "${REPO_ROOT}"
compass_resolve_config "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/slurm_logs"

VENV="${COMPASS_VENV:-${HOME}/myenv}"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

python "${SRC_ROOT}/compass_analysis.py" --config "${CONFIG}" merge-fa
echo "Finished merge-fa at $(date)"

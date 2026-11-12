#!/bin/bash
# Stage 4: bootstrap f_P / f_S from saved test_predictions.npz (CPU only).
#
# Submit standalone:
#   bash slurm/submit_bootstrap.sh
#
# Or automatically chained after training via slurm/submit_train.sh
#
#SBATCH --job-name=compass_bootstrap
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/bootstrap_%j.out
#SBATCH --error=slurm_logs/bootstrap_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit with: sbatch --chdir=\$REPO_ROOT slurm/bootstrap.slurm}"
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

N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

echo "Job ID=${SLURM_JOB_ID} config=${CONFIG} N_BOOTSTRAP=${N_BOOTSTRAP}"
python "${SRC_ROOT}/compass_analysis.py" \
  --config "${CONFIG}" \
  bootstrap \
  --n-bootstrap "${N_BOOTSTRAP}"
echo "Finished bootstrap at $(date)"

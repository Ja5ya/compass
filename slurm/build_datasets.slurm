#!/bin/bash
# Submit from repository root:
#   export COMPASS_CONFIG=configs/dataset.template.yaml
#   export COMPASS_DATA_ROOT=/path/to/writable/run
#   sbatch slurm/build_datasets.slurm
#
#SBATCH --job-name=compass_build
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/build_datasets_%j.out
#SBATCH --error=slurm_logs/build_datasets_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
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

python "${SRC_ROOT}/build_datasets.py" --config "${CONFIG}"
echo "Finished build at $(date)"

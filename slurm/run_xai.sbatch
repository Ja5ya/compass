#!/bin/bash
#SBATCH --job-name=compass_xai
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/xai_%A_%a.out
#SBATCH --error=slurm_logs/xai_%A_%a.err
#SBATCH --array=0-0

set -euo pipefail

# Slurm copies this script to a spool directory — use SLURM_SUBMIT_DIR, not BASH_SOURCE.
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit with: sbatch --chdir=\$REPO_ROOT slurm/run_xai.sbatch}"
SRC_ROOT="${REPO_ROOT}/src"
MODELS_PARENT="${COMPASS_MODELS_PARENT:-${SRC_ROOT}}"
XAI_MODE="${XAI_MODE:-sweep}"

# shellcheck source=common.sh
source "${REPO_ROOT}/slurm/common.sh"
compass_source_env_if_needed "${REPO_ROOT}"
compass_resolve_config "${REPO_ROOT}"
DATA_ROOT="${COMPASS_DATA_ROOT:?Set COMPASS_DATA_ROOT (e.g. source configs/env.mimic)}"

mkdir -p "${REPO_ROOT}/slurm_logs" "${DATA_ROOT}/logs" "${DATA_ROOT}/xai_results/hyper_sweep_fa"
cd "${DATA_ROOT}"

export PYTHONPATH="${SRC_ROOT}:${MODELS_PARENT}:${PYTHONPATH:-}"
mapfile -t CODES < <(python "${SRC_ROOT}/pipeline_config.py" --config "${CONFIG}" codes)
CODE="${CODES[$SLURM_ARRAY_TASK_ID]}"

echo "Job ID=${SLURM_JOB_ID} Task=${SLURM_ARRAY_TASK_ID} CODE=${CODE}"

VENV="${COMPASS_VENV:-${HOME}/myenv}"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
fi

export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1

N_SAMPLES="${N_SAMPLES:-}"
SHAP_BG="${SHAP_BG:-}"
XAI_ARGS=(--config "${CONFIG}" --data-root "${DATA_ROOT}" --code "${CODE}")
[[ -n "${N_SAMPLES}" ]] && XAI_ARGS+=(--n_samples "${N_SAMPLES}")
[[ -n "${SHAP_BG}" ]] && XAI_ARGS+=(--shap_bg "${SHAP_BG}")

if [[ "${XAI_MODE}" == "pareto" ]]; then
  python "${SRC_ROOT}/run_xai.py" "${XAI_ARGS[@]}"
else
  SWEEP_FA_CSV="${DATA_ROOT}/xai_results/hyper_sweep_fa/sweep_fa_${CODE}.csv"
  python "${SRC_ROOT}/run_xai.py" \
    --hyper_sweep_fa \
    "${XAI_ARGS[@]}" \
    --sweep_fa_csv "${SWEEP_FA_CSV}"
fi

echo "Finished task ${SLURM_ARRAY_TASK_ID} (${CODE}) at $(date)"

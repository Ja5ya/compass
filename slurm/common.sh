#!/bin/bash
# Shared helpers for COMPASS Slurm scripts.

# Load env file only when COMPASS_CONFIG is not already set.
compass_source_env_if_needed() {
  local repo_root="$1"
  local env_file="${2:-${repo_root}/configs/env.ecgpsych}"
  if [[ -z "${COMPASS_CONFIG:-}" && -f "${env_file}" ]]; then
    # shellcheck source=/dev/null
    source "${env_file}"
  fi
}

# Resolve COMPASS config to an absolute path (relative paths are under repo_root).
# Sets CONFIG and exports COMPASS_CONFIG. Exits with an error if the file is missing.
compass_resolve_config() {
  local repo_root="$1"
  local config="${COMPASS_CONFIG:-${repo_root}/configs/ecgpsych.yaml}"
  if [[ "${config}" != /* ]]; then
    config="${repo_root}/${config}"
  fi
  if [[ ! -f "${config}" ]]; then
    echo "ERROR: Config not found: ${config}" >&2
    echo "  COMPASS_CONFIG=${COMPASS_CONFIG:-<unset>}" >&2
    echo "  repo_root=${repo_root}" >&2
    echo "  Hint: relative COMPASS_CONFIG paths must live under the repository (e.g. configs/mimic.yaml)." >&2
    exit 1
  fi
  CONFIG="${config}"
  export COMPASS_CONFIG="${CONFIG}"
}

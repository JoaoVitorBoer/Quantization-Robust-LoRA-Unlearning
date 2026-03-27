#!/bin/bash

#SBATCH --output=/home/joaoabitante/Sout/%j__%x.out
#SBATCH --error=/home/joaoabitante/Sout/%j__%x.out

#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=52G
#SBATCH --time=2-00:00:00
#SBATCH --gpus=2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_dir() {
  local candidate=""

  # Best signal when running with sbatch: original submission directory.
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    candidate="${SLURM_SUBMIT_DIR}"
    if [[ -f "${candidate}/calculate_model_deltas_by_layer.py" ]]; then
      echo "${candidate}"
      return 0
    fi
  fi

  # Common local execution paths.
  for candidate in "${PWD}" "${SCRIPT_DIR}" "${SCRIPT_DIR}/.."; do
    if [[ -f "${candidate}/calculate_model_deltas_by_layer.py" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

if ! REPO_DIR="$(find_repo_dir)"; then
  echo "Error: could not locate repository root containing calculate_model_deltas_by_layer.py." >&2
  echo "Hint: run from repo root, or submit with sbatch from repo root so SLURM_SUBMIT_DIR is set." >&2
  exit 1
fi

cd "${REPO_DIR}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: python3/python not found in PATH." >&2
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-saves/unlearn/norm_calculation}"
OUTPUT_CSV="${OUTPUT_CSV:-${ROOT_DIR}/delta_frobenius_norms_by_layer.csv}"
DTYPE="${DTYPE:-float32}"
CACHE_DIR="${CACHE_DIR:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
FAIL_FAST="${FAIL_FAST:-false}"
TARGET_LAYERS="${TARGET_LAYERS:-[\"q_proj\",\"v_proj\",\"k_proj\",\"o_proj\",\"gate_proj\",\"down_proj\",\"up_proj\"]}"

cmd=(
  "${PYTHON_BIN}" "${REPO_DIR}/calculate_model_deltas_by_layer.py"
  "--root-dir" "${ROOT_DIR}"
  "--output-csv" "${OUTPUT_CSV}"
  "--dtype" "${DTYPE}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  cmd+=("--cache-dir" "${CACHE_DIR}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  cmd+=("--trust-remote-code")
fi

if [[ "${FAIL_FAST}" == "true" ]]; then
  cmd+=("--fail-fast")
fi

if [[ -n "${TARGET_LAYERS}" ]]; then
  cmd+=("--target-layers" "${TARGET_LAYERS}")
fi

# Optional extra args passed directly to calculate_model_deltas_by_layer.py
cmd+=("$@")

echo "Running: ${cmd[*]}"
"${cmd[@]}"

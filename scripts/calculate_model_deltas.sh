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
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
OUTPUT_CSV="${OUTPUT_CSV:-${ROOT_DIR}/delta_frobenius_norms.csv}"
DTYPE="${DTYPE:-float16}"
CACHE_DIR="${CACHE_DIR:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
FAIL_FAST="${FAIL_FAST:-false}"

cmd=(
  "${PYTHON_BIN}" "calculate_model_deltas.py"
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

# Optional extra args passed directly to calculate_model_deltas.py
cmd+=("$@")

echo "Running: ${cmd[*]}"
"${cmd[@]}"

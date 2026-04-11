#!/usr/bin/env bash
set -euo pipefail

RAVP_DIR="${RAVP_DIR:-/my_workspace/Resume/RAVP}"
DATASET_ROOT="${DATASET_ROOT:-/my_workspace/Resume/RAVP_Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${RAVP_DIR}/target_pose_runs}"

source_conda() {
  eval "$(/root/miniconda3/bin/conda shell.bash hook)"
  conda activate ravp
}

cd "${RAVP_DIR}"
source_conda

for label_source in gt pred; do
  echo "Training target pose model with label source: ${label_source}"
  python -m target_pose_regression.train \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --label-source "${label_source}" \
    "$@"
done

#!/usr/bin/env bash
set -euo pipefail

RAVP_DIR="${RAVP_DIR:-/my_workspace/Resume/RAVP}"
OUTPUT_DIR="${OUTPUT_DIR:-/my_workspace/Resume/RAVP_Dataset_Train_AllButTown10HD_Compact}"
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2150}"
HELD_OUT_TOWN="${HELD_OUT_TOWN:-Town10HD}"
TARGET_FRAMES_PER_TOWN="${TARGET_FRAMES_PER_TOWN:-10000}"
TARGET_SAMPLES_PER_TOWN="${TARGET_SAMPLES_PER_TOWN:-1000000}"
NUM_TRAFFIC_VEHICLES="${NUM_TRAFFIC_VEHICLES:-80}"
EPISODE_FRAME_BUDGET="${EPISODE_FRAME_BUDGET:-4000}"
MAX_EPISODES_PER_RUN="${MAX_EPISODES_PER_RUN:-25}"
IMAGE_WIDTH="${IMAGE_WIDTH:-768}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-768}"
RGB_JPEG_QUALITY="${RGB_JPEG_QUALITY:-95}"
SAM3_DEVICE="${SAM3_DEVICE:-cuda:0}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-cuda:0}"
LOG_DIR="${LOG_DIR:-/my_workspace/Resume/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/collect_train_towns.log}"

TRAIN_TOWNS=(
  Town01 Town01_Opt
  Town02 Town02_Opt
  Town03 Town03_Opt
  Town04 Town04_Opt
  Town05 Town05_Opt
  Town10HD Town10HD_Opt
)

mkdir -p "${LOG_DIR}"
touch "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

wait_for_carla() {
  while ! python3 - "${CARLA_HOST}" "${CARLA_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
  do
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] waiting for CARLA at ${CARLA_HOST}:${CARLA_PORT}"
    sleep 5
  done
}

existing_frame_count() {
  local town="$1"
  python3 - "${OUTPUT_DIR}" "${town}" <<'PY'
import json
import os
import sys

output_dir = sys.argv[1]
town = sys.argv[2]
manifest_path = os.path.join(output_dir, "frames.jsonl")
count = 0
if os.path.exists(manifest_path):
    with open(manifest_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if payload.get("town") == town:
                count += 1
print(count)
PY
}

is_held_out_town() {
  local town="$1"
  local held_out_family="${HELD_OUT_TOWN%_Opt}"
  local town_family="${town%_Opt}"
  [[ "${town_family}" == "${held_out_family}" ]]
}

source_conda() {
  eval "$(/root/miniconda3/bin/conda shell.bash hook)"
  conda activate ravp
}

cd "${RAVP_DIR}"
source_conda

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] held-out town family: ${HELD_OUT_TOWN}"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] output dir: ${OUTPUT_DIR}"

for town in "${TRAIN_TOWNS[@]}"; do
  if is_held_out_town "${town}"; then
    continue
  fi

  while true; do
    current_frames="$(existing_frame_count "${town}")"
    if [[ "${current_frames}" -ge "${TARGET_FRAMES_PER_TOWN}" ]]; then
      echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ${town} already has ${current_frames} frames; skipping"
      break
    fi

    wait_for_carla

    fresh_args=()
    if [[ ! -f "${OUTPUT_DIR}/frames.jsonl" ]]; then
      fresh_args=(--fresh)
    fi

    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] starting ${town} with ${current_frames}/${TARGET_FRAMES_PER_TOWN} frames collected so far"
    set +e
    python -m carla_data_collection.run_collection collect-dataset \
      --output-dir "${OUTPUT_DIR}" \
      --carla-host "${CARLA_HOST}" \
      --carla-port "${CARLA_PORT}" \
      --towns "${town}" \
      --follow-only \
      --min-follow-actors-per-frame 1 \
      --max-follow-actors-per-frame 4 \
      --follow-lateral-limit-m 12 \
      --follow-yaw-limit-deg 120 \
      --target-samples-per-town "${TARGET_SAMPLES_PER_TOWN}" \
      --max-frames-per-town "${TARGET_FRAMES_PER_TOWN}" \
      --num-traffic-vehicles "${NUM_TRAFFIC_VEHICLES}" \
      --traffic-mode traffic_manager \
      --max-episodes-per-town "${MAX_EPISODES_PER_RUN}" \
      --episode-frame-budget "${EPISODE_FRAME_BUDGET}" \
      --image-width "${IMAGE_WIDTH}" \
      --image-height "${IMAGE_HEIGHT}" \
      --rgb-jpeg-quality "${RGB_JPEG_QUALITY}" \
      --sam3-device "${SAM3_DEVICE}" \
      --detector-device "${DETECTOR_DEVICE}" \
      "${fresh_args[@]}"
    exit_code=$?
    set -e

    if [[ "${exit_code}" == "0" ]]; then
      current_frames="$(existing_frame_count "${town}")"
      if [[ "${current_frames}" -ge "${TARGET_FRAMES_PER_TOWN}" ]]; then
        echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ${town} complete with ${current_frames} frames"
        break
      fi
      echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ${town} returned cleanly but only has ${current_frames}/${TARGET_FRAMES_PER_TOWN} frames; retrying"
    else
      echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ${town} collection exited with code ${exit_code}; retrying after CARLA becomes available"
    fi

    sleep 10
  done
done

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] collection loop finished"

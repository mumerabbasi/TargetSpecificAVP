#!/usr/bin/env bash
set -euo pipefail

CARLA_DIR="${CARLA_DIR:-/my_workspace/Resume/CARLA}"
CARLA_RPC_PORT="${CARLA_RPC_PORT:-2150}"
CARLA_STREAM_PORT="${CARLA_STREAM_PORT:-2151}"
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-5}"
LOG_DIR="${LOG_DIR:-/my_workspace/Resume/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/carla_watchdog.log}"

mkdir -p "${LOG_DIR}"
touch "${LOG_PATH}"

stop_requested=0
on_exit() {
  stop_requested=1
}
trap on_exit INT TERM

cd "${CARLA_DIR}"

while true; do
  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "[${timestamp}] starting CARLA on rpc=${CARLA_RPC_PORT} stream=${CARLA_STREAM_PORT}" | tee -a "${LOG_PATH}"

  mkdir -p /tmp/runtime-1001
  chown 1001:1001 /tmp/runtime-1001 || true

  set +e
  HOME=/tmp XDG_RUNTIME_DIR=/tmp/runtime-1001 SDL_AUDIODRIVER=dummy DISPLAY= \
    setpriv --reuid=1001 --regid=1001 --clear-groups \
    ./CarlaUE4.sh \
      -opengl \
      -RenderOffScreen \
      -quality-level=Epic \
      -carla-port="${CARLA_RPC_PORT}" \
      -carla-streaming-port="${CARLA_STREAM_PORT}" \
      -nosound >> "${LOG_PATH}" 2>&1
  exit_code=$?
  set -e

  timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "[${timestamp}] CARLA exited with code ${exit_code}" | tee -a "${LOG_PATH}"

  if [[ "${stop_requested}" == "1" ]]; then
    echo "[${timestamp}] watchdog stop requested; exiting" | tee -a "${LOG_PATH}"
    exit 0
  fi

  sleep "${RESTART_DELAY_SEC}"
done

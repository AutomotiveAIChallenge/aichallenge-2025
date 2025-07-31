#!/bin/bash

target="${1}"
device="${2}"
mode="${3}"
case "${target}" in
"eval")
    volume="output:/output"
    ;;
"dev")
    volume="output:/output aichallenge:/aichallenge remote:/remote vehicle:/vehicle"
    ;;
*)
    echo "invalid argument (use 'dev' or 'eval')"
    exit 1
    ;;
esac

case "${mode}" in
"autoware")
    cmd=" bash /aichallenge/run_autoware.bash vehicle"
    echo "[INFO] Running in Autoware mode (forced by argument)"
    ;;
"build")
    cmd=" /aichallenge/build_autoware.bash clean"
    echo "[INFO] Running in Build mode (forced by argument)"
    ;;
*)
    cmd=""
    ;;
esac

if [ "${device}" = "cpu" ]; then
    opts=""
    echo "[INFO] Running in CPU mode (forced by argument)"
elif [ "${device}" = "gpu" ]; then
    opts="--nvidia"
    echo "[INFO] Running in GPU mode (forced by argument)"
elif command -v nvidia-smi &>/dev/null && [[ -e /dev/nvidia0 ]]; then
    opts="--nvidia"
    echo "[INFO] NVIDIA GPU detected → enabling --nvidia"
else
    opts=""
    echo "[INFO] No NVIDIA GPU detected → running on CPU"
fi
mkdir -p output

LOG_DIR="output/latest"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/docker_run.log"
echo "A rocker run log is stored at : file://$LOG_FILE"

# shellcheck disable=SC2086
rocker ${opts} --x11 --devices /dev/dri --user --net host --privileged --name "aichallenge-2025-$(date "+%Y-%m-%d-%H-%M-%S")" --volume ${volume} -- "aichallenge-2025-${target}-${USER}" ${cmd} 2>&1 | tee "$LOG_FILE"

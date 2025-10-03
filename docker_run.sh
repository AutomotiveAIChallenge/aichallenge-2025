#!/bin/bash
if [ $# -lt 1 ]; then
    echo "エラー: ターゲットを指定してください。"
    echo "使用法: $0 [dev|eval|rviz|rm] [cpu|gpu] [コンテナ内で実行したいコマンド...]"
    exit 1
fi

target="${1}"
if [[ ${target} != "rviz" && ${target} != "rm" && $# -lt 2 ]]; then
    echo "エラー: 'dev' または 'eval' ターゲットにはデバイス指定 [cpu|gpu] が必要です。"
    echo "使用法: $0 ${target} [cpu|gpu]"
    exit 1
fi

device="${2}"
shift 2
CMD_TO_RUN="$*"

case "${target}" in
"eval")
    volume="output:/output"
    ;;
"dev")
    volume="output:/output aichallenge:/aichallenge remote:/remote vehicle:/vehicle"
    ;;
"rviz")
    target="dev"
    volume="output:/output aichallenge:/aichallenge remote:/remote vehicle:/vehicle"
    CMD_TO_RUN="/aichallenge/run_rviz.bash vehicle"
    echo "[INFO] RViz起動モードで実行します。"
    ;;
"rm")
    # clean up old <none> images
    docker image prune -f
    exit 1
    ;;
*)
    echo "invalid argument (use 'dev', 'rviz' 'rm' or 'eval')"
    exit 1
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

CONTAINER_NAME="aichallenge-2025-$(date "+%Y-%m-%d-%H-%M-%S")"
IMAGE_NAME="aichallenge-2025-${target}-${USER}"

if [ -n "$CMD_TO_RUN" ]; then
    # shellcheck disable=SC2086
    rocker ${opts} --x11 --devices /dev/dri --user --net host --privileged --name "${CONTAINER_NAME}" --volume ${volume} -- "${IMAGE_NAME}" /aichallenge/run_rviz.bash vehicle 2>&1 | tee "$LOG_FILE"
else
    # shellcheck disable=SC2086
    rocker ${opts} --x11 --devices /dev/dri --user --net host --privileged --name "${CONTAINER_NAME}" --volume ${volume} -- ${IMAGE_NAME} 2>&1 | tee "$LOG_FILE"
fi

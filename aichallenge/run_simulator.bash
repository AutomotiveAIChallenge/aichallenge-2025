#!/bin/bash
AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM_Multi/AWSIM

mode="${1}"
id="${2:-0}" # デフォルト値0を設定

if command -v nvidia-smi &>/dev/null && [[ -e /dev/nvidia0 ]]; then
    echo "[INFO] NVIDIA GPU detected"
    opts=()
else
    echo "[INFO] No NVIDIA GPU detected → running on headless mode"
    opts=("-headless")
fi

case "${mode}" in
"endless")
    opts+=(" --endless")
    ;;
*) ;;
esac

# shellcheck disable=SC1091
source /aichallenge/workspace/install/setup.bash
sudo ip link set multicast on lo
sudo sysctl -w net.core.rmem_max=2147483647 >/dev/null
# idが0の場合
if [ "$id" -eq 0 ]; then
    export ROS_LOCAL_HOST_ONLY=0
    export ROS_DOMAIN_ID=$id
fi
$AWSIM_DIRECTORY/AWSIM.x86_64 "${opts[@]}"

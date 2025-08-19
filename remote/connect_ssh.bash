#!/bin/bash
# Usage: ./connect_zenoh.bash <target-vehicle (A1 to A8)> <user>
if [ $# -lt 2 ]; then
    echo "エラー: 引数が不足しています。" >&2
    echo "使用法: $0 <target-vehicle (A1 to A8)> <user>" >&2
    exit 1
fi
set -e
SCRIPT_DIR=$(readlink -f "$(dirname "$0")")
IP_ADDR=$(python3 "${SCRIPT_DIR}/detail/scan_ip_addr.py" "$1")
echo "Executing command: ssh -AX $2@${IP_ADDR}"
ssh -AX "$2@${IP_ADDR}"

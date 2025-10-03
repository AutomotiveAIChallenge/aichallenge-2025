#!/bin/bash
CURRENT_DIR=$(pwd)
if [[ ! $CURRENT_DIR =~ aichallenge-2025/remote$ ]]; then
    echo "Error: This script must be run from the 'remote' directory inside 'aichallenge-2025/remote'. Current directory: $CURRENT_DIR"
    exit 1
fi
if [ $# -lt 2 ]; then
    echo "エラー: 接続先とユーザー名を指定してください。"
    echo "使用法: $0 [A2|A3|A6|A7] ユーザー名"
    exit 1
fi
./connect_ssh.bash "$1" "$2" "cd ~/aichallenge-2025/vehicle && make build-autoware"

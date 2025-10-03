#!/bin/bash

# スクリプト終了時に実行されるクリーンアップ関数
cleanup() {
    echo "スクリプトを終了します..."
    if [ -n "$CONTAINER_NAME" ]; then
        echo "Dockerコンテナ '${CONTAINER_NAME}' を停止します..."
        docker kill "$CONTAINER_NAME" >/dev/null
    fi
}

trap cleanup EXIT INT TERM

# --- 以下、元のスクリプト ---
CURRENT_DIR=$(pwd)
if [[ ! $CURRENT_DIR =~ aichallenge-2025/remote$ ]]; then
    echo "Error: This script must be run from the 'remote' directory inside 'aichallenge-2025/remote'. Current directory: $CURRENT_DIR"
    exit 1
fi
if [ $# -lt 1 ]; then
    echo "エラー: 接続先を指定してください。"
    echo "使用法: $0 [A2|A3|A6|A7]"
    exit 1
fi
cd ../

./docker_run.sh rviz cpu &

sleep 2
# 最後に起動したコンテナの「名前」を取得
CONTAINER_NAME=$(docker ps -l --format "{{.Names}}")

if [ -z "$CONTAINER_NAME" ]; then
    echo "エラー: コンテナ名の取得に失敗しました。Dockerが起動しているか確認してください。"
    exit 1
fi
echo "起動したコンテナ名: ${CONTAINER_NAME}"

echo "3秒待機しzenohに接続します..."
sleep 3

echo "zenohに接続します..."
cd remote || exit 1
./connect_zenoh.bash "$1"

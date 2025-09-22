#!/bin/bash

# 1. 引数が指定されているかチェック
if [ $# -ne 2 ]; then
    echo "エラー: 接続先とユーザー名を指定してください。"
    echo "使用法: $0 [A2|A3|A6|A7] ユーザー名"
    exit 1
fi

TARGET_ID=$1
USERNAME=$2
PORT=""

# 2. 引数に応じてポート番号を設定
case "$TARGET_ID" in
A2)
    PORT=10025
    ;;
A3)
    PORT=10024
    ;;
A6)
    PORT=10023
    ;;
A7)
    PORT=10022
    ;;
*)
    echo "エラー: 不明な接続先です: $TARGET_ID"
    echo "利用可能な接続先: A2, A3, A6, A7"
    exit 1
    ;;
esac

# 3. 選択されたポートとユーザーでautosshを実行
echo "Connecting... Target Vehicle: $TARGET_ID, User: $USERNAME"
autossh -AC -M 0 -p "$PORT" \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    "${USERNAME}@57.180.63.135"

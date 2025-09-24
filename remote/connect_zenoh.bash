##!/bin/bash

# スクリプトに引数が1つだけ渡されているかチェック
if [ "$#" -ne 1 ]; then
    echo "エラー: Vechicle IDを指定してください。" >&2
    echo "使用法: $0 {A2|A3|A6|A7}" >&2
    exit 1
fi

NAMESPACE=$1

case "$NAMESPACE" in
A2 | A3 | A6 | A7)
    echo "Connecting Zenoh. Target Vehicle：'$NAMESPACE'"
    RUST_BACKTRACE=1 zenoh-bridge-ros2dds client \
        -e tls/57.180.63.135:7447 \
        -c zenoh-user.json5 \
        -n /"$NAMESPACE"
    ;;
*)
    echo "エラー: 無効な名前空間です: '$NAMESPACE'" >&2
    echo "A2, A3, A6, A7 のいずれかを指定してください。" >&2
    exit 1
    ;;
esac

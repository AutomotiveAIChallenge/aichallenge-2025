##!/bin/bash

# スクリプトに引数が1つだけ渡されているかチェック
if [ "$#" -ne 1 ]; then
    echo "エラー: Vechicle IDを指定してください。" >&2
    echo "使用法: $0 {A2|A3|A6|A7|local}" >&2
    exit 1
fi

NAMESPACE=$1

case "$NAMESPACE" in
A2)
    echo "Connecting Zenoh. Target Vehicle: '$NAMESPACE' (ECU-RK-01) - Port 7448"
    RUST_BACKTRACE=1 zenoh-bridge-ros2dds client \
        -e tls/57.180.63.135:7448 \
        -c zenoh-user.json5
    ;;
A3)
    echo "Connecting Zenoh. Target Vehicle: '$NAMESPACE' (ECU-RK-02) - Port 7449"
    RUST_BACKTRACE=1 zenoh-bridge-ros2dds client \
        -e tls/57.180.63.135:7449 \
        -c zenoh-user.json5
    ;;
A6)
    echo "Connecting Zenoh. Target Vehicle: '$NAMESPACE' (ECU-RK-06) - Port 7450"
    RUST_BACKTRACE=1 zenoh-bridge-ros2dds client \
        -e tls/57.180.63.135:7450 \
        -c zenoh-user.json5
    ;;
A7)
    echo "Connecting Zenoh. Target Vehicle: '$NAMESPACE' (ECU-RK-00) - Port 7451"
    RUST_BACKTRACE=1 zenoh-bridge-ros2dds client \
        -e tls/57.180.63.135:7451 \
        -c zenoh-user.json5
    ;;
test-remote)
    ENDPOINT="${ZENOH_LOCAL_ENDPOINT:-tcp/127.0.0.1:7448}"
    echo "Connecting Zenoh. Target Vehicle: 'local' - Endpoint ${ENDPOINT}"
    RUST_BACKTRACE=1 zenoh-bridge-ros2dds client \
        -e "${ENDPOINT}" \
        -c zenoh-user.json5
    ;;
test-vehicle)
    ENDPOINT="${ZENOH_LOCAL_ENDPOINT:-tcp/127.0.0.1:7448}"
    echo "Connecting Zenoh. Target Vehicle: 'local' - Endpoint ${ENDPOINT}"
    RUST_BACKTRACE=1 zenoh-bridge-ros2dds client \
        -e "${ENDPOINT}" \
        -c ../vehicle/zenoh.json5
    ;;
test-server)
    zenohd --listen tcp/127.0.0.1:7448
    ;;
*)
    echo "エラー: 無効な名前空間です: '$NAMESPACE'" >&2
    echo "A2, A3, A6, A7, local のいずれかを指定してください。" >&2
    exit 1
    ;;
esac

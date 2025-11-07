#!/bin/bash

# $1 が "clean" だった場合、ビルドディレクトリを削除
if [[ ${1} == "clean" ]]; then
    echo "Cleaning build directories..."
    # rm -r ./workspace/build/* ./workspace/install/* # ディレクトリごと削除する方が安全です
    rm -rf ./workspace/build ./workspace/install ./workspace/log
    echo "Clean complete."
    # 引数を左に1つシフトする (clean を消費し、 $2 が $1 に、$3 が $2 になる)
    shift
fi

# --- 引数から UID/GID を取得 ---
# $1 (cleanの後は $2) を HOST_UID として、$2 (cleanの後は $3) を HOST_GID として受け取る
HOST_UID=$1
HOST_GID=$2
# ---

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source /autoware/install/setup.bash

cd ./workspace || exit
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# colcon build の終了コード（ステータス）を取得
BUILD_STATUS=$?

# ビルドが失敗した場合は、エラーコードで終了
if [ "$BUILD_STATUS" -ne 0 ]; then
    echo "Build failed with status ${BUILD_STATUS}. Exiting."
    exit $BUILD_STATUS
fi

echo "Build successful."

# --- ファイル所有者の変更 (引数でチェック) ---
# 引数から HOST_UID と HOST_GID が渡されているかチェック
if [ -n "$HOST_UID" ] && [ -n "$HOST_GID" ]; then

    # このスクリプトがroot (UID 0) で実行されているかチェック
    if [ "$(id -u)" -eq 0 ]; then
        echo "Running as root. Changing ownership of artifacts to ${HOST_UID}:${HOST_GID}..."
        # rootの場合のみ chown を実行
        chown -R "${HOST_UID}:${HOST_GID}" /aichallenge/workspace/build /aichallenge/workspace/install /aichallenge/workspace/log
        echo "Ownership change complete."
    else
        # root以外 (おそらく指定されたHOST_UID) で実行されている場合
        echo "Running as non-root user ($(id -u)). Files should already have correct ownership. Skipping chown."
    fi
else
    # 引数が設定されていなかった場合
    echo "HOST_UID/HOST_GID not provided as arguments. Skipping ownership change."
fi

# 正常終了
exit 0

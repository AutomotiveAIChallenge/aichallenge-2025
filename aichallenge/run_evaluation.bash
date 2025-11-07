#!/bin/bash

IS_ROSBAG_MODE=0
IS_CAPTURE_MODE=0
HOST_UID=""
HOST_GID=""
RE_NUMBER='^[0-9]+$' # 数字のみにマッチする正規表現
OTHER_ARGS=()        # rosbag/capture/数字 以外の引数を保持

# "$@" の引数をループ処理
for arg in "$@"; do
    if [ "$arg" = "rosbag" ]; then
        IS_ROSBAG_MODE=1
        continue # 次の引数へ
    fi

    if [ "$arg" = "capture" ]; then
        IS_CAPTURE_MODE=1
        continue # 次の引数へ
    fi

    # 数字のみの引数かチェック
    if [[ $arg =~ $RE_NUMBER ]]; then
        if [ -z "$HOST_UID" ]; then
            # 1つ目の数字をUIDとする
            HOST_UID=$arg
            continue
        elif [ -z "$HOST_GID" ]; then
            # 2つ目の数字をGIDとする
            HOST_GID=$arg
            continue
        fi
        # 3つ目以降の数字は無視
    fi

    # 上記のどれにも当てはまらない引数を保持（このスクリプトでは使わないが将来のため）
    OTHER_ARGS+=("$arg")
done

# デバッグ表示 (引数解析の結果)
if [ "$IS_ROSBAG_MODE" -eq 1 ]; then
    echo "ROS Bag recording mode enabled."
fi
if [ "$IS_CAPTURE_MODE" -eq 1 ]; then
    echo "Screen capture mode enabled."
fi
if [ -n "$HOST_UID" ]; then
    echo "HOST_UID set to: $HOST_UID"
fi
if [ -n "$HOST_GID" ]; then
    echo "HOST_GID set to: $HOST_GID"
fi
move_window() {
    echo "Move window"

    if ! wmctrl -l >/dev/null 2>&1; then
        echo "wmctrl command not available. Skipping window management."
        sleep 5
        return 0
    fi

    local has_gpu has_awsim has_rviz
    has_gpu=$(command -v nvidia-smi >/dev/null && echo 1 || echo 0)

    # Add timeout to prevent infinite hanging
    local timeout=60 # 60 seconds timeout
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        has_awsim=$(wmctrl -l | grep -q "AWSIM" && echo 1 || echo 0)
        has_rviz=$(wmctrl -l | grep -q "RViz" && echo 1 || echo 0)

        if [ "$has_rviz" -eq 1 ] && { [ "$has_awsim" -eq 1 ] || [ "$has_gpu" -eq 0 ]; }; then
            break
        fi
        sleep 1
        ((elapsed++))
        echo "Move window: $elapsed seconds elapsed"
    done

    if [ $elapsed -ge $timeout ]; then
        echo "WARNING: Timeout waiting for AWSIM/RViz windows after ${timeout} seconds"
        echo "AWSIM window found: $has_awsim"
        echo "RViz window found: $has_rviz"
        echo "GPU available: $has_gpu"
        echo "Continuing without window positioning..."
        return 1
    fi

    echo "AWSIM and RViz windows found"
    # Move windows
    wmctrl -a "RViz" && wmctrl -r "RViz" -e 0,0,0,1920,1043
    sleep 1
    wmctrl -a "AWSIM" && wmctrl -r "AWSIM" -e 0,0,0,900,1043
    sleep 2
}

# Move working directory
OUTPUT_DIRECTORY=$(date +%Y%m%d-%H%M%S)
cd /output || exit
mkdir "$OUTPUT_DIRECTORY"
ln -nfs "$OUTPUT_DIRECTORY" latest
cd "$OUTPUT_DIRECTORY" || exit

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source /autoware/install/setup.bash
# shellcheck disable=SC1091
source /aichallenge/workspace/install/setup.bash
sudo ip link set multicast on lo
sudo sysctl -w net.core.rmem_max=2147483647 >/dev/null

# Start AWSIM with nohup
echo "Start AWSIM"
nohup /aichallenge/run_simulator.bash >/dev/null &
PID_AWSIM=$!
echo "AWSIM PID: $PID_AWSIM"
sleep 3

# Start Autoware with nohup
echo "Start Autoware"
nohup /aichallenge/run_autoware.bash awsim >autoware.log 2>&1 &
sleep 3

move_window
bash /aichallenge/publish.bash check
move_window
bash /aichallenge/publish.bash all
# Capture screen
if [ "$IS_CAPTURE_MODE" -eq 1 ]; then
    bash /aichallenge/publish.bash screen
    echo "Screen capture started."
else
    echo "Screen capture skipped."
fi

# Start recording rosbag with nohup
if [ "$IS_ROSBAG_MODE" -eq 1 ]; then
    echo "Start rosbag"
    nohup /aichallenge/record_rosbag.bash >/dev/null 2>&1 &
    PID_ROSBAG=$!
    echo "ROS Bag PID: $PID_ROSBAG"
    # Wait a moment for rosbag to initialize and verify it's running
    sleep 2
    if ! kill -0 "$PID_ROSBAG" 2>/dev/null; then
        echo "Warning: Rosbag process is not running"
    else
        echo "Rosbag recording started successfully"
    fi
else
    # ROS Bagモードでない場合、PIDをクリアにしておく
    PID_ROSBAG=""
    echo "ROS Bag recording skipped."
fi

# Wait for AWSIM to finish (this is the main process we're waiting for)
wait "$PID_AWSIM"

# Stop recording rviz2
if [ "$IS_CAPTURE_MODE" -eq 1 ]; then
    echo "Stop screen capture"
    bash /aichallenge/publish.bash screen
    sleep 3
fi

# Convert result
echo "Convert result"
python3 /aichallenge/workspace/src/aichallenge_system/script/result-converter.py 60 11

if [ -n "$HOST_UID" ] && [ -n "$HOST_GID" ]; then

    # このスクリプトがroot (UID 0) で実行されているかチェック
    if [ "$(id -u)" -eq 0 ]; then
        echo "Running as root. Changing ownership of artifacts to ${HOST_UID}:${HOST_GID}..."

        # $OUTPUT_DIRECTORY (例: /output/20251107-173000) はカレントディレクトリになっている
        # /output/XXX を chown する
        echo "Target directory: $(pwd)"
        chown -R "${HOST_UID}:${HOST_GID}" "$(pwd)"

        # /output/latest リンク自体の所有者も変更 (-h オプション)
        # 1つ上の階層 (/output) にある "latest" リンクを変更する
        chown -h "${HOST_UID}:${HOST_GID}" /output/latest

        echo "Ownership change complete."
    else
        # root以外 (おそらく指定されたHOST_UID) で実行されている場合
        echo "Running as non-root user ($(id -u)). Files should already have correct ownership. Skipping chown."
    fi
else
    # 引数が設定されていなかった場合
    echo "HOST_UID/HOST_GID not provided as arguments. Skipping ownership change."
fi
echo "Evaluation Script finished."

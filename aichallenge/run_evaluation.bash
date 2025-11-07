#!/bin/bash

# Check if rosbag mode is enabled
IS_ROSBAG_MODE=0
for arg in "$@"; do
    if [ "$arg" = "rosbag" ]; then
        IS_ROSBAG_MODE=1
        break
    fi
done

if [ "$IS_ROSBAG_MODE" -eq 1 ]; then
    echo "ROS Bag recording mode enabled."
fi

# Check if capture mode is requested
IS_CAPTURE_MODE=0
for arg in "$@"; do
    if [ "$arg" = "capture" ]; then
        IS_CAPTURE_MODE=1
        break
    fi
done

if [ "$IS_CAPTURE_MODE" -eq 1 ]; then
    echo "Screen capture mode enabled."
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

echo "Evaluation Script finished."

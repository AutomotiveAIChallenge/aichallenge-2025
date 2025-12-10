#!/bin/bash

# 色付きメッセージ用の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}  IMU高度フィルタシステム セットアップ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${YELLOW}解決する課題:${NC}"
echo -e "  1. ${RED}ドリフト現象${NC} - IMU積分計算による誤差累積"
echo -e "  2. ${RED}振動ノイズ${NC} - 車両走行時の振動混入"
echo ""

# 引数チェック
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}使用方法:${NC}"
    echo -e "  $0 [preset] [debug_mode]"
    echo -e ""
    echo -e "${YELLOW}プリセット選択:${NC}"
    echo -e "  ${GREEN}balanced${NC}     - バランス型（推奨）"
    echo -e "  ${GREEN}precision${NC}    - 高精度要求環境"
    echo -e "  ${GREEN}speed${NC}        - 高速走行環境"
    echo -e "  ${GREEN}noisy${NC}        - ノイズ多環境"
    echo -e ""
    echo -e "${YELLOW}例:${NC}"
    echo -e "  $0 balanced false   # バランス型、デバッグOFF"
    echo -e "  $0 precision true   # 高精度型、デバッグON"
    exit 1
fi

PRESET=${1:-balanced}
DEBUG_MODE=${2:-false}

echo -e "${YELLOW}設定:${NC}"
echo -e "  プリセット: ${GREEN}$PRESET${NC}"
echo -e "  デバッグモード: ${GREEN}$DEBUG_MODE${NC}"
echo ""

# プリセット別パラメータ設定
case $PRESET in
    "precision")
        echo -e "${BLUE}高精度要求環境用設定を適用${NC}"
        DRIFT_GAIN=0.015
        CUTOFF_FREQ=8.0
        VIBRATION_THRESH=0.4
        ;;
    "speed")
        echo -e "${BLUE}高速走行環境用設定を適用${NC}"
        DRIFT_GAIN=0.035
        CUTOFF_FREQ=25.0
        VIBRATION_THRESH=1.2
        ;;
    "noisy")
        echo -e "${BLUE}ノイズ多環境用設定を適用${NC}"
        DRIFT_GAIN=0.02
        CUTOFF_FREQ=10.0
        VIBRATION_THRESH=0.3
        ;;
    *)
        echo -e "${BLUE}バランス型設定を適用${NC}"
        DRIFT_GAIN=0.025
        CUTOFF_FREQ=12.0
        VIBRATION_THRESH=0.6
        ;;
esac

echo ""
echo -e "${YELLOW}システム起動中...${NC}"

# ROS2起動コマンド実行
ros2 launch aichallenge_submit_launch aichallenge_imu_integration.launch.xml \
    drift_compensation_gain:=$DRIFT_GAIN \
    lowpass_cutoff_frequency:=$CUTOFF_FREQ \
    vibration_detection_threshold:=$VIBRATION_THRESH \
    debug_output:=$DEBUG_MODE

echo ""
echo -e "${GREEN}IMU高度フィルタシステム起動完了！${NC}"
echo -e "${BLUE}============================================${NC}"

import argparse
import multiprocessing
import os
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore
from rclpy.serialization import deserialize_message
from autoware_auto_control_msgs.msg import AckermannControlCommand
from sensor_msgs.msg import LaserScan
from concurrent.futures import ProcessPoolExecutor


CONTROL_TOPIC = '/awsim/control_cmd'
SCAN_TOPIC = '/scan'
MAX_RANGE = 30.0  


def clean_scan_array(scan_array: np.ndarray, max_range: float = MAX_RANGE) -> np.ndarray:
    """LiDARスキャン配列のinf/nanをクレンジング。"""
    if not isinstance(scan_array, np.ndarray):
        scan_array = np.array(scan_array, dtype=np.float32)

    cleaned = np.nan_to_num(scan_array, nan=0.0, posinf=max_range, neginf=0.0)
    cleaned = np.clip(cleaned, 0.0, max_range)
    return cleaned.astype(np.float32)


def extract_and_save_per_bag(bag_path, output_dir, cmd_topic, scan_topic, debug=False):
    pid = os.getpid()
    bag_path = Path(bag_path).expanduser().resolve()
    bag_name = bag_path.name
    out_dir = Path(output_dir) / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd_data, cmd_times = [], []
    scan_data, scan_times = [], []

    # Typestore initialization: library expects an object with .value (Enum in newer versions).
    # Current installed rosbags version does not expose TypeStoreId; emulate minimal wrapper.
    class _StoreName:
        def __init__(self, value: str):
            self.value = value

    typestore = None
    for candidate in ["ros2_humble", "ros2_foxy", "ros2_galactic", "ros2_iron"]:
        try:
            typestore = get_typestore(_StoreName(candidate))
            if debug:
                print(f"[PID:{pid} DEBUG] Typestore initialized using wrapper for '{candidate}'")
            break
        except Exception as ex:
            if debug:
                print(f"[PID:{pid} WARN] Failed typestore candidate '{candidate}': {ex}")
            continue
    if typestore is None:
        try:
            # fallback: call without arguments (older signature)
            typestore = get_typestore()  # type: ignore
            if debug:
                print(f"[PID:{pid} DEBUG] Fallback get_typestore() without args succeeded")
        except Exception as final_ex:
            print(f"[PID:{pid} ERROR] Unable to initialize typestore: {final_ex}")
            return

    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            # Build connections list; if explicit topics missing attempt auto-detect.
            requested_topics = [cmd_topic, scan_topic]
            connections = [c for c in reader.connections if c.topic in requested_topics]

            if debug:
                available = [(c.topic, c.msgtype) for c in reader.connections]
                print(f"[DEBUG PID:{pid}] Available topics count={len(available)}")
                for t, mt in available[:50]:  # limit spam
                    print(f"  - {t} :: {mt}")

            # Auto-detect control topic if not found
            if not any(c.topic == cmd_topic for c in connections):
                ctrl_candidates = [c for c in reader.connections if c.msgtype == 'autoware_auto_control_msgs/msg/AckermannControlCommand']
                if ctrl_candidates:
                    chosen = ctrl_candidates[0]
                    connections.append(chosen)
                    if debug:
                        print(f"[PID:{pid} WARN] Requested control topic '{cmd_topic}' missing. Using '{chosen.topic}' instead.")
                    cmd_topic = chosen.topic  # update for downstream logic

            # Auto-detect scan topic if not found
            if not any(c.topic == scan_topic for c in connections):
                scan_candidates = [c for c in reader.connections if c.msgtype == 'sensor_msgs/msg/LaserScan']
                if scan_candidates:
                    chosen = scan_candidates[0]
                    connections.append(chosen)
                    if debug:
                        print(f"[PID:{pid} WARN] Requested scan topic '{scan_topic}' missing. Using '{chosen.topic}' instead.")
                    scan_topic = chosen.topic

            if debug:
                print(f"[DEBUG PID:{pid}] Reading {bag_name} with topics: {[c.topic for c in connections]}")

            for conn, timestamp, raw in reader.messages(connections=connections):
                # primary deserialization; fallback to rclpy for selected message types
                try:
                    msg = reader.deserialize(raw, conn.msgtype)
                except Exception as des_ex:
                    if debug:
                        print(f"[PID:{pid} WARN] deserialize failed for {conn.msgtype}: {des_ex}; trying rclpy fallback")
                    try:
                        if conn.msgtype == 'autoware_auto_control_msgs/msg/AckermannControlCommand':
                            msg = deserialize_message(raw, AckermannControlCommand)
                        elif conn.msgtype == 'sensor_msgs/msg/LaserScan':
                            msg = deserialize_message(raw, LaserScan)
                        else:
                            continue  # unsupported type fallback; skip
                    except Exception as fb_ex:
                        if debug:
                            print(f"[PID:{pid} ERROR] Fallback failed for {conn.msgtype}: {fb_ex}")
                        continue

                # --- control ---
                if conn.topic == cmd_topic and conn.msgtype == 'autoware_auto_control_msgs/msg/AckermannControlCommand':
                    accel = msg.longitudinal.acceleration
                    steer = msg.lateral.steering_tire_angle
                    cmd_vec = np.array([steer, accel], dtype=np.float32)
                    cmd_data.append(cmd_vec)
                    cmd_times.append(timestamp)

                # --- scan ---
                elif conn.topic == scan_topic and conn.msgtype == 'sensor_msgs/msg/LaserScan':
                    scan_vec = clean_scan_array(np.array(msg.ranges, dtype=np.float32))
                    scan_data.append(scan_vec)
                    scan_times.append(timestamp)

    except Exception as e:
        print(f"[PID:{pid} ERROR] {bag_name}: Failed to read bag file. {e}")
        return

    if len(cmd_data) == 0 or len(scan_data) == 0:
        print(f'[PID:{pid} WARN] Skipping {bag_name}: insufficient data.')
        return

    cmd_data, cmd_times = np.array(cmd_data), np.array(cmd_times)
    scan_data, scan_times = np.array(scan_data), np.array(scan_times)

    # --- 範囲チェック ---
    if debug:
        if scan_times[0] < cmd_times[0]:
            diff = (cmd_times[0] - scan_times[0]) / 1e9
            print(f"[PID:{pid} WARN] {bag_name}: first scan before first cmd ({diff:.3f}s)")
        if scan_times[-1] > cmd_times[-1]:
            diff = (scan_times[-1] - cmd_times[-1]) / 1e9
            print(f"[PID:{pid} WARN] {bag_name}: last scan after last cmd ({diff:.3f}s)")

    # --- 同期処理 ---
    synced_scans, synced_steers, synced_accels, delta_times = [], [], [], []
    for i, base_time in enumerate(scan_times):
        idx_cmd = np.argmin(np.abs(cmd_times - base_time))
        delta_t = abs(cmd_times[idx_cmd] - base_time)
        delta_times.append(delta_t)

        steer, accel = cmd_data[idx_cmd]
        synced_steers.append(steer)
        synced_accels.append(accel)
        synced_scans.append(scan_data[i])

    delta_times = np.array(delta_times) / 1e9  # 秒単位に換算

    if debug:
        print(
            f"[PID:{pid} DEBUG] {bag_name}: Δt mean={delta_times.mean():.4f}s, "
            f"max={delta_times.max():.4f}s, min={delta_times.min():.4f}s"
        )

    # --- 保存 ---
    np.save(out_dir / 'scans.npy', np.array(synced_scans))
    np.save(out_dir / 'steers.npy', np.array(synced_steers))
    np.save(out_dir / 'accelerations.npy', np.array(synced_accels))

    if debug:
        np.save(out_dir / 'delta_times.npy', delta_times)
        print(f"[PID:{pid} DEBUG] Saved delta_times.npy ({len(delta_times)} samples)")

    print(f'[PID:{pid} SAVE] {bag_name}: {len(scan_times)} samples saved to {out_dir}')


def main():
    parser = argparse.ArgumentParser(description='Extract and synchronize scan and control data from rosbags.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bags_dir', help='Path to directory containing rosbag folders (searches recursively)')
    group.add_argument('--seq_dirs', nargs='+', help='List of specific sequence directories to process (non-recursive)')
    parser.add_argument('--outdir', required=True, help='Output root directory')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers. (Default: CPU count - 1, max 8)')
    parser.add_argument('--debug', action='store_true', help='Enable detailed synchronization debug output')
    parser.add_argument('--no-parallel', action='store_true', help='Run sequentially (diagnostic mode, ignores --workers)')
    parser.add_argument('--control-topic', default=CONTROL_TOPIC, help=f'Override control command topic (default {CONTROL_TOPIC})')
    parser.add_argument('--scan-topic', default=SCAN_TOPIC, help=f'Override LaserScan topic (default {SCAN_TOPIC})')
    args = parser.parse_args()

    # --- rosbag directory 検索 ---
    bag_dirs = []
    if args.bags_dir:
        bags_dir_path = Path(args.bags_dir).expanduser().resolve()
        for p in bags_dir_path.rglob("metadata.yaml"):
            if p.is_file():
                bag_dirs.append(p.parent)
        if not bag_dirs and (bags_dir_path / "metadata.yaml").exists():
            bag_dirs = [bags_dir_path]
    elif args.seq_dirs:
        for seq_path_str in args.seq_dirs:
            seq_path = Path(seq_path_str).expanduser().resolve()
            if (seq_path / "metadata.yaml").is_file():
                bag_dirs.append(seq_path)

    if not bag_dirs:
        print("[ERROR] No valid rosbag directories to process.")
        return

    tasks = [(bag_path, args.outdir, args.control_topic, args.scan_topic, args.debug) for bag_path in sorted(bag_dirs)]

    # --- 並列処理ワーカー数 ---
    if args.workers:
        num_workers = args.workers
    else:
        cpu_count = os.cpu_count()
        num_workers = min(max(1, (cpu_count or 4) - 1), 8)

    if args.no_parallel:
        print(f"[INFO] Running sequentially (diagnostic mode). Processing {len(tasks)} bag(s)...")
        for t in tasks:
            try:
                extract_and_save_per_bag(*t)
            except Exception as e:
                import traceback
                print(f"[ERROR] Sequential processing failed for {t[0]}: {e}")
                traceback.print_exc()
        print("[INFO] Sequential processing finished.")
        return

    print(f"[INFO] Starting parallel processing with {num_workers} workers...")

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(extract_and_save_per_bag, tasks)
        print("[INFO] All processing finished.")
    except Exception as e:
        import traceback
        print(f"[ERROR] Parallel processing failed: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            print(f"[WARN] Could not set start method 'spawn': {e}")
    main()

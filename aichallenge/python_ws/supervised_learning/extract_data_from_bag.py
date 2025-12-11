#!/usr/bin/env python3
import os
import cv2
import yaml
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from rosbags.highlevel import AnyReader


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_rosbag_dirs(root: Path) -> list[Path]:
    print(f"Searching for rosbags under {root} ...")
    bags = []
    for dp, _, files in os.walk(root):
        if "metadata.yaml" in files:
            bags.append(Path(dp))
            print(f"  -> Found: {Path(dp).relative_to(root)}")
    return bags


def to_numpy(msg):
    """ROS msg to plain Python dict"""
    d = {}
    for field in dir(msg):
        if field.startswith("_") or field in ["header", "stamp"]:
            continue
        val = getattr(msg, field)
        if hasattr(val, "__dict__"):
            val = val.__dict__
        d[field] = val
    return d


def extract_and_sync_bag(bag_path: Path, cfg: dict, output_root: Path):
    topic_map = cfg["topic_mapping"]
    active = [t for k, t in topic_map.items() if cfg["topics"].get(k, False)]
    ref_topic = cfg["sync"]["reference_topic"]
    tol = cfg["sync"]["tolerance_ms"] / 1000.0
    filter_human = cfg.get("filter_human_only", False)
    target_laps = cfg.get("target_laps", None)

    # 出力構造
    seq_root = output_root / bag_path.name
    seq_data = seq_root / "sequence_data"
    cam_dir = seq_root / "camera_front"
    seq_data.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Extracting & Synchronizing {bag_path.name} ===")

    # --- Read all topics ---
    topic_msgs, topic_ts = {}, {}
    with AnyReader([bag_path]) as reader:
        for conn in reader.connections:
            if conn.topic not in active:
                continue
            msgs, ts = [], []
            for _, _, raw in reader.messages(connections=[conn]):
                m = reader.deserialize(raw, conn.msgtype)
                if hasattr(m, "header"):
                    sec, nsec = m.header.stamp.sec, m.header.stamp.nanosec
                elif hasattr(m, "stamp"):
                    sec, nsec = m.stamp.sec, m.stamp.nanosec
                else:
                    continue
                ts.append(sec + nsec * 1e-9)
                msgs.append(m)
            topic_msgs[conn.topic] = msgs
            topic_ts[conn.topic] = np.array(ts, dtype=np.float64)

    if ref_topic not in topic_ts:
        print(f"⚠️ Reference topic '{ref_topic}' not found in bag.")
        return

    ref_ts = topic_ts[ref_topic]
    n_ref = len(ref_ts)
    print(f"  -> Reference '{ref_topic}' frames: {n_ref}")

    # --- Synchronization ---
    synced = {}
    for topic, ts in topic_ts.items():
        msgs = topic_msgs[topic]
        if topic == ref_topic:
            synced[topic] = msgs
            continue
        idx = np.searchsorted(ts, ref_ts)
        idx = np.clip(idx, 0, len(ts) - 1)
        diff = np.abs(ts[idx] - ref_ts)
        valid = diff < tol
        synced[topic] = [msgs[i] if v else None for i, v in zip(idx, valid)]

    print("  -> In-memory synchronization done.")

    # --- Control extraction ---
    control_msgs = synced[ref_topic]
    ctrl_dtype = np.dtype(
        [("sec", "i4"), ("nanosec", "u4"), ("lap_id", "f4"),
         ("accel", "f4"), ("steer", "f4"), ("mode_flag", "i1")]
    )
    ctrl_arr = np.zeros(n_ref, dtype=ctrl_dtype)
    for i, m in enumerate(control_msgs):
        if m is None:
            continue
        s, l, a = m.stamp, m.longitudinal, m.lateral
        ctrl_arr[i] = (
            s.sec, s.nanosec, l.speed, l.acceleration,
            a.steering_tire_angle,
            1 if abs(a.steering_tire_rotation_rate - 1.0) < 1e-3 else 0,
        )

    # --- Filter by lap & human ---
    mask = np.ones(len(ctrl_arr), dtype=bool)
    if target_laps:
        mask &= np.isin(ctrl_arr["lap_id"].astype(int), target_laps)
    if filter_human:
        mask &= ctrl_arr["mode_flag"] == 1
    valid_idx = np.nonzero(mask)[0]
    ref_ts = ref_ts[valid_idx]
    ctrl_arr = ctrl_arr[valid_idx]

    print(f"  -> Filtered {len(valid_idx)} frames from {n_ref}")

    # Apply mask to all
    for topic, msgs in synced.items():
        if topic == ref_topic:
            continue
        synced[topic] = [msgs[i] if i < len(msgs) else None for i in valid_idx]

    # --- Save numeric topics as .npy ---
    np.save(seq_data / "timestamps.npy", ref_ts)
    np.save(seq_data / "control_cmd.npy", ctrl_arr)

    def save_numpy(name, data_list):
        np.save(seq_data / f"{name}.npy", np.array(data_list, dtype=object))

    # IMU
    if "/sensing/imu/imu_raw" in synced:
        imu_data = []
        for m in synced["/sensing/imu/imu_raw"]:
            if m is None:
                imu_data.append({})
                continue
            imu_data.append({
                "angular_velocity": [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z],
                "linear_acceleration": [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z],
            })
        save_numpy("imu_raw", imu_data)

    # Velocity
    if "/vehicle/status/velocity_status" in synced:
        vel_data = []
        for m in synced["/vehicle/status/velocity_status"]:
            if m is None:
                vel_data.append({})
                continue
            vel_data.append({
                "vx": m.longitudinal_velocity,
                "vy": m.lateral_velocity,
                "wz": m.heading_rate,
            })
        save_numpy("velocity_status", vel_data)

    # Pose
    if "/localization/pose_with_covariance" in synced:
        pose_data = []
        for m in synced["/localization/pose_with_covariance"]:
            if m is None:
                pose_data.append({})
                continue
            pose_data.append({
                "position": [m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z],
                "orientation": [m.pose.pose.orientation.x, m.pose.pose.orientation.y,
                                m.pose.pose.orientation.z, m.pose.pose.orientation.w],
            })
        save_numpy("pose_with_covariance", pose_data)

    # GNSS
    if "/sensing/gnss/nav_sat_fix" in synced:
        gps_data = []
        for m in synced["/sensing/gnss/nav_sat_fix"]:
            if m is None:
                gps_data.append({})
                continue
            gps_data.append({
                "lat": m.latitude,
                "lon": m.longitude,
                "alt": m.altitude,
            })
        save_numpy("nav_sat_fix", gps_data)

    # Steering
    if "/vehicle/status/steering_status" in synced:
        steer_data = []
        for m in synced["/vehicle/status/steering_status"]:
            if m is None:
                steer_data.append({})
                continue

            d = {}
            for field in dir(m):
                if field.startswith("_") or field in ["__class__", "stamp"]:
                    continue
                val = getattr(m, field)
                # ネストした Time 型などを展開
                if hasattr(val, "__dict__"):
                    val = val.__dict__
                d[field] = val

            if hasattr(m, "stamp"):
                d["sec"] = m.stamp.sec
                d["nanosec"] = m.stamp.nanosec

            steer_data.append(d)

        save_numpy("steering_status", steer_data)

    # LiDAR
    if "/scan" in synced:
        scan_data = []
        for m in synced["/scan"]:
            if m is None:
                scan_data.append({})
                continue
            scan_data.append({
                "ranges": np.array(m.ranges, dtype=np.float32),
                "intensities": np.array(m.intensities, dtype=np.float32),
            })
        save_numpy("scan", scan_data)

    # --- Camera ---
    if "/sensing/camera/image_raw" in synced:
        from cv_bridge import CvBridge
        bridge = CvBridge()
        for i, m in enumerate(tqdm(synced["/sensing/camera/image_raw"], desc="Saving camera frames")):
            if m is None:
                continue
            img = bridge.imgmsg_to_cv2(m, desired_encoding="bgr8")
            cv2.imwrite(str(cam_dir / f"{i:06d}.png"), img)

    # --- Meta info ---
    meta = {
        "ref_topic": ref_topic,
        "tolerance": tol,
        "num_frames": len(ref_ts),
        "filtered_laps": target_laps,
        "human_only": filter_human,
        "topics_saved": list(synced.keys()),
    }
    with open(seq_data / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    print(f"✅ Saved sequence: {seq_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract + synchronize + filter ROS2 bag topics → PNG + NPY hybrid output."
    )
    parser.add_argument("--search_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/extract_data_from_bag.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_root = Path(args.output_dir)
    search_root = Path(args.search_dir)
    bags = find_rosbag_dirs(search_root)

    for b in bags:
        extract_and_sync_bag(b, cfg, output_root)

    print("\n✅ All tasks completed.")


if __name__ == "__main__":
    main()

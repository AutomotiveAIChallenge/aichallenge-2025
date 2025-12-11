#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
import numpy as np
import h5py
import hdf5plugin
import yaml
from rosbags.highlevel import AnyReader
from tqdm import tqdm


def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_blosc_opts(cfg):
    c = cfg.get("compression", {})
    complib = c.get("complib", "blosc:zstd")
    complevel = c.get("complevel", 1)
    shuffle = c.get("shuffle", "byte")
    shuffle_map = {"bit": 2, "byte": 1, "none": 0}
    complib_name = complib.split(":")[-1]
    return {
        **hdf5plugin.Blosc(clevel=complevel, cname=complib_name, shuffle=shuffle_map.get(shuffle, 1)),
        "chunks": True,
    }


def find_rosbag_dirs(root: Path) -> list[Path]:
    print(f"Searching for rosbags under {root} ...")
    bags = []
    for dp, _, files in os.walk(root):
        if "metadata.yaml" in files:
            bags.append(Path(dp))
            print(f"  -> Found: {Path(dp).relative_to(root)}")
    return bags


def extract_and_sync_bag(bag_path: Path, cfg: dict, output_root: Path):
    topic_map = cfg["topic_mapping"]
    active = [t for k, t in topic_map.items() if cfg["topics"].get(k, False)]
    ref_key = cfg["sync"]["reference_topic"]
    tol = cfg["sync"]["tolerance_ms"] / 1000.0
    out_h5 = output_root / f"{bag_path.name}_synced.h5"

    print(f"\n=== Extracting & Synchronizing {bag_path.name} ===")
    # --- Step 1: 読み込みと全トピック抽出 ---
    topic_msgs = {}
    topic_ts = {}

    with AnyReader([bag_path]) as reader:
        for conn in reader.connections:
            if conn.topic not in active:
                continue
            msgs = []
            ts = []
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

    if ref_key not in topic_ts:
        print(f"⚠️ Reference topic '{ref_key}' not found in bag.")
        return

    ref_ts = topic_ts[ref_key]
    n_ref = len(ref_ts)
    print(f"  -> Reference '{ref_key}' frames: {n_ref}")

    # --- Step 2: 各トピックを同期 ---
    synced = {}
    for topic, ts in topic_ts.items():
        msgs = topic_msgs[topic]
        if topic == ref_key:
            synced[topic] = msgs
            continue
        idx = np.searchsorted(ts, ref_ts)
        idx = np.clip(idx, 0, len(ts) - 1)
        diff = np.abs(ts[idx] - ref_ts)
        valid = diff < tol
        aligned = [msgs[i] if v else None for i, v in zip(idx, valid)]
        synced[topic] = aligned
    print("  -> In-memory synchronization done.")

    # --- Step 3: HDF5出力 ---
    blosc_opts = get_blosc_opts(cfg)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("timestamp", data=ref_ts)
        for topic, msgs in synced.items():
            name = [k for k, v in topic_map.items() if v == topic][0]  # dataset key名
            if len(msgs) == 0:
                continue

            if name == "control_cmd":
                dtype = np.dtype([
                    ("sec", "i4"), ("nanosec", "u4"),
                    ("speed", "f4"), ("accel", "f4"),
                    ("steer", "f4"), ("steer_rate", "f4")
                ])
                arr = np.zeros(n_ref, dtype=dtype)
                for i, m in enumerate(msgs):
                    if m is None:
                        continue
                    s, l, a = m.stamp, m.longitudinal, m.lateral
                    arr[i] = (s.sec, s.nanosec, l.speed, l.acceleration, a.steering_tire_angle, a.steering_tire_rotation_rate)
                f.create_dataset(name, data=arr, **blosc_opts)

            elif name == "velocity_status":
                dtype = np.dtype([
                    ("sec", "i4"), ("nanosec", "u4"),
                    ("vx", "f4"), ("vy", "f4"), ("wz", "f4")
                ])
                arr = np.zeros(n_ref, dtype=dtype)
                for i, m in enumerate(msgs):
                    if m is None:
                        continue
                    h = m.header
                    arr[i] = (h.stamp.sec, h.stamp.nanosec, m.longitudinal_velocity, m.lateral_velocity, m.heading_rate)
                f.create_dataset(name, data=arr, **blosc_opts)

            elif name == "steering_status":
                dtype = np.dtype([("sec", "i4"), ("nanosec", "u4"), ("angle", "f4")])
                arr = np.zeros(n_ref, dtype=dtype)
                for i, m in enumerate(msgs):
                    if m is None:
                        continue
                    s = m.stamp
                    arr[i] = (s.sec, s.nanosec, m.steering_tire_angle)
                f.create_dataset(name, data=arr, **blosc_opts)

            elif name == "nav_sat_fix":
                dtype = np.dtype([
                    ("sec", "i4"), ("nanosec", "u4"),
                    ("lat", "f8"), ("lon", "f8"), ("alt", "f8")
                ])
                arr = np.zeros(n_ref, dtype=dtype)
                for i, m in enumerate(msgs):
                    if m is None:
                        continue
                    h = m.header
                    arr[i] = (h.stamp.sec, h.stamp.nanosec, m.latitude, m.longitude, m.altitude)
                f.create_dataset(name, data=arr, **blosc_opts)

            elif name == "pose_with_covariance":
                dtype = np.dtype([
                    ("sec", "i4"), ("nanosec", "u4"),
                    ("pos", "f4", (3,)), ("ori", "f4", (4,))
                ])
                arr = np.zeros(n_ref, dtype=dtype)
                for i, m in enumerate(msgs):
                    if m is None:
                        continue
                    h = m.header
                    p, o = m.pose.pose.position, m.pose.pose.orientation
                    arr[i] = (h.stamp.sec, h.stamp.nanosec, (p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
                f.create_dataset(name, data=arr, **blosc_opts)

    meta = {
        "source_bag": str(bag_path),
        "reference_topic": ref_key,
        "tolerance_ms": cfg["sync"]["tolerance_ms"],
        "total_ref_frames": int(n_ref),
        "topics": list(topic_map.keys())
    }
    with open(out_h5.with_suffix(".meta.yaml"), "w") as mf:
        yaml.dump(meta, mf)
    print(f"✅ Created synced dataset: {out_h5}")


def main():
    parser = argparse.ArgumentParser(description="Extract + synchronize ROS2 bag topics → HDF5 (single pass).")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--search_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/extract_data_from_bag.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_root = Path(args.output_dir)
    search_root = Path(args.search_dir)
    bags = find_rosbag_dirs(search_root)

    for b in bags:
        extract_and_sync_bag(b, cfg, output_root)

    print("\nAll tasks completed ✅")


if __name__ == "__main__":
    main()

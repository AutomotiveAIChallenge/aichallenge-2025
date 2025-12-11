import os
import yaml
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from rosbags.highlevel import AnyReader

import h5py
import hdf5plugin  # for Blosc/Zstd compression filters


# ============================================================
# Config / IO
# ============================================================

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


def extract_ts(msg):
    if hasattr(msg, "header"):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if hasattr(msg, "stamp"):
        return msg.stamp.sec + msg.stamp.nanosec * 1e-9
    return None


# ============================================================
# Core
# ============================================================

def extract_and_write_h5(bag_path: Path, cfg: dict, out_root: Path):
    topic_map = cfg["topic_mapping"]
    enabled = [t for k, t in topic_map.items() if cfg["topics"].get(k, False)]
    ref_topic = cfg["sync"]["reference_topic"]
    tol_s = float(cfg["sync"]["tolerance_ms"]) / 1000.0

    filter_human_only = bool(cfg.get("filter_human_only", False))
    target_laps = cfg.get("target_laps", None)  # 例: [0,1,2,3] or None

    # 出力先: bag名/merged_sequence.h5
    seq_root = out_root / bag_path.name
    seq_root.mkdir(parents=True, exist_ok=True)
    h5_path = seq_root / "merged_sequence.h5"

    print(f"\n=== Extracting {bag_path.name} ===")

    # ---------- 読み込み ----------
    topic_msgs, topic_ts = {}, {}
    with AnyReader([bag_path]) as reader:
        for conn in reader.connections:
            if conn.topic not in enabled:
                continue
            msgs, ts = [], []
            for _, _, raw in reader.messages(connections=[conn]):
                m = reader.deserialize(raw, conn.msgtype)
                t = extract_ts(m)
                if t is not None:
                    msgs.append(m)
                    ts.append(t)
            topic_msgs[conn.topic] = msgs
            topic_ts[conn.topic] = np.asarray(ts, dtype=np.float64)

    if ref_topic not in topic_ts or len(topic_ts[ref_topic]) == 0:
        print(f"⚠️ Reference topic '{ref_topic}' not found or empty. Skip.")
        return

    ref_ts = topic_ts[ref_topic]
    T = len(ref_ts)
    print(f"  -> Reference: {ref_topic} ({T} frames)")

    # ---------- 同期 ----------
    synced = {}
    for topic, ts in topic_ts.items():
        msgs = topic_msgs[topic]
        if topic == ref_topic:
            synced[topic] = msgs
            continue
        if len(ts) == 0:
            synced[topic] = [None] * T
            continue
        idx = np.searchsorted(ts, ref_ts)
        idx = np.clip(idx, 0, len(ts) - 1)
        diff = np.abs(ts[idx] - ref_ts)
        synced[topic] = [msgs[idx[i]] if diff[i] < tol_s else None for i in range(T)]

    # ---------- 各トピック抽出（型を統一） ----------
    # LiDAR (T,1080)
    scan = None
    if topic_map.get("scan") in synced:
        scan_list = []
        for m in synced[topic_map["scan"]]:
            if m is None:
                scan_list.append(np.full(1080, np.nan, dtype=np.float32))
                continue
            arr = np.asarray(m.ranges, dtype=np.float32)
            # 長さが1080と限らない場合もあるのでパディング/トリム
            if arr.size < 1080:
                pad = np.full(1080 - arr.size, np.nan, dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            elif arr.size > 1080:
                arr = arr[:1080]
            arr[np.isposinf(arr)] = np.nan
            scan_list.append(arr)
        scan = np.stack(scan_list, axis=0)  # (T,1080)

    # Control (steer, accel) と mode_flag, optional lap_id
    control = np.full((T, 2), np.nan, dtype=np.float32)
    mode_flag = np.zeros(T, dtype=np.int8)
    lap_id = np.full(T, np.nan, dtype=np.float32)

    if ref_topic in synced:
        for i, m in enumerate(synced[ref_topic]):
            if m is None:
                continue
            try:
                l = m.longitudinal
                a = m.lateral
                steer = float(getattr(a, "steering_tire_angle", np.nan))
                accel = float(getattr(l, "acceleration", np.nan))
                control[i] = [steer, accel]
                # 人手判定
                mode_flag[i] = 1 if abs(getattr(a, "steering_tire_rotation_rate", 0.0) - 1.0) < 1e-3 else 0
                # lap_id は separate に取得（speedは絶対に速度として使わない）
                # もし l に lap_id があるなら採用。なければ、速度に誤って入っているケースもあるため注意深く取得。
                if hasattr(l, "lap_id"):
                    lap_id[i] = float(l.lap_id)
                elif hasattr(l, "speed"):
                    # 注意: 現状 speed に lap が入っているケースがあるため、
                    # "速度" としては絶対に使わず、lap候補として扱うに留める。
                    # target_laps が指定されたときのみフィルタ目的で利用する。
                    lap_id[i] = float(l.speed)
            except Exception:
                continue

    # Velocity (vx, vy, wz) from velocity_status
    velocity = None
    if topic_map.get("velocity_status") in synced:
        vel_list = []
        for m in synced[topic_map["velocity_status"]]:
            if m is None:
                vel_list.append([np.nan, np.nan, np.nan])
                continue
            vx = float(getattr(m, "longitudinal_velocity", np.nan))
            vy = float(getattr(m, "lateral_velocity", np.nan))
            wz = float(getattr(m, "heading_rate", np.nan))
            vel_list.append([vx, vy, wz])
        velocity = np.asarray(vel_list, dtype=np.float32)

    # Pose (x,y,z,qx,qy,qz,qw)
    pose = None
    if topic_map.get("pose_with_covariance") in synced:
        pose_list = []
        for m in synced[topic_map["pose_with_covariance"]]:
            if m is None:
                pose_list.append([np.nan] * 7)
                continue
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            pose_list.append([float(p.x), float(p.y), float(p.z),
                              float(q.x), float(q.y), float(q.z), float(q.w)])
        pose = np.asarray(pose_list, dtype=np.float32)

    # IMU (optional)
    imu = None
    if topic_map.get("imu_raw") in synced:
        imu_list = []
        for m in synced[topic_map["imu_raw"]]:
            if m is None:
                imu_list.append([np.nan]*6)
                continue
            av = m.angular_velocity
            la = m.linear_acceleration
            imu_list.append([float(av.x), float(av.y), float(av.z),
                             float(la.x), float(la.y), float(la.z)])
        imu = np.asarray(imu_list, dtype=np.float32)

    # GNSS (optional)
    gnss = None
    if topic_map.get("nav_sat_fix") in synced:
        gnss_list = []
        for m in synced[topic_map["nav_sat_fix"]]:
            if m is None:
                gnss_list.append([np.nan, np.nan, np.nan])
                continue
            gnss_list.append([float(getattr(m, "latitude", np.nan)),
                              float(getattr(m, "longitude", np.nan)),
                              float(getattr(m, "altitude", np.nan))])
        gnss = np.asarray(gnss_list, dtype=np.float32)

    timestamps = ref_ts.astype(np.float64)

    # ---------- フィルタ（人手 / lap） ----------
    # 人間のみ
    valid_mask = np.ones(T, dtype=bool)
    if filter_human_only:
        valid_mask &= (mode_flag == 1)

    # 指定lapのみ（lap_id が有効なときだけ適用）
    if target_laps is not None and len(target_laps) > 0 and np.isfinite(lap_id).any():
        target_laps = set(int(x) for x in target_laps)
        lap_ok = np.array([int(x) in target_laps if np.isfinite(x) else False for x in lap_id], dtype=bool)
        valid_mask &= lap_ok

    # マスク適用
    def apply_mask(x):
        if x is None:
            return None
        return x[valid_mask]

    scan = apply_mask(scan)
    control = apply_mask(control)
    mode_flag_m = apply_mask(mode_flag)
    lap_id_m = apply_mask(lap_id)
    velocity = apply_mask(velocity)
    pose = apply_mask(pose)
    imu = apply_mask(imu)
    gnss = apply_mask(gnss)
    timestamps = apply_mask(timestamps)

    Tm = int(np.sum(valid_mask))
    print(f"  -> Kept {Tm}/{T} frames after filters")

    if Tm < 2:
        print("⚠️ Not enough frames after filtering. Skip writing.")
        return

    # ---------- transitions 構築（BC/RL用の (t,t+1) インデックスと done） ----------
    # 有効: scan と control と velocity が両時刻で有限
    def is_finite_row(row):
        return np.all(np.isfinite(row))

    valid_t = np.ones(Tm-1, dtype=bool)
    if scan is not None:
        valid_t &= np.array([is_finite_row(scan[i]) and is_finite_row(scan[i+1]) for i in range(Tm-1)])
    if control is not None:
        valid_t &= np.array([is_finite_row(control[i]) and is_finite_row(control[i+1]) for i in range(Tm-1)])
    if velocity is not None:
        valid_t &= np.array([is_finite_row(velocity[i]) and is_finite_row(velocity[i+1]) for i in range(Tm-1)])

    idx = np.nonzero(valid_t)[0]
    # done判定: lap境界が取れるなら境界でTrue、なければ「次時刻が無効」でTrue
    done = np.zeros_like(idx, dtype=np.uint8)
    if lap_id_m is not None and np.isfinite(lap_id_m).any():
        for k, i in enumerate(idx):
            if int(lap_id_m[i]) != int(lap_id_m[i+1]):
                done[k] = 1
    # 最後の有効インデックスがエピソード末尾であるケースも True にしておく
    if idx.size > 0 and idx[-1] == (Tm - 2):
        done[-1] = 1

    # ---------- HDF5 書き出し ----------
    comp = cfg.get("compression", {})
    cname = "zstd"
    clevel = int(comp.get("complevel", 1))
    shuffle = (hdf5plugin.Blosc.SHUFFLE if comp.get("shuffle", "byte") == "byte"
               else hdf5plugin.Blosc.NOSHUFFLE)
    filter_args = hdf5plugin.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)

    with h5py.File(h5_path, "w") as f:
        f.attrs["bag_name"] = bag_path.name
        f.attrs["reference_topic"] = ref_topic
        f.attrs["tolerance_s"] = tol_s
        f.attrs["frames"] = Tm
        f.attrs["filter_human_only"] = int(filter_human_only)
        f.attrs["target_laps"] = ",".join(map(str, target_laps)) if target_laps else ""

        if scan is not None:
            f.create_dataset("scan", data=scan, dtype="f4", **filter_args)
        if control is not None:
            f.create_dataset("control", data=control, dtype="f4", **filter_args)  # [steer, accel]
        if mode_flag_m is not None:
            f.create_dataset("mode_flag", data=mode_flag_m.astype(np.int8), **filter_args)
        if lap_id_m is not None:
            f.create_dataset("lap_id", data=lap_id_m.astype(np.float32), **filter_args)
        if velocity is not None:
            f.create_dataset("velocity", data=velocity, dtype="f4", **filter_args)  # [vx, vy, wz]
        if pose is not None:
            f.create_dataset("pose", data=pose, dtype="f4", **filter_args)          # [x,y,z,qx,qy,qz,qw]
        if imu is not None:
            f.create_dataset("imu", data=imu, dtype="f4", **filter_args)            # [wx,wy,wz,ax,ay,az]
        if gnss is not None:
            f.create_dataset("gnss", data=gnss, dtype="f4", **filter_args)
        f.create_dataset("timestamps", data=timestamps.astype(np.float64), **filter_args)

        # transitions
        grp = f.create_group("transitions")
        grp.create_dataset("index", data=idx.astype(np.int64), **filter_args)  # t の配列（t+1 は暗黙）
        grp.create_dataset("done", data=done.astype(np.uint8), **filter_args)

    print(f"✅ Wrote HDF5: {h5_path}")


# ============================================================
# Entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract+sync ROS2 bags → single HDF5 ready for BC/Offline-RL (no camera)."
    )
    parser.add_argument("--search_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/extract_data_from_bag.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_root = Path(args.output_dir)
    search_root = Path(args.search_dir)
    bags = find_rosbag_dirs(search_root)

    for b in bags:
        extract_and_write_h5(b, cfg, out_root)

    print("\n✅ All tasks completed.")


if __name__ == "__main__":
    main()

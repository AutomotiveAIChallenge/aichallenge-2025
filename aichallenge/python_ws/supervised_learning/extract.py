import os
from pathlib import Path
import argparse
import h5py
import numpy as np
import hdf5plugin
from rosbags.highlevel import AnyReader
from tqdm import tqdm
import yaml

# --- 対応するトピックの定数定義 ---
CONTROL_TOPIC = '/awsim/control_cmd'
SCAN_TOPIC = '/scan'  

# --- 処理対象のトピックリスト ---
TARGET_TOPICS = [
    CONTROL_TOPIC,
    SCAN_TOPIC  
]

def load_config(config_path):
    """YAML設定ファイルを読み込むヘルパー関数"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_blosc_opts(complevel=1, complib='blosc:zstd', shuffle='byte'):
    """Blosc圧縮の設定を返すヘルパー関数"""
    shuffle_map = {'bit': 2, 'byte': 1, 'none': 0}
    shuffle_val = shuffle_map.get(shuffle, 0)
    complib_name = complib.split(':')[-1]

    return {
        **hdf5plugin.Blosc(clevel=complevel, cname=complib_name, shuffle=shuffle_val),
        'chunks': True
    }

def find_rosbag_directories(root_search_path: Path) -> list[Path]:
    """指定されたパス以下を再帰的に探索し、rosbagのディレクトリのリストを返す関数"""
    bag_directories = []
    print(f"Searching for rosbags in '{root_search_path}'...")
    for dirpath, _, filenames in os.walk(root_search_path):
        if 'metadata.yaml' in filenames:
            bag_path = Path(dirpath)
            bag_directories.append(bag_path)
            print(f"  -> Found: {bag_path}")
    return bag_directories

def process_bag(bag_path: Path, output_h5_path: Path, config: dict):
    """単一のrosbagファイルを処理し、HDF5ファイルに変換する。"""
    print(f"\nProcessing '{bag_path.name}'...")
    
    # 1. メッセージ数の事前カウント
    topic_msg_counts = {}
    total_messages = 0
    duration_sec = 0.0
    try:
        with AnyReader([bag_path]) as reader:
            duration_ns = reader.end_time - reader.start_time
            duration_sec = duration_ns / 1_000_000_000.0
            
            connections = [c for c in reader.connections if c.topic in TARGET_TOPICS]
            for conn in connections:
                topic_msg_counts[conn.topic] = conn.msgcount
                total_messages += conn.msgcount
    except Exception as e:
        print(f"  -> Error reading bag file: {e}. Skipping.")
        return

    if not total_messages:
        print("  -> No target topics found in this bag. Skipping.")
        return

    print(f"  -> Bag duration: {duration_sec:.2f} seconds")
    print("  -> Message counts:")
    for topic, count in topic_msg_counts.items():
        estimated_hz = count / duration_sec if duration_sec > 0 else 0
        print(f"     - {topic}: {count} ({estimated_hz:.2f} Hz)")

    # 2. HDF5ファイルの作成とデータセットの初期化
    with h5py.File(output_h5_path, 'w') as f:
        blosc_opts = get_blosc_opts()
        datasets = {}

        if CONTROL_TOPIC in topic_msg_counts:
            dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'), ('speed', 'f4'), ('acceleration', 'f4'), ('steering_tire_angle', 'f4'), ('steering_tire_rotation_rate', 'f4')])
            datasets[CONTROL_TOPIC] = f.create_dataset('control_cmd', (topic_msg_counts[CONTROL_TOPIC],), dtype=dtype, **blosc_opts)

        if SCAN_TOPIC in topic_msg_counts:
            vlen_float_dtype = h5py.vlen_dtype(np.float32)
            dtype = np.dtype([
                ('sec', 'i4'), ('nanosec', 'u4'),
                ('ranges', vlen_float_dtype),
                ('intensities', vlen_float_dtype)
            ])
            datasets[SCAN_TOPIC] = f.create_dataset('scan', (topic_msg_counts[SCAN_TOPIC],), dtype=dtype, **blosc_opts)
            
            # LaserScanのメタデータをHDF5の属性として保存
            with AnyReader([bag_path]) as reader:
                scan_conn = next((c for c in reader.connections if c.topic == SCAN_TOPIC), None)
                if scan_conn:
                    _, _, rawdata = next(reader.messages(connections=[scan_conn]))
                    msg = reader.deserialize(rawdata, scan_conn.msgtype)
                    
                    scan_ds = datasets[SCAN_TOPIC]
                    scan_ds.attrs['angle_min'] = msg.angle_min
                    scan_ds.attrs['angle_max'] = msg.angle_max
                    scan_ds.attrs['angle_increment'] = msg.angle_increment
                    scan_ds.attrs['time_increment'] = msg.time_increment
                    scan_ds.attrs['scan_time'] = msg.scan_time
                    scan_ds.attrs['range_min'] = msg.range_min
                    scan_ds.attrs['range_max'] = msg.range_max
                    print("  -> Saved LaserScan metadata as HDF5 attributes.")

        BUFFER_SIZE = config.get('buffer_size', 1000) # configにない場合のデフォルト値
        buffers = {topic: [] for topic in topic_msg_counts.keys()}
        topic_indices = {topic: 0 for topic in topic_msg_counts.keys()}

        def flush_buffer(topic):
            if topic in buffers:
                buffer_list = buffers[topic]
                if not buffer_list: return
                start_idx, end_idx = topic_indices[topic], topic_indices[topic] + len(buffer_list)
                datasets[topic][start_idx:end_idx] = np.array(buffer_list, dtype=datasets[topic].dtype)
                buffer_list.clear()
                topic_indices[topic] = end_idx

        with AnyReader([bag_path]) as reader:
            connections_to_read = [c for c in reader.connections if c.topic in TARGET_TOPICS]
            with tqdm(total=total_messages, desc="  -> Writing data") as pbar:
                for connection, _, rawdata in reader.messages(connections=connections_to_read):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    topic = connection.topic

                    if topic == CONTROL_TOPIC:
                        s, l, a = msg.stamp, msg.longitudinal, msg.lateral
                        buffers[topic].append((s.sec, s.nanosec, l.speed, l.acceleration, a.steering_tire_angle, a.steering_tire_rotation_rate))
                    
                    elif topic == SCAN_TOPIC:
                        h = msg.header
                        buffers[topic].append((
                            h.stamp.sec, 
                            h.stamp.nanosec, 
                            np.array(msg.ranges, dtype=np.float32), 
                            np.array(msg.intensities, dtype=np.float32)
                        ))
                    
                    if len(buffers[topic]) >= BUFFER_SIZE:
                        flush_buffer(topic)
                    pbar.update(1)

        print("\n  -> Flushing remaining buffers...")
        for topic in buffers.keys():
            if topic in topic_msg_counts:
                flush_buffer(topic)

    print(f"  -> Successfully created '{output_h5_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert specific topics from ROS2 bags to HDF5 files.')
    parser.add_argument('search_dir', type=str, help='Root directory to search for rosbags.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output HDF5 files.')
    parser.add_argument('--config', type=str, default='config/extract.yaml', help='Path to the config file (for buffer_size).')
    args = parser.parse_args()

    search_path = Path(args.search_dir)
    output_path = Path(args.output_dir)
    config_path = Path(args.config)

    config = {}
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"Warning: Config file not found at {config_path}. Using default buffer size.")

    output_path.mkdir(parents=True, exist_ok=True)
    bag_directories = find_rosbag_directories(search_path)

    if not bag_directories:
        print("No rosbag directories found.")
    else:
        for bag_dir in bag_directories:
            relative_path = bag_dir.relative_to(search_path)
            output_h5_path = output_path / relative_path.with_suffix('.h5')
            output_h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            process_bag(bag_dir, output_h5_path, config)
        print("\nAll tasks completed.")
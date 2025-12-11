import os
from pathlib import Path
import argparse
import h5py
import numpy as np
import hdf5plugin
from rosbags.highlevel import AnyReader
from tqdm import tqdm
import yaml

def load_config(config_path):
    """YAML設定ファイルを読み込むヘルパー関数"""
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}.")
        return None
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
            print(f"  -> Found: {bag_path.relative_to(root_search_path)}")
    return bag_directories

def pre_check_topics(bag_directories: list[Path], active_topics_ros: list[str], base_path: Path) -> bool:
    """
    指定された全bagファイルに、アクティブなトピックがすべて含まれているか事前検証する。
    base_path: ログ出力時の相対パスの基準点
    """
    print(f"\nVerifying topics in {len(bag_directories)} bag(s)...")
    
    required_topics_set = set(active_topics_ros)
    all_bags_ok = True

    with tqdm(total=len(bag_directories), desc="  -> Checking bags") as pbar:
        for bag_dir in bag_directories:
            available_topics_in_bag = set()
            try:
                with AnyReader([bag_dir]) as reader:
                    available_topics_in_bag = {conn.topic for conn in reader.connections}
            except Exception as e:
                pbar.write(f"\n--- 警告: {bag_dir.relative_to(base_path)} ---")
                pbar.write(f"  -> bagの読み込み中にエラーが発生しました: {e}")
                all_bags_ok = False
                pbar.update(1)
                continue

            missing_topics = required_topics_set - available_topics_in_bag
            
            if missing_topics:
                all_bags_ok = False
                pbar.write(f"\n--- 警告: {bag_dir.relative_to(base_path)} ---")
                for topic in missing_topics:
                    pbar.write(f"  -> 設定で要求されたトピック '{topic}' が見つかりませんでした。")
            
            pbar.update(1)

    return all_bags_ok

def process_bag(bag_path: Path, output_h5_path: Path, config: dict, active_topics_ros: list[str], dataset_map: dict, base_path: Path):
    """
    単一のrosbagファイルを処理し、HDF5ファイルに変換する。
    base_path: ログ出力時の相対パスの基準点
    """
    print(f"\nProcessing '{bag_path.relative_to(base_path)}'...") 
    
    # 1. メッセージ数の事前カウント
    topic_msg_counts = {}
    total_messages = 0
    duration_sec = 0.0
    try:
        with AnyReader([bag_path]) as reader:
            duration_ns = reader.end_time - reader.start_time
            duration_sec = duration_ns / 1_000_000_000.0
            
            connections = [c for c in reader.connections if c.topic in active_topics_ros]
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
        print(f"        - {dataset_map[topic]} ({topic}): {count} ({estimated_hz:.2f} Hz)")

    # 2. HDF5ファイルの作成とデータセットの初期化
    with h5py.File(output_h5_path, 'w') as f:
        blosc_opts = get_blosc_opts()
        datasets = {} # topic_name -> HDF5 Dataset Object

        vlen_f32 = h5py.vlen_dtype(np.float32)
        vlen_f64 = h5py.vlen_dtype(np.float64)
        vlen_u8 = h5py.vlen_dtype(np.uint8)
        vlen_str = h5py.vlen_dtype(str)

        for topic_name, count in topic_msg_counts.items():
            dataset_name = dataset_map[topic_name]
            dtype = None

            if dataset_name == 'control_cmd':
                dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'), ('speed', 'f4'), ('acceleration', 'f4'), ('steering_tire_angle', 'f4'), ('steering_tire_rotation_rate', 'f4')])
            
            elif dataset_name == 'scan':
                dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'), ('ranges', vlen_f32), ('intensities', vlen_f32)])

            elif dataset_name == 'image_raw':
                dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'), ('height', 'u4'), ('width', 'u4'), ('encoding', vlen_str), ('step', 'u4'), ('data', vlen_u8)])

            elif dataset_name == 'camera_info':
                dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'), ('height', 'u4'), ('width', 'u4'),
                                  ('K', 'f4', (9,)), ('D', vlen_f64), ('R', 'f4', (9,)), ('P', 'f4', (12,))])
            
            elif dataset_name == 'imu_raw':
                dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'),
                                  ('orientation', 'f4', (4,)), ('orientation_covariance', 'f4', (9,)),
                                  ('angular_velocity', 'f4', (3,)), ('angular_velocity_covariance', 'f4', (9,)),
                                  ('linear_acceleration', 'f4', (3,)), ('linear_acceleration_covariance', 'f4', (9,))])

            elif dataset_name == 'nav_sat_fix':
                dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'), ('latitude', 'f8'), ('longitude', 'f8'), ('altitude', 'f8'), ('position_covariance', 'f4', (9,))])

            elif dataset_name == 'steering_status':
                dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'), ('steering_tire_angle', 'f4')])

            elif dataset_name == 'velocity_status':
                dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'), ('longitudinal_velocity', 'f4'), ('lateral_velocity', 'f4'), ('heading_rate', 'f4')])
            
            elif dataset_name == 'pose_with_covariance':
                dtype = np.dtype([('sec', 'i4'), ('nanosec', 'u4'),
                                  ('position', 'f4', (3,)), ('orientation', 'f4', (4,)), ('covariance', 'f4', (36,))])

            if dtype:
                datasets[topic_name] = f.create_dataset(dataset_name, (count,), dtype=dtype, **blosc_opts)
            else:
                print(f"Warning: No dtype defined for HDF5 key '{dataset_name}' (Topic: {topic_name}). It will be skipped.")

        # LaserScanのメタデータをHDF5の属性として保存
        scan_h5_key = 'scan'
        scan_ros_topic = config.get('topic_mapping', {}).get(scan_h5_key)
        if scan_ros_topic in datasets:
            try:
                with AnyReader([bag_path]) as reader:
                    scan_conn = next((c for c in reader.connections if c.topic == scan_ros_topic), None)
                    if scan_conn:
                        _, _, rawdata = next(reader.messages(connections=[scan_conn]))
                        msg = reader.deserialize(rawdata, scan_conn.msgtype)
                        
                        scan_ds = datasets[scan_ros_topic]
                        scan_ds.attrs['angle_min'] = msg.angle_min
                        scan_ds.attrs['angle_max'] = msg.angle_max
                        scan_ds.attrs['angle_increment'] = msg.angle_increment
                        scan_ds.attrs['time_increment'] = msg.time_increment
                        scan_ds.attrs['scan_time'] = msg.scan_time
                        scan_ds.attrs['range_min'] = msg.range_min
                        scan_ds.attrs['range_max'] = msg.range_max
                        print(f"  -> Saved LaserScan metadata as HDF5 attributes to '{scan_h5_key}'.")
            except Exception as e:
                print(f"  -> Warning: Could not save LaserScan metadata. {e}")


        # 3. データ抽出とバッファリング書き込み
        BUFFER_SIZE = config.get('buffer_size', 1000)
        buffers = {topic: [] for topic in topic_msg_counts.keys()}
        topic_indices = {topic: 0 for topic in topic_msg_counts.keys()}

        def flush_buffer(topic):
            if topic in buffers:
                buffer_list = buffers[topic]
                if not buffer_list: return
                start_idx, end_idx = topic_indices[topic], topic_indices[topic] + len(buffer_list)
                try:
                    datasets[topic][start_idx:end_idx] = np.array(buffer_list, dtype=datasets[topic].dtype)
                except ValueError as e:
                    print(f"\nError flushing buffer for topic {topic}: {e}")
                    if buffer_list:
                         print(f"Sample data (first item): {buffer_list[0]}")
                buffer_list.clear()
                topic_indices[topic] = end_idx

        with AnyReader([bag_path]) as reader:
            connections_to_read = [c for c in reader.connections if c.topic in active_topics_ros]
            with tqdm(total=total_messages, desc="  -> Writing data") as pbar:
                for connection, _, rawdata in reader.messages(connections=connections_to_read):
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    topic = connection.topic
                    dataset_name = dataset_map[topic] # HDF5キー

                    try:
                        if dataset_name == 'control_cmd':
                            s, l, a = msg.stamp, msg.longitudinal, msg.lateral
                            buffers[topic].append((s.sec, s.nanosec, l.speed, l.acceleration, a.steering_tire_angle, a.steering_tire_rotation_rate))
                        
                        elif dataset_name == 'scan':
                            h = msg.header
                            buffers[topic].append((h.stamp.sec, h.stamp.nanosec, 
                                                 np.array(msg.ranges, dtype=np.float32), 
                                                 np.array(msg.intensities, dtype=np.float32)))
                        
                        elif dataset_name == 'image_raw':
                            h = msg.header
                            buffers[topic].append((h.stamp.sec, h.stamp.nanosec, 
                                                 msg.height, msg.width, msg.encoding, msg.step, 
                                                 msg.data)) # msg.data は bytes
                        
                        elif dataset_name == 'camera_info':
                            h = msg.header
                            buffers[topic].append((h.stamp.sec, h.stamp.nanosec, msg.height, msg.width,
                                                 np.array(msg.k, dtype=np.float32), 
                                                 np.array(msg.d, dtype=np.float64), 
                                                 np.array(msg.r, dtype=np.float32), 
                                                 np.array(msg.p, dtype=np.float32)))

                        elif dataset_name == 'imu_raw':
                            h = msg.header
                            o, a, l = msg.orientation, msg.angular_velocity, msg.linear_acceleration
                            buffers[topic].append((h.stamp.sec, h.stamp.nanosec,
                                                 (o.x, o.y, o.z, o.w), msg.orientation_covariance,
                                                 (a.x, a.y, a.z), msg.angular_velocity_covariance,
                                                 (l.x, l.y, l.z), msg.linear_acceleration_covariance))
                        
                        elif dataset_name == 'nav_sat_fix':
                            h = msg.header
                            buffers[topic].append((h.stamp.sec, h.stamp.nanosec, 
                                                 msg.latitude, msg.longitude, msg.altitude, 
                                                 msg.position_covariance))

                        elif dataset_name == 'steering_status':
                            s = msg.stamp
                            buffers[topic].append((s.sec, s.nanosec, msg.steering_tire_angle))

                        elif dataset_name == 'velocity_status':
                            s = msg.stamp
                            buffers[topic].append((s.sec, s.nanosec, 
                                                 msg.longitudinal_velocity, msg.lateral_velocity, msg.heading_rate))
                        
                        elif dataset_name == 'pose_with_covariance':
                            h = msg.header
                            p, o = msg.pose.pose.position, msg.pose.pose.orientation
                            buffers[topic].append((h.stamp.sec, h.stamp.nanosec,
                                                 (p.x, p.y, p.z), (o.x, o.y, o.z, o.w), 
                                                 msg.pose.covariance))
                        
                        if len(buffers[topic]) >= BUFFER_SIZE:
                            flush_buffer(topic)
                    
                    except Exception as e:
                        print(f"\nError processing message for topic {topic} (HDF5 key '{dataset_name}'): {e}")
                        
                    pbar.update(1)

        print("\n  -> Flushing remaining buffers...")
        for topic in buffers.keys():
            if topic in topic_msg_counts:
                flush_buffer(topic)
        
        # 4. メタデータ生成
        print("  -> Generating metadata file...")
        metadata = {
            'source_bag_name': bag_path.name,
            'bag_duration_sec': round(duration_sec, 4),
            'total_messages_processed': total_messages,
            'datasets': {}
        }
        
        ros_topic_lookup = {v: k for k, v in dataset_map.items()} 

        for h5_key in f.keys():
            ds = f[h5_key]
            ds_info = {
                'ros_topic': ros_topic_lookup.get(h5_key, 'unknown'),
                'message_count': ds.shape[0],
                'estimated_hz': round(ds.shape[0] / duration_sec, 2) if duration_sec > 0 else 0,
                'dtype': str(ds.dtype),
                'attributes': dict(ds.attrs)
            }
            metadata['datasets'][h5_key] = ds_info
            
        metadata_output_path = output_h5_path.with_suffix('.meta.yaml')
        try:
            with open(metadata_output_path, 'w') as mf:
                yaml.dump(metadata, mf, default_flow_style=False, sort_keys=False)
            print(f"  -> Successfully created metadata at '{metadata_output_path}'")
        except Exception as e:
            print(f"  -> Error writing metadata file: {e}")

    print(f"  -> Successfully created '{output_h5_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert specific topics from ROS2 bags to HDF5 files.')
    
    parser.add_argument('output_dir', type=str, help='Directory to save the output HDF5 files.')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--search_dir', type=str, 
                             help='Root directory to search for rosbags recursively.')
    input_group.add_argument('--bag_list', type=str, 
                             help='A text file containing a list of paths to rosbag directories (one path per line).')
    
    parser.add_argument('--config', type=str, default='config/extract_data_from_bag.yaml', help='Path to the config file.')
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    config_path = Path(args.config)

    config = load_config(config_path)
    if config is None:
        exit(1)

    topic_mapping_config = config.get('topic_mapping', {})
    active_topics_config = config.get('topics', {})

    if not topic_mapping_config:
        print("Error: 'topic_mapping' section not found in config. Exiting.")
        exit(1)
    if not active_topics_config:
        print("Error: 'topics' section not found in config. Exiting.")
        exit(1)

    active_topics_ros = []
    dataset_map = {} # ROSトピック名 -> HDF5キー(configキー)

    for key, is_active in active_topics_config.items():
        if is_active:
            if key in topic_mapping_config:
                ros_topic_name = topic_mapping_config[key]
                active_topics_ros.append(ros_topic_name)
                dataset_map[ros_topic_name] = key
            else:
                print(f"Warning: Active topic key '{key}' not found in 'topic_mapping' section. Skipping.")
    
    if not active_topics_ros:
        print("Error: No active and mapped topics found. Exiting.")
        exit(1)
        
    print("--- Active Topics (ROS Name -> HDF5 Key) ---")
    for ros_topic, h5_key in dataset_map.items():
        print(f"- {ros_topic} -> {h5_key}")
    print("------------------------------------------")

    output_path.mkdir(parents=True, exist_ok=True)
    
    bag_directories = []
    base_path = None # 相対パス計算の基準点

    if args.search_dir:
        print("--- Input Mode: Recursive Search ---")
        base_path = Path(args.search_dir).resolve()
        bag_directories = find_rosbag_directories(base_path)
    
    elif args.bag_list:
        print("--- Input Mode: Bag List File ---")
        bag_list_path = Path(args.bag_list).resolve()
        if not bag_list_path.exists():
            print(f"Error: Bag list file not found at {bag_list_path}")
            exit(1)
        
        print(f"Reading bag list from: {bag_list_path}")
        with open(bag_list_path, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    bag_path = Path(stripped_line)
                    if not bag_path.is_absolute():
                         bag_path = bag_list_path.parent / bag_path
                    
                    bag_path = bag_path.resolve()
                    
                    if bag_path.exists() and (bag_path / 'metadata.yaml').exists():
                        bag_directories.append(bag_path)
                        print(f"  -> Added: {bag_path}")
                    else:
                        print(f"Warning: Path '{bag_path}' is not a valid rosbag directory. Skipping.")
        
        if not bag_directories:
            print("No valid rosbag directories found in the list file.")
            exit(0)
            
        str_paths = [str(p) for p in bag_directories]
        common_prefix = os.path.commonpath(str_paths)
        base_path = Path(common_prefix)
        print(f"Using common base path for output structure: {base_path}")

    if not bag_directories:
        print("No rosbag directories found to process.")
        exit(0)

    # 1. 事前検証の実行
    print("\n--- Topic Pre-check Phase ---")
    all_topics_valid = pre_check_topics(bag_directories, active_topics_ros, base_path) 

    # 2. 検証結果の確認
    if not all_topics_valid:
        print("\n--- 処理中断 ---")
        print("エラー: 設定ファイル (config.yaml) で 'true' に指定されたトピックが、")
        print("処理対象のrosbag内に存在しませんでした。上記の警告ログを確認してください。")
        print("プログラムを終了します。")
        exit(1)

    print("\n--- Topic Pre-check OK ---")
    print("すべてのbagで要求されたトピックが確認できました。")

    # 3. メイン処理
    print("\n--- HDF5 Conversion Phase ---")
    for bag_dir in bag_directories:
        relative_path = bag_dir.relative_to(base_path)
        output_h5_path = output_path / relative_path.with_suffix('.h5')
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        
        process_bag(bag_dir, output_h5_path, config, active_topics_ros, dataset_map, base_path)
            
    print("\nAll tasks completed.")
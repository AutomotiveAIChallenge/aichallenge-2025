import argparse
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# --- 1. 設定とデータの読み込み ---

def load_config(config_path: Path) -> Dict:
    """YAML設定ファイルを読み込む。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data_from_h5(h5_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """HDF5ファイルからscanとcontrolのデータをDataFrameとして読み込む。"""
    with h5py.File(h5_path, 'r') as f:
        if 'scan' not in f or 'control_cmd' not in f:
            raise FileNotFoundError("Required datasets ('scan', 'control_cmd') not found.")
            
        scan_df = pd.DataFrame(f['scan'][:])
        control_df = pd.DataFrame(f['control_cmd'][:])
        scan_attributes = dict(f['scan'].attrs.items())
        
        # タイムスタンプをナノ秒単位の単一カラムとして生成
        scan_df['timestamp'] = scan_df['sec'].astype(np.int64) * 1e9 + scan_df['nanosec']
        control_df['timestamp'] = control_df['sec'].astype(np.int64) * 1e9 + control_df['nanosec']
        
    return scan_df, control_df, scan_attributes

# --- 2. データの前処理とクリーニング ---

def clean_scan_data(scan_df: pd.DataFrame, attributes: Dict) -> pd.DataFrame:
    """scanデータ内のinf/nan値をクレンジングする。"""
    max_range = float(attributes.get('range_max', 30.0))
    
    def clean_array(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(arr, nan=0.0, posinf=max_range, neginf=0.0).astype(np.float32)

    # inf/nanの数を報告
    total_inf_nan = scan_df['ranges'].apply(lambda r: np.isinf(r).sum() + np.isnan(r).sum()).sum()
    if total_inf_nan > 0:
        print(f"  -> 🧼 Cleaning {total_inf_nan} inf/nan values from 'ranges' (using max_range: {max_range}).")
        
    scan_df['ranges'] = scan_df['ranges'].apply(clean_array)
    if 'intensities' in scan_df.columns:
        scan_df['intensities'] = scan_df['intensities'].apply(clean_array)
        
    return scan_df

def synchronize_dataframes(scan_df: pd.DataFrame, control_df: pd.DataFrame, tolerance_sec: float) -> pd.DataFrame:
    """タイムスタンプを基準にscanとcontrolのDataFrameを同期する。"""
    tolerance_ns = int(tolerance_sec * 1e9)
    
    merged_df = pd.merge_asof(
        scan_df.sort_values('timestamp'),
        control_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=tolerance_ns
    )
    merged_df.dropna(inplace=True)
    return merged_df.reset_index(drop=True)

# --- 3. データのフィルタリング ---

def filter_dataframe(df: pd.DataFrame, filter_config: Dict) -> pd.DataFrame:
    """設定に基づいてDataFrameをフィルタリングする。"""
    original_count = len(df)
    
    # フィルタリング1: 人間介入
    intervention_cfg = filter_config.get('human_intervention', {})
    if intervention_cfg.get('enabled', False):
        col, val = intervention_cfg['column'], intervention_cfg['flag_value']
        if col in df.columns:
            df = df[np.isclose(df[col], val)].copy()
            print(f"  -> Filtered by intervention '{col}=={val}': {original_count} -> {len(df)} rows")
            original_count = len(df)

    # フィルタリング2: ラップ数
    laps_cfg = filter_config.get('target_laps', {})
    if laps_cfg.get('enabled', False):
        col, laps = laps_cfg['column'], laps_cfg.get('laps')
        if laps and col in df.columns:
            df = df[df[col].round().astype(int).isin(laps)].copy()
            print(f"  -> Filtered by laps using '{col}': {original_count} -> {len(df)} rows")
            
    return df.reset_index(drop=True)

# --- 4. データの保存 ---

def save_preprocessed_h5(output_path: Path, df: pd.DataFrame, attributes: Dict, config: Dict):
    """処理済みのDataFrameを新しいHDF5ファイルに保存する。"""
    num_samples = len(df)
    print(f"  -> 💾 Writing {num_samples} cleaned samples to {output_path.name}...")
    
    cfg_keys = config.get('data_keys', {})
    cfg_output = config.get('output', {})
    
    with h5py.File(output_path, 'w') as f_out:
        # Scanデータセット
        vlen_float = h5py.vlen_dtype(np.float32)
        scan_dtype = np.dtype([
            (cfg_keys.get('time_columns', {}).get('sec', 'sec'), 'i4'),
            (cfg_keys.get('time_columns', {}).get('nanosec', 'nanosec'), 'u4'),
            (cfg_keys.get('scan_columns', {}).get('ranges', 'ranges'), vlen_float),
            (cfg_keys.get('scan_columns', {}).get('intensities', 'intensities'), vlen_float)
        ])
        dset_scan = f_out.create_dataset('scan', (num_samples,), dtype=scan_dtype, **hdf5plugin.Blosc())
        dset_scan.attrs.update(attributes)

        scan_data = np.empty(num_samples, dtype=scan_dtype)
        for col in scan_dtype.names:
            if col in df:
                scan_data[col] = df[col].tolist()
        dset_scan[:] = scan_data

        # Controlデータセット
        control_cols = cfg_output.get('control_columns_to_write', [])
        dset_control = f_out.create_dataset('control_cmd', data=df[control_cols].to_numpy(dtype=np.float32), **hdf5plugin.Blosc())

# --- 5. メイン処理とワーカー ---

def process_single_file(h5_path: Path, input_root: Path, output_root: Path, config: Dict):
    """単一ファイルを処理するためのワーカー関数。"""
    relative_path = h5_path.relative_to(input_root)
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Processing: {h5_path.name} ---")
    
    try:
        # Step 1: 読み込み
        scan_df, control_df, attrs = load_data_from_h5(h5_path)
        
        # Step 2: クリーニング
        scan_df = clean_scan_data(scan_df, attrs)
        
        # Step 3: 同期
        merged_df = synchronize_dataframes(scan_df, control_df, config['sync']['tolerance_seconds'])
        if merged_df.empty:
            print(f"  -> No data after synchronization. Skipping.")
            return

        # Step 4: フィルタリング
        final_df = filter_dataframe(merged_df, config.get('filtering', {}))
        if final_df.empty:
            print(f"  -> No data after filtering. Skipping.")
            return
            
        # Step 5: 保存
        save_preprocessed_h5(output_path, final_df, attrs, config)
        
    except Exception as e:
        print(f"  -> 💥 ERROR processing {h5_path.name}: {e}")

def main():
    """スクリプトのメインエントリポイント。"""
    parser = argparse.ArgumentParser(description='Preprocess HDF5 files by cleaning, synchronizing, and filtering.')
    parser.add_argument('input_dir', type=str, help='Directory containing HDF5 files to be preprocessed.')
    parser.add_argument('output_dir', type=str, help='Directory to save the filtered HDF5 files.')
    parser.add_argument('--config', type=str, default='config/preprocess.yaml', help='Path to the preprocess config file.')
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    config = load_config(Path(args.config))
    
    output_root.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(list(input_root.glob('**/*.h5')))

    if not h5_files:
        print(f"No .h5 files found in {input_root}")
        return
        
    print(f"Found {len(h5_files)} files to process.")
    num_jobs = config.get('num_workers', multiprocessing.cpu_count())
    print(f"Starting parallel processing with {num_jobs} workers...")

    worker_func = partial(process_single_file, input_root=input_root, output_root=output_root, config=config)
    
    with multiprocessing.Pool(processes=num_jobs) as pool:
        list(tqdm(pool.imap_unordered(worker_func, h5_files), total=len(h5_files), desc="Overall Progress"))
        
    print("\n--- ✅ All preprocessing finished. ---")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
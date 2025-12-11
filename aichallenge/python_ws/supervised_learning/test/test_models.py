import sys
sys.path.append('../') 

import numpy as np
import torch
import timeit
from typing import Dict

try:
    from src.model.tinylidarnet import TinyLidarNet, TinyLidarNetSmall, TinyLidarNetNp, TinyLidarNetSmallNp
    print("Successfully imported models from 'src.model.tinylidarnet'")
except ImportError as e:
    print(f"ImportError: {e}")
    print("Error: Could not import models.")
    print("Please make sure 'sys.path.append('../')' points to your project root directory,")
    print("and this script is running from a subdirectory (e.g., 'scripts/').")
    sys.exit(1)

# --- ユーティリティ関数 ---

def transfer_weights_pytorch_to_numpy(model_pt: torch.nn.Module, model_np):

    print(f"Transferring weights from PyTorch model ({model_pt.__class__.__name__}) to NumPy model ({model_np.__class__.__name__})...")
    pt_state_dict = model_pt.state_dict()
    
    transferred_count = 0
    for pt_key, pt_tensor in pt_state_dict.items():
        # PyTorchのキー ('conv1.weight') を NumPyのキー ('conv1_w') に変換
        np_key = pt_key.replace('.weight', '_w').replace('.bias', '_b')
        
        if np_key in model_np.params:
            # NumPy配列に変換してコピー
            model_np.params[np_key] = pt_tensor.detach().cpu().numpy()
            transferred_count += 1
        else:
            print(f"  [Warning] Key '{np_key}' (from '{pt_key}') not found in NumPy model params.")
    
    print(f"Successfully transferred {transferred_count} parameter sets.")

def run_benchmark(model_pt: torch.nn.Module, model_np, 
                  input_data_pt: torch.Tensor, input_data_np: np.ndarray, 
                  number: int, repeat: int):
    """
    指定されたモデルとデータで推論ベンチマークを実行します。
    """
    
    # --- 1. ウォームアップ ---
    # JITコンパイル、キャッシュなどを安定させるため
    print("  Warming up...")
    for _ in range(max(10, number // 10)): 
        with torch.no_grad():
            _ = model_pt(input_data_pt)
        _ = model_np(input_data_np)

    # --- 2. PyTorch (CPU) ベンチマーク ---
    print(f"  Running PyTorch benchmark (Repeat={repeat}, Number={number})...")
    stmt_pt = "with torch.no_grad(): model_pt(input_data_pt)"
    
    results_pt_total = timeit.repeat(
        stmt=stmt_pt,
        globals={'model_pt': model_pt, 'input_data_pt': input_data_pt, 'torch': torch},
        number=number,
        repeat=repeat
    )
    # 1回あたりの平均時間に変換 (秒)
    results_pt_sec = np.array(results_pt_total) / number
    
    # --- 3. NumPy ベンチマーク ---
    print(f"  Running NumPy benchmark (Repeat={repeat}, Number={number})...")
    stmt_np = "model_np(input_data_np)"
    
    results_np_total = timeit.repeat(
        stmt=stmt_np,
        globals={'model_np': model_np, 'input_data_np': input_data_np},
        number=number,
        repeat=repeat
    )
    results_np_sec = np.array(results_np_total) / number
    
    # 結果を辞書にまとめる (単位: ミリ秒)
    results_pt_ms = {'mean': np.mean(results_pt_sec) * 1000, 'std': np.std(results_pt_sec) * 1000}
    results_np_ms = {'mean': np.mean(results_np_sec) * 1000, 'std': np.std(results_np_sec) * 1000}
    
    return results_pt_ms, results_np_ms

def print_results(model_name: str, results_pt: Dict, results_np: Dict):
    """ベンチマーク結果を整形して表示します。"""
    
    mean_pt = results_pt['mean']
    std_pt = results_pt['std']
    mean_np = results_np['mean']
    std_np = results_np['std']
    
    print(f"\n--- Results for: {model_name} ---")
    print(f"  PyTorch (CPU): {mean_pt:9.4f} ms ± {std_pt:8.4f} ms")
    print(f"  NumPy (Custom):  {mean_np:9.4f} ms ± {std_np:8.4f} ms")
    
    if mean_np < mean_pt:
        speedup = mean_pt / mean_np
        print(f"  🚀 NumPy is {speedup:.2f}x faster.")
    else:
        slowdown = mean_np / mean_pt
        print(f"  🐢 PyTorch is {slowdown:.2f}x faster (or NumPy is slower).")
    print("-" * 40)

# --- メイン実行 ---
if __name__ == "__main__":
    
    # --- ベンチマーク設定 ---
    BATCH_SIZE = 1      
    INPUT_DIM = 1080
    NUMBER = 100        
    REPEAT = 7          
    
    print(f"--- CPU Inference Benchmark ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CPU: {torch.get_num_threads()} threads (PyTorch default)")
    print(f"Settings: Batch Size={BATCH_SIZE}, Input Dim={INPUT_DIM}")
    print(f"Timeit: Repeat={REPEAT}, Number={NUMBER} (Total runs per model: {REPEAT * NUMBER})")
    print("=" * 40)

    # --- ダミーデータの準備 ---
    print("Preparing dummy data...")
    # (B, C, L) 形式, float32
    np_data = np.random.rand(BATCH_SIZE, 1, INPUT_DIM).astype(np.float32)
    # PyTorchはCPUテンソルに
    pt_data = torch.from_numpy(np_data).to('cpu')

    # --- 1. TinyLidarNet (標準モデル) のベンチマーク ---
    print("\n[Benchmarking TinyLidarNet (Standard Model)]")
    
    # モデルをCPUに配置し、推論モード(.eval())に設定
    model_pt_large = TinyLidarNet(INPUT_DIM).to('cpu').eval()
    model_np_large = TinyLidarNetNp(INPUT_DIM)
    
    # 重みを転送
    transfer_weights_pytorch_to_numpy(model_pt_large, model_np_large)
    
    # ベンチマーク実行
    pt_res, np_res = run_benchmark(
        model_pt_large, model_np_large, pt_data, np_data,
        number=NUMBER, repeat=REPEAT
    )
    
    # 結果表示
    print_results("TinyLidarNet (Standard)", pt_res, np_res)
    
    # --- 2. TinyLidarNetSmall (軽量モデル) のベンチマーク ---
    print("\n[Benchmarking TinyLidarNetSmall (Lightweight Model)]")
    
    model_pt_small = TinyLidarNetSmall(INPUT_DIM).to('cpu').eval()
    model_np_small = TinyLidarNetSmallNp(INPUT_DIM)
    
    # 重みを転送
    transfer_weights_pytorch_to_numpy(model_pt_small, model_np_small)
    
    # ベンチマーク実行
    pt_res_small, np_res_small = run_benchmark(
        model_pt_small, model_np_small, pt_data, np_data,
        number=NUMBER * 2, 
        repeat=REPEAT
    )

    # 結果表示
    print_results("TinyLidarNetSmall (Lightweight)", pt_res_small, np_res_small)
    
    print("\nBenchmark finished.")
import sys
sys.path.append('../')  

import numpy as np
import timeit

try:
    import src.model.numpy.layers as np_layers
    import src.model.numba.layers as nb_layers
except ImportError as e:
    print(f"エラー: モジュールが見つかりません。")
    print(f"このスクリプトはプロジェクトのルートディレクトリから実行していますか？ (例: python {sys.argv[0]})")
    print(f"詳細: {e}")
    sys.exit(1)


def benchmark(layer_name, func_np, func_nb, args_np, args_nb, number=10):
    """
    指定されたNumPy関数とNumba関数の実行速度を比較します。
    """
    print(f"--- Benchmarking {layer_name} (実行回数={number}) ---")
    
    # Numbaのウォームアップ (JITコンパイルのため)
    try:
        # print("Numbaウォームアップ中...")
        func_nb(*args_nb)
    except Exception as e:
        print(f"Numba ウォームアップ失敗: {e}")
        return

    # NumPy版の計測
    # timeitに渡すグローバルスコープを設定
    np_globals = {'func': func_np, 'args': args_np}
    time_np = timeit.timeit(
        'func(*args)',  # 実行するコード
        globals=np_globals, 
        number=number
    ) / number

    # Numba版の計測
    nb_globals = {'func': func_nb, 'args': args_nb}
    time_nb = timeit.timeit(
        'func(*args)', 
        globals=nb_globals, 
        number=number
    ) / number

    time_np_ms = time_np * 1000
    time_nb_ms = time_nb * 1000

    print(f"  NumPy 実装: {time_np_ms:.4f} ミリ秒")
    print(f"  Numba 実装: {time_nb_ms:.4f} ミリ秒")
    
    if time_nb < time_np:
        print(f"  -> Numba は {time_np / time_nb:.2f} 倍 高速です 🚀")
    else:
        print(f"  -> NumPy は {time_nb / time_np:.2f} 倍 高速です")
    print("-" * (25 + len(layer_name)) + "\n")


def run_all_benchmarks():
    """
    すべてのレイヤーのベンチマークを実行します。
    """
    print("=" * 50)
    print("Numpy vs Numba レイヤー実装 速度比較")
    print("=" * 50 + "\n")

    # --- テスト設定 ---
    N = 32      # バッチサイズ
    C_IN = 16   # 入力チャネル
    C_OUT = 32  # 出力チャネル
    H = 64      # 高さ
    W = 64      # 幅
    L = 128     # シーケンス長
    K = 3       # カーネルサイズ (1D)
    K_H, K_W = 3, 3 # カーネルサイズ (2D)
    D_IN = 256  # Linear入力
    D_OUT = 512 # Linear出力
    STRIDE_1D = 1
    STRIDE_2D = (1, 1)
    POOL_K = (2, 2)
    POOL_S = (2, 2)
    
    # 実行回数 (重い処理は少なく、軽い処理は多く)
    RUNS_FAST = 100 
    RUNS_SLOW = 10  
    
    # データ型をFloat32に統一 (ディープラーニングで一般的)
    DTYPE = np.float32

    # --- 1. Linear ---
    x_linear = np.random.rand(N, D_IN).astype(DTYPE)
    w_linear = np.random.rand(D_OUT, D_IN).astype(DTYPE)
    b_linear = np.random.rand(D_OUT).astype(DTYPE)
    benchmark(
        "Linear (行列積)",
        np_layers.linear, nb_layers.linear,
        (x_linear, w_linear, b_linear),
        (x_linear, w_linear, b_linear),
        number=RUNS_SLOW
    )
    
    # --- 2. Conv1D ---
    x_conv1d = np.random.rand(N, C_IN, L).astype(DTYPE)
    w_conv1d = np.random.rand(C_OUT, C_IN, K).astype(DTYPE)
    b_conv1d = np.random.rand(C_OUT).astype(DTYPE)
    benchmark(
        "Conv1D (畳み込み)",
        np_layers.conv1d, nb_layers.conv1d,
        (x_conv1d, w_conv1d, b_conv1d, STRIDE_1D),
        (x_conv1d, w_conv1d, b_conv1d, STRIDE_1D),
        number=RUNS_SLOW
    )

    # --- 3. Conv2D ---
    x_conv2d = np.random.rand(N, C_IN, H, W).astype(DTYPE)
    w_conv2d = np.random.rand(C_OUT, C_IN, K_H, K_W).astype(DTYPE)
    b_conv2d = np.random.rand(C_OUT).astype(DTYPE)
    benchmark(
        "Conv2D (畳み込み)",
        np_layers.conv2d, nb_layers.conv2d,
        (x_conv2d, w_conv2d, b_conv2d, STRIDE_2D),
        (x_conv2d, w_conv2d, b_conv2d, STRIDE_2D),
        number=RUNS_SLOW
    )

    # --- 4. MaxPool2D ---
    # Conv2Dの出力(N, C_OUT, H_out, W_out)を想定した入力
    H_out = (H - K_H) // STRIDE_2D[0] + 1
    W_out = (W - K_W) // STRIDE_2D[1] + 1
    x_pool = np.random.rand(N, C_OUT, H_out, W_out).astype(DTYPE)
    benchmark(
        "MaxPool2D (プーリング)",
        np_layers.max_pool2d, nb_layers.max_pool2d,
        (x_pool, POOL_K, POOL_S),
        (x_pool, POOL_K, POOL_S),
        number=RUNS_SLOW
    )

    # --- 5. ReLU ---
    x_large = np.random.rand(N, C_OUT, H_out, W_out).astype(DTYPE) # 大きめのデータ
    benchmark(
        "ReLU (活性化関数)",
        np_layers.relu, nb_layers.relu,
        (x_large,), (x_large,),
        number=RUNS_FAST
    )

    # --- 6. Softmax ---
    x_softmax = np.random.rand(N, 1000).astype(DTYPE) # 多クラス分類を想定
    benchmark(
        "Softmax (出力関数)",
        np_layers.softmax, nb_layers.softmax,
        (x_softmax,), (x_softmax,),
        number=RUNS_FAST
    )

    # --- 7. Flatten ---
    benchmark(
        "Flatten (形状変更)",
        np_layers.flatten, nb_layers.flatten,
        (x_large,), (x_large,),
        number=RUNS_FAST
    )
    
    # --- 8. Sigmoid ---
    benchmark(
        "Sigmoid (活性化関数)",
        np_layers.sigmoid, nb_layers.sigmoid,
        (x_large,), (x_large,),
        number=RUNS_FAST
    )
    
    # --- 9. Tanh ---
    benchmark(
        "Tanh (活性化関数)",
        np_layers.tanh, nb_layers.tanh,
        (x_large,), (x_large,),
        number=RUNS_FAST
    )


if __name__ == "__main__":
    run_all_benchmarks()
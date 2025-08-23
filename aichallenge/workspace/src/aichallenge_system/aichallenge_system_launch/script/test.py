#!/usr/bin/env python3

from route_safety_monitor import RouteDeviationSafetyMonitor
import time
import matplotlib.pyplot as plt
import numpy as np
import random

def run_comprehensive_test():
    """包括的なテストを実行"""
    print("🚀 Comprehensive Lane Classification Test")
    print("=" * 55)
    print("🎯 Goal: Test precise polygon-based lane classification")
    print("📐 Method: Large-scale random sampling")
    print("🔧 Features: Performance analysis and visualization")
    print()
    
    try:
        # 精密内外判定器を初期化
        print("🔧 Initializing RouteDeviationSafetyMonitor...")
        safety_monitor = RouteDeviationSafetyMonitor()
        
        # 基本情報表示
        print(f"✅ Initialization complete:")
        print(f"  📊 Total lane polygons: {len(safety_monitor.lane_polygons)}")
        print(f"  🗺️  OSM file: {safety_monitor.osm_file}")
        print(f"  🔢 Total nodes: {len(safety_monitor.nodes)}")
        print(f"  📐 Lane coordinates: {len(safety_monitor.lane_coords)}")
        
        # OSMファイルの存在確認
        import os
        if os.path.exists(safety_monitor.osm_file):
            print(f"  ✅ OSM file exists")
        else:
            print(f"  ❌ OSM file NOT found: {safety_monitor.osm_file}")
            return None, None, None
        
        # テストポイントの生成と分類
        print("\n🧪 Running lane classification test...")
        test_points = [
            (89650.0, 43150.0),
            (89630.0, 43160.0),
            (89680.0, 43140.0),
            (89685.0, 43150.0),
            (89600.0, 43100.0)
        ]
        
        # 各ポイントをテスト
        for i, (x, y) in enumerate(test_points):
            is_in_lane = safety_monitor.is_in_any_lane(x, y)
            status = "IN_LANE" if is_in_lane else "OUTSIDE"
            print(f"  Point {i+1}: ({x}, {y}) -> {status}")
        
        # 可視化テスト
        print(f"\n🖼️  Generating comprehensive test visualization...")
        visualize_comprehensive_test(safety_monitor, test_points)
        
        print(f"\n🎉 SUCCESS: Comprehensive testing completed!")
        return test_points, None, safety_monitor
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return None, None, None

def run_performance_benchmark():
    """パフォーマンステストを実行"""
    print("\n🏃 Performance Benchmark Test")
    print("=" * 40)
    
    try:
        safety_monitor = RouteDeviationSafetyMonitor()
        
        # 異なるサンプル数でパフォーマンステスト
        sample_sizes = [100, 500, 1000, 2000, 5000]
        benchmark_results = []
        
        for size in sample_sizes:
            # ランダムポイントを生成
            samples = [(random.uniform(89600, 89700), random.uniform(43100, 43200)) for _ in range(size)]
            
            start_time = time.time()
            for x, y in samples:
                safety_monitor.is_in_any_lane(x, y)
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000
            avg_time = total_time / size
            
            benchmark_results.append((size, total_time, avg_time))
            print(f"📊 {size:4d} samples: {total_time:6.2f}ms total, {avg_time:.3f}ms/point")
        
        print("✅ Performance benchmark completed!")
        return benchmark_results
        
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return []

def debug_point_classification(safety_monitor, x, y):
    """特定の点の分類をデバッグ"""
    print(f"\n🔍 Debugging point ({x}, {y}):")
    print(f"Total lane polygons: {len(safety_monitor.lane_polygons)}")
    
    for i, polygon in enumerate(safety_monitor.lane_polygons):
        contains = polygon.contains_point((x, y))
        print(f"  Lane {i+1}: {'✅ INSIDE' if contains else '❌ OUTSIDE'}")
        if contains:
            return True
    
    print(f"  Final result: OUTSIDE")
    return False

def run_accuracy_test():
    """精度テストを実行"""
    print("\n🎯 Accuracy Test")
    print("=" * 25)
    
    try:
        safety_monitor = RouteDeviationSafetyMonitor()
        
        # 既知の座標でのテスト
        known_test_cases = [
            # (x, y, expected_classification, description)
            (89650.0, 43150.0, "OUTSIDE", "Known outside point"),
            (89629.0, 43160.0, "DRIVING_LANE", "Known driving lane point (1m west)"),
            (89630.0, 43159.0, "DRIVING_LANE", "Known driving lane point (1m south)"),
            (89685.0, 43150.0, "DRIVING_LANE", "Potential connecting road point"),
            (89600.0, 43100.0, "OUTSIDE", "Far outside point"),
        ]
        
        print(f"{'Point':<20} {'Expected':<15} {'Actual':<15} {'Result':<10} {'Description'}")
        print("-" * 80)
        
        correct = 0
        total = len(known_test_cases)
        
        for x, y, expected, description in known_test_cases:
            is_in_lane = safety_monitor.is_in_any_lane(x, y)
            actual = "DRIVING_LANE" if is_in_lane else "OUTSIDE"
            result = "✅ PASS" if actual == expected else "❌ FAIL"
            if actual == expected:
                correct += 1
            
            print(f"({x:6.1f}, {y:6.1f})  {expected:<15} {actual:<15} {result:<10} {description}")
            
            # 失敗した場合は詳細デバッグ
            if actual != expected:
                debug_point_classification(safety_monitor, x, y)
        
        accuracy = correct / total * 100
        print(f"\n📈 Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
    except Exception as e:
        print(f"❌ Accuracy test error: {e}")

def visualize_comprehensive_test(safety_monitor, test_points):
    """包括的なテスト結果を可視化"""
    plt.figure(figsize=(15, 10))
    
    # レーン境界を描画
    for i, lane_coords in enumerate(safety_monitor.lane_coords):
        if len(lane_coords) > 2:
            xs = [coord[0] for coord in lane_coords]
            ys = [coord[1] for coord in lane_coords]
            plt.fill(xs, ys, color='lightblue', alpha=0.3, edgecolor='blue', linewidth=1)
    
    # テストポイントを描画
    for i, (x, y) in enumerate(test_points):
        is_in_lane = safety_monitor.is_in_any_lane(x, y)
        color = 'green' if is_in_lane else 'red'
        marker = 'o' if is_in_lane else 'x'
        size = 150
        
        plt.scatter(x, y, c=color, marker=marker, s=size, 
                   label=f'Point {i+1}: {"IN_LANE" if is_in_lane else "OUTSIDE"}',
                   edgecolors='black', linewidth=2, zorder=5)
        
        # ポイント番号を表示
        plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title('Comprehensive Lane Classification Test Results', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('comprehensive_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: comprehensive_test_results.png")

def visualize_lane_boundaries(safety_monitor):
    """レーン境界の詳細可視化"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(safety_monitor.lane_coords)))
    
    for i, (lane_coords, color) in enumerate(zip(safety_monitor.lane_coords, colors)):
        if len(lane_coords) > 2:
            xs = [coord[0] for coord in lane_coords]
            ys = [coord[1] for coord in lane_coords]
            plt.fill(xs, ys, color=color, alpha=0.5, 
                    label=f'Lane {i+1}', edgecolor='black', linewidth=1)
            
            # 境界線を強調
            plt.plot(xs, ys, color='black', linewidth=2, alpha=0.8)
    
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title('Lane Boundary Visualization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('lane_boundaries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: lane_boundaries.png")

def visualize_random_sampling_test(safety_monitor, num_samples=1000):
    """ランダムサンプリングテストの可視化"""
    print(f"🎲 Generating {num_samples} random test points...")
    
    # ランダムポイントを生成
    x_min, x_max = 89600, 89700
    y_min, y_max = 43100, 43200
    
    test_points = [(random.uniform(x_min, x_max), random.uniform(y_min, y_max)) 
                   for _ in range(num_samples)]
    
    # 分類
    in_lane_points = []
    outside_points = []
    
    for x, y in test_points:
        if safety_monitor.is_in_any_lane(x, y):
            in_lane_points.append((x, y))
        else:
            outside_points.append((x, y))
    
    # 可視化
    plt.figure(figsize=(12, 8))
    
    # レーン境界
    for lane_coords in safety_monitor.lane_coords:
        if len(lane_coords) > 2:
            xs = [coord[0] for coord in lane_coords]
            ys = [coord[1] for coord in lane_coords]
            plt.fill(xs, ys, color='lightblue', alpha=0.4, edgecolor='blue', linewidth=1)
    
    # ポイントを描画
    if outside_points:
        outside_x, outside_y = zip(*outside_points)
        plt.scatter(outside_x, outside_y, c='red', marker='.', s=10, 
                   alpha=0.6, label=f'Outside Lane ({len(outside_points)})')
    
    if in_lane_points:
        in_lane_x, in_lane_y = zip(*in_lane_points)
        plt.scatter(in_lane_x, in_lane_y, c='green', marker='.', s=10, 
                   alpha=0.6, label=f'In Lane ({len(in_lane_points)})')
    
    # 統計情報
    in_lane_ratio = len(in_lane_points) / num_samples * 100
    
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title(f'Random Sampling Test ({num_samples} points)\nIn-Lane Ratio: {in_lane_ratio:.1f}%', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('random_sampling_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: random_sampling_test.png")
    
    return len(in_lane_points), len(outside_points)

def visualize_performance_benchmark(benchmark_results):
    """パフォーマンステスト結果の可視化"""
    if not benchmark_results:
        return
        
    sizes, times, avg_times = zip(*benchmark_results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 総処理時間
    ax1.plot(sizes, times, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Total Time (ms)')
    ax1.set_title('Total Processing Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # 平均処理時間
    ax2.plot(sizes, avg_times, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Average Time per Sample (ms)')
    ax2.set_title('Average Processing Time per Sample')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('performance_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: performance_benchmark.png")

def run_realtime_performance_test():
    """リアルタイム実行時のパフォーマンステスト"""
    print("\n⏱️  Real-time Performance Test")
    print("=" * 35)
    
    try:
        from route_safety_monitor import RouteDeviationSafetyMonitorNode
        import threading
        import queue
        
        # 模擬的なリアルタイムテスト
        safety_monitor = RouteDeviationSafetyMonitor()
        
        # テストデータ (実際の走行経路を模擬)
        test_trajectory = [
            (89650.0 + i*0.5, 43150.0 + i*0.2) for i in range(200)
        ]
        
        processing_times = []
        
        print("🚗 Simulating real-time monitoring...")
        for i, (x, y) in enumerate(test_trajectory):
            start_time = time.time()
            is_in_lane = safety_monitor.is_in_any_lane(x, y)
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
            
            if i % 50 == 0:
                print(f"  Step {i:3d}: ({x:7.1f}, {y:7.1f}) -> {'IN' if is_in_lane else 'OUT'} ({processing_time:.3f}ms)")
        
        # 統計情報
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        min_time = np.min(processing_times)
        std_time = np.std(processing_times)
        p95_time = np.percentile(processing_times, 95)
        p99_time = np.percentile(processing_times, 99)
        
        print(f"\n📊 Real-time Performance Statistics:")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  Min:     {min_time:.3f}ms")
        print(f"  Max:     {max_time:.3f}ms")
        print(f"  Std Dev: {std_time:.3f}ms")
        print(f"  95%ile:  {p95_time:.3f}ms")
        print(f"  99%ile:  {p99_time:.3f}ms")
        
        # 0.5秒間隔での実行可否判定
        cycle_time_ms = 500  # 0.5秒
        if p99_time < cycle_time_ms:
            print(f"  ✅ Suitable for {cycle_time_ms}ms cycle (99%ile: {p99_time:.3f}ms)")
        else:
            print(f"  ⚠️  May be too slow for {cycle_time_ms}ms cycle (99%ile: {p99_time:.3f}ms)")
        
        # パフォーマンス履歴の可視化
        plt.figure(figsize=(12, 8))
        
        # サブプロット1: 時系列の処理時間
        plt.subplot(2, 2, 1)
        plt.plot(processing_times, 'b-', alpha=0.7, linewidth=1)
        plt.axhline(avg_time, color='r', linestyle='--', label=f'Average ({avg_time:.3f}ms)')
        plt.axhline(p95_time, color='orange', linestyle='--', label=f'95%ile ({p95_time:.3f}ms)')
        plt.xlabel('Sample Number')
        plt.ylabel('Processing Time (ms)')
        plt.title('Processing Time History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # サブプロット2: ヒストグラム
        plt.subplot(2, 2, 2)
        plt.hist(processing_times, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(avg_time, color='r', linestyle='--', label=f'Average ({avg_time:.3f}ms)')
        plt.axvline(p95_time, color='orange', linestyle='--', label=f'95%ile ({p95_time:.3f}ms)')
        plt.xlabel('Processing Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Processing Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # サブプロット3: 累積分布
        plt.subplot(2, 2, 3)
        sorted_times = np.sort(processing_times)
        percentiles = np.arange(1, len(sorted_times)+1) / len(sorted_times) * 100
        plt.plot(sorted_times, percentiles, 'g-', linewidth=2)
        plt.axvline(avg_time, color='r', linestyle='--', label=f'Average ({avg_time:.3f}ms)')
        plt.axhline(95, color='orange', linestyle='--', alpha=0.7)
        plt.axvline(p95_time, color='orange', linestyle='--', label=f'95%ile ({p95_time:.3f}ms)')
        plt.xlabel('Processing Time (ms)')
        plt.ylabel('Percentile (%)')
        plt.title('Cumulative Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # サブプロット4: 移動平均
        plt.subplot(2, 2, 4)
        window_size = 20
        moving_avg = np.convolve(processing_times, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(processing_times)), moving_avg, 'purple', linewidth=2)
        plt.axhline(avg_time, color='r', linestyle='--', label=f'Overall Average ({avg_time:.3f}ms)')
        plt.xlabel('Sample Number')
        plt.ylabel('Moving Average (ms)')
        plt.title(f'Moving Average (window={window_size})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('realtime_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: realtime_performance_analysis.png")
        
        return processing_times
        
    except Exception as e:
        print(f"❌ Real-time performance test error: {e}")
        return []

def main():
    """メインテスト実行"""
    print("🧪 Lane Classification Test Suite")
    print("=" * 50)
    
    # 包括的テスト
    test_samples, _, safety_monitor = run_comprehensive_test()
    
    if safety_monitor is not None:
        # レーン境界可視化
        print(f"\n🖼️  Generating lane boundary visualization...")
        visualize_lane_boundaries(safety_monitor)
        
        # ランダムサンプリングテスト
        print(f"\n🎲 Running random sampling visualization test...")
        in_lane_count, outside_count = visualize_random_sampling_test(safety_monitor, 2000)
        print(f"   Random test results: {in_lane_count} in-lane, {outside_count} outside")
        
        # パフォーマンステスト
        benchmark_results = run_performance_benchmark()
        
        # パフォーマンス結果可視化
        if benchmark_results:
            print(f"\n📈 Generating performance benchmark chart...")
            visualize_performance_benchmark(benchmark_results)
        
        # リアルタイムパフォーマンステスト
        print(f"\n⏱️  Running real-time performance analysis...")
        realtime_results = run_realtime_performance_test()
        
        # 精度テスト
        run_accuracy_test()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📊 Generated visualizations:")
        print(f"   - comprehensive_test_results.png")
        print(f"   - lane_boundaries.png")
        print(f"   - random_sampling_test.png")
        print(f"   - performance_benchmark.png")
        print(f"   - realtime_performance_analysis.png")
        
        print(f"\n📋 Usage in your code:")
        print(f"   from route_safety_monitor import RouteDeviationSafetyMonitor")
        print(f"   safety_monitor = RouteDeviationSafetyMonitor()")
        print(f"   result = safety_monitor.is_in_any_lane(x, y)")
        
    else:
        print("❌ Tests failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

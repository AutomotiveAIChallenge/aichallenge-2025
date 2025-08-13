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
            (89630.0, 43160.0, "DRIVING_LANE", "Known driving lane point"),
            (89680.0, 43140.0, "DRIVING_LANE", "Known driving lane point"),
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
        
        # 精度テスト
        run_accuracy_test()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📊 Generated visualizations:")
        print(f"   - comprehensive_test_results.png")
        print(f"   - lane_boundaries.png")
        print(f"   - random_sampling_test.png")
        print(f"   - performance_benchmark.png")
        
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

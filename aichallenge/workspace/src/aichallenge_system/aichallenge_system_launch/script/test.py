#!/usr/bin/env python3

from precise_inout_distinguisher import PreciseInOutDistinguisher
import time

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
        print("🔧 Initializing PreciseInOutDistinguisher...")
        distinguisher = PreciseInOutDistinguisher()
        
        # 基本情報表示
        print(f"✅ Initialization complete:")
        print(f"  🛣️  Driving lanes: {len(distinguisher.driving_lanes)}")
        print(f"  🏪 Connecting roads: {len(distinguisher.connecting_roads)}")
        print(f"  📊 Total polygons: {len(distinguisher.driving_lane_polygons) + len(distinguisher.connecting_road_polygons)}")
        
        # 大量テストを実行
        print("\n🧪 Running large-scale test...")
        test_samples, classifications = distinguisher.run_precise_test(num_samples=500)
        
        # テストケース可視化
        print(f"\n🖼️  Generating test case visualization...")
        distinguisher.visualize_test_cases(test_samples, classifications)
        
        # 境界可視化
        print(f"\n🖼️  Generating precise boundary visualization...")
        distinguisher.visualize_precise_boundaries()
        
        print(f"\n🎉 SUCCESS: Comprehensive testing completed!")
        return test_samples, classifications, distinguisher
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return None, None, None

def run_performance_benchmark():
    """パフォーマンステストを実行"""
    print("\n🏃 Performance Benchmark Test")
    print("=" * 40)
    
    try:
        distinguisher = PreciseInOutDistinguisher()
        
        # 異なるサンプル数でパフォーマンステスト
        sample_sizes = [100, 500, 1000, 2000, 5000]
        
        for size in sample_sizes:
            samples = distinguisher.generate_test_samples(size)
            
            start_time = time.time()
            classifications = distinguisher.batch_classify_points(samples)
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000
            avg_time = total_time / size
            
            print(f"📊 {size:4d} samples: {total_time:6.2f}ms total, {avg_time:.3f}ms/point")
        
        print("✅ Performance benchmark completed!")
        
    except Exception as e:
        print(f"❌ Benchmark error: {e}")

def run_accuracy_test():
    """精度テストを実行"""
    print("\n🎯 Accuracy Test")
    print("=" * 25)
    
    try:
        distinguisher = PreciseInOutDistinguisher()
        
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
            actual = distinguisher.classify_point_precise(x, y)
            result = "✅ PASS" if actual == expected else "❌ FAIL"
            if actual == expected:
                correct += 1
            
            print(f"({x:6.1f}, {y:6.1f})  {expected:<15} {actual:<15} {result:<10} {description}")
        
        accuracy = correct / total * 100
        print(f"\n📈 Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
    except Exception as e:
        print(f"❌ Accuracy test error: {e}")

def main():
    """メインテスト実行"""
    print("🧪 Lane Classification Test Suite")
    print("=" * 50)
    
    # 包括的テスト
    test_samples, classifications, distinguisher = run_comprehensive_test()
    
    if distinguisher is not None:
        # パフォーマンステスト
        run_performance_benchmark()
        
        # 精度テスト
        run_accuracy_test()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📊 Generated visualizations:")
        print(f"   - test_case_visualization.png")
        print(f"   - precise_inout_distinguisher.png")
        
        print(f"\n📋 Usage in your code:")
        print(f"   from precise_inout_distinguisher import PreciseInOutDistinguisher")
        print(f"   distinguisher = PreciseInOutDistinguisher()")
        print(f"   result = distinguisher.classify_point_precise(x, y)")
        
    else:
        print("❌ Tests failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
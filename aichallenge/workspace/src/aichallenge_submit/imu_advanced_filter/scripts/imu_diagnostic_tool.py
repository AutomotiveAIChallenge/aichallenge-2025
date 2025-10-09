#!/usr/bin/env python3
"""
IMU高度フィルタ診断ツール
システムの状態を診断し、問題を特定します
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import subprocess
import time
import json
import sys

class ImuDiagnosticTool(Node):
    
    def __init__(self):
        super().__init__('imu_diagnostic_tool')
        
        # 診断結果保存
        self.diagnostic_results = {
            'system_connectivity': {'status': 'unknown', 'issues': []},
            'data_flow': {'status': 'unknown', 'issues': []},
            'parameter_validation': {'status': 'unknown', 'issues': []}
        }
        
        # データ受信確認用
        self.raw_imu_received = False
        self.filtered_imu_received = False
        self.odom_received = False
        
        # 購読設定（診断用）
        self.raw_imu_sub = self.create_subscription(
            Imu, '/sensing/imu/imu_basic_corrected', self.raw_imu_callback, 10)
        self.filtered_imu_sub = self.create_subscription(
            Imu, '/sensing/imu/imu_data', self.filtered_imu_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/localization/kinematic_state', self.odom_callback, 10)
        
        self.get_logger().info('IMU診断ツール初期化完了')
    
    def raw_imu_callback(self, msg):
        self.raw_imu_received = True
    
    def filtered_imu_callback(self, msg):
        self.filtered_imu_received = True
    
    def odom_callback(self, msg):
        self.odom_received = True
    
    def run_comprehensive_diagnosis(self):
        """包括的な診断を実行"""
        print("IMU高度フィルタ システム診断開始")
        print("=" * 60)
        
        # 診断1: システム接続状況
        print("1. システム接続状況の確認...")
        self.diagnostic_results['system_connectivity'] = self.diagnose_system_connectivity()
        self.print_diagnosis_result('システム接続', self.diagnostic_results['system_connectivity'])
        
        # 診断2: データフロー
        print("\n2. データフロー診断...")
        self.diagnostic_results['data_flow'] = self.diagnose_data_flow()
        self.print_diagnosis_result('データフロー', self.diagnostic_results['data_flow'])
        
        # 診断3: パラメータ設定
        print("\n3. パラメータ設定診断...")
        self.diagnostic_results['parameter_validation'] = self.diagnose_parameters()
        self.print_diagnosis_result('パラメータ設定', self.diagnostic_results['parameter_validation'])
        
        # 総合診断結果
        print("\n" + "=" * 60)
        self.generate_comprehensive_report()
        
        return self.diagnostic_results
    
    def diagnose_system_connectivity(self):
        """システム接続状況を診断"""
        result = {'status': 'good', 'issues': [], 'details': {}}
        
        # ROSノードの存在確認
        try:
            node_output = subprocess.run(['ros2', 'node', 'list'], 
                                       capture_output=True, text=True, timeout=10)
            nodes = node_output.stdout.split('\n')
            
            required_nodes = ['/imu_advanced_filter', '/simple_pure_pursuit_node']
            missing_nodes = []
            
            for required_node in required_nodes:
                if not any(required_node in node for node in nodes):
                    missing_nodes.append(required_node)
            
            if missing_nodes:
                result['status'] = 'warning'
                result['issues'].append(f"必要なノードが起動していません: {missing_nodes}")
            else:
                result['details']['nodes_ok'] = True
                
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"ノード確認エラー: {e}")
        
        return result
    
    def diagnose_data_flow(self):
        """データフロー診断"""
        result = {'status': 'good', 'issues': [], 'details': {}}
        
        # データ受信状況をテスト
        print("  データ受信テスト中... (5秒間)")
        
        # 受信フラグリセット
        self.raw_imu_received = False
        self.filtered_imu_received = False
        self.odom_received = False
        
        # 5秒間データ受信を待機
        start_time = time.time()
        while time.time() - start_time < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        # 受信結果確認
        if not self.raw_imu_received:
            result['status'] = 'error'
            result['issues'].append("フィルタ前IMUデータが受信されていません")
        
        if not self.filtered_imu_received:
            result['status'] = 'error'
            result['issues'].append("フィルタ後IMUデータが受信されていません")
        
        if not self.odom_received:
            result['status'] = 'warning'
            result['issues'].append("オドメトリデータが受信されていません")
        
        return result
    
    def diagnose_parameters(self):
        """パラメータ設定診断"""
        result = {'status': 'good', 'issues': [], 'details': {}}
        
        # 基本的なパラメータチェック
        try:
            # パラメータ存在確認（簡易版）
            param_output = subprocess.run([
                'ros2', 'param', 'list', '/imu_advanced_filter'
            ], capture_output=True, text=True, timeout=5)
            
            if param_output.returncode == 0:
                result['details']['parameters_accessible'] = True
            else:
                result['status'] = 'warning'
                result['issues'].append("パラメータにアクセスできません")
                
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"パラメータ確認エラー: {e}")
        
        return result
    
    def print_diagnosis_result(self, test_name, result):
        """診断結果を表示"""
        status_mark = "✅" if result['status'] == 'good' else "⚠️" if result['status'] == 'warning' else "❌"
        print(f"   {status_mark} {test_name}: {result['status'].upper()}")
        
        if result['issues']:
            for issue in result['issues']:
                print(f"      - {issue}")
    
    def generate_comprehensive_report(self):
        """包括的な診断レポートを生成"""
        print("包括的診断結果")
        print("-" * 40)
        
        # 全体的な健全性判定
        overall_status = 'good'
        total_issues = 0
        
        for category, result in self.diagnostic_results.items():
            if result['status'] == 'error':
                overall_status = 'error'
            elif result['status'] == 'warning' and overall_status == 'good':
                overall_status = 'warning'
            
            total_issues += len(result['issues'])
        
        # 総合評価
        if overall_status == 'good':
            print("🎉 総合評価: 良好 - システムは正常に動作しています")
        elif overall_status == 'warning':
            print("⚠️  総合評価: 注意 - 軽微な問題が検出されました")
        else:
            print("❌ 総合評価: 問題あり - 修復が必要です")
        
        print(f"検出された問題数: {total_issues}")

def main():
    rclpy.init()
    
    print("IMU高度フィルタ 診断ツール")
    print("=" * 50)
    print("システムの状態を診断し、問題を特定します")
    print("")
    
    diagnostic_tool = ImuDiagnosticTool()
    
    try:
        # 診断実行
        results = diagnostic_tool.run_comprehensive_diagnosis()
        
        # 結果をファイルに保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"/tmp/imu_diagnostic_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n診断結果をファイルに保存: {filename}")
        
    except KeyboardInterrupt:
        print("\n診断が中断されました")
    except Exception as e:
        print(f"\n診断中にエラーが発生: {e}")
    finally:
        diagnostic_tool.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

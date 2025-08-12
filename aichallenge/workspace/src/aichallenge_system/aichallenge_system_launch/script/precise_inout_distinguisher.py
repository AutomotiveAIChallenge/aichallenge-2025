#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import path
import matplotlib.pyplot as plt
from functools import lru_cache
import os
import random
import time

class PreciseInOutDistinguisher:
    def __init__(self, osm_file_path=None):
        # 複数のパスを試す
        possible_paths = [
            osm_file_path,
            './lanelet2_map.osm'
        ]
        
        self.osm_file = None
        for path in possible_paths:
            if path and os.path.exists(path):
                self.osm_file = path
                print(f"✅ Found OSM file: {path}")
                break
        
        if not self.osm_file:
            raise FileNotFoundError("lanelet2_map.osm not found")
        
        self.nodes = {}
        
        # 走行レーン関連
        self.driving_lanes = []  # メイン走行レーン
        self.connecting_roads = []  # 接続道路（車庫・出入り口）
        
        # 個別ポリゴン
        self.driving_lane_polygons = []  # 各走行レーンのポリゴン
        self.connecting_road_polygons = []  # 各接続道路のポリゴン
        
        # OSMファイルを解析
        try:
            tree = ET.parse(self.osm_file)
            self.root = tree.getroot()
            self._load_nodes()
            self._classify_lanelets()
            self._create_precise_polygons()
        except ET.ParseError as e:
            print(f"Error parsing OSM file: {e}")
            raise
    
    def _load_nodes(self):
        """全ノードの座標を辞書に格納"""
        nodes = {}
        for node in self.root.findall("node"):
            node_id = node.attrib['id']
            local_x = local_y = None
            
            for tag in node.findall('tag'):
                k = tag.attrib['k']
                if k == 'local_x':
                    local_x = float(tag.attrib['v'])
                elif k == 'local_y':
                    local_y = float(tag.attrib['v'])
                if local_x is not None and local_y is not None:
                    break
            
            if local_x is not None and local_y is not None:
                nodes[node_id] = {'x': local_x, 'y': local_y}
        
        self.nodes = nodes
        print(f"📊 Loaded {len(self.nodes)} nodes with coordinates")
    
    @lru_cache(maxsize=256)
    def _get_way_coordinates(self, way_id):
        """wayの座標リストを取得"""
        way = self.root.find(f"way[@id='{way_id}']")
        if way is None:
            return tuple()
        
        coords = [
            (self.nodes[nd.attrib['ref']]['x'], self.nodes[nd.attrib['ref']]['y'])
            for nd in way.findall('nd')
            if nd.attrib['ref'] in self.nodes
        ]
        return tuple(coords)
    
    def _classify_lanelets(self):
        """laneletを走行レーンと接続道路に分類"""
        print("🔍 Classifying lanelets based on geometry and position...")
        
        all_lanelets = []
        
        # 全laneletを収集
        for relation in self.root.findall("relation"):
            if relation.find("tag[@k='type'][@v='lanelet']") is not None:
                lanelet_id = relation.attrib.get('id')
                tags = {}
                left_way = right_way = centerline_way = None
                
                # タグ情報を取得
                for tag in relation.findall('tag'):
                    tags[tag.attrib['k']] = tag.attrib['v']
                
                # メンバー情報を取得
                for member in relation.findall("member"):
                    role = member.attrib.get('role')
                    ref = member.attrib.get('ref')
                    
                    if role == 'left':
                        left_way = ref
                    elif role == 'right':
                        right_way = ref
                    elif role == 'centerline':
                        centerline_way = ref
                
                if left_way and right_way:
                    left_coords = list(self._get_way_coordinates(left_way))
                    right_coords = list(self._get_way_coordinates(right_way))
                    centerline_coords = list(self._get_way_coordinates(centerline_way)) if centerline_way else []
                    
                    lanelet_data = {
                        'id': lanelet_id,
                        'tags': tags,
                        'left': left_coords,
                        'right': right_coords,
                        'centerline': centerline_coords,
                        'left_way': left_way,
                        'right_way': right_way,
                        'centerline_way': centerline_way
                    }
                    
                    all_lanelets.append(lanelet_data)
        
        # 改良された分類アルゴリズム
        self._improved_classification(all_lanelets)
        
        print(f"📋 Improved classification results:")
        print(f"  🛣️  Driving lanes: {len(self.driving_lanes)}")
        print(f"  🏪 Connecting roads: {len(self.connecting_roads)}")
    
    def _improved_classification(self, lanelets):
        """改良された分類（位置、長さ、形状を考慮）"""
        for lanelet in lanelets:
            # 分類要素
            factors = {}
            
            # 1. 長さによる判定
            centerline_length = self._calculate_path_length(lanelet['centerline']) if lanelet['centerline'] else 0
            left_length = self._calculate_path_length(lanelet['left'])
            right_length = self._calculate_path_length(lanelet['right'])
            max_length = max(centerline_length, left_length, right_length)
            factors['length'] = max_length
            
            # 2. 位置による判定（重心のX座標）
            all_points = lanelet['left'] + lanelet['right']
            avg_x = np.mean([p[0] for p in all_points])
            avg_y = np.mean([p[1] for p in all_points])
            factors['position_x'] = avg_x
            factors['position_y'] = avg_y
            
            # 3. 幅による判定
            width = self._calculate_average_width(lanelet['left'], lanelet['right'])
            factors['width'] = width
            
            # 4. 方向性による判定（直線性）
            straightness = self._calculate_straightness(lanelet['centerline'] or lanelet['left'])
            factors['straightness'] = straightness
            
            # 分類判定
            is_connecting = self._classify_lanelet(factors, lanelet['id'])
            
            if is_connecting:
                self.connecting_roads.append(lanelet)
                category = "🏪 CONNECTING"
                reason = self._get_classification_reason(factors, True)
            else:
                self.driving_lanes.append(lanelet)
                category = "🛣️  DRIVING"
                reason = self._get_classification_reason(factors, False)
            
            print(f"  Lanelet {lanelet['id']}: {category} - {reason}")
    
    def _classify_lanelet(self, factors, lanelet_id):
        """個別laneletの分類判定"""
        # 右側位置（車庫エリア）
        if factors['position_x'] > 89675:
            return True
        
        # 非常に短い（小さな接続部分）
        if factors['length'] < 15.0:
            return True
        
        # 幅が極端に狭い（接続部分の特徴）
        if factors['width'] < 2.0:
            return True
        
        # 上部エリア（出入り口付近）
        if factors['position_y'] > 43180 and factors['length'] < 30.0:
            return True
        
        # 複合判定：右側 + 短い + 狭い
        if (factors['position_x'] > 89665 and 
            factors['length'] < 25.0 and 
            factors['width'] < 3.5):
            return True
        
        return False
    
    def _get_classification_reason(self, factors, is_connecting):
        """分類理由を取得"""
        if is_connecting:
            if factors['position_x'] > 89675:
                return f"Right side (X={factors['position_x']:.1f})"
            elif factors['length'] < 15.0:
                return f"Very short ({factors['length']:.1f}m)"
            elif factors['width'] < 2.0:
                return f"Narrow ({factors['width']:.1f}m)"
            elif factors['position_y'] > 43180:
                return f"Upper area (Y={factors['position_y']:.1f})"
            else:
                return f"Combined factors"
        else:
            return f"Main lane (L={factors['length']:.1f}m, W={factors['width']:.1f}m)"
    
    def _calculate_path_length(self, coords):
        """パスの長さを計算"""
        if len(coords) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i-1][0]
            dy = coords[i][1] - coords[i-1][1]
            length += np.sqrt(dx*dx + dy*dy)
        
        return length
    
    def _calculate_average_width(self, left_coords, right_coords):
        """平均幅を計算"""
        if not left_coords or not right_coords:
            return 0.0
        
        # 対応点間の距離を計算
        distances = []
        min_len = min(len(left_coords), len(right_coords))
        
        for i in range(min_len):
            left_point = left_coords[i]
            right_point = right_coords[i]
            dist = np.sqrt((left_point[0] - right_point[0])**2 + 
                         (left_point[1] - right_point[1])**2)
            distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_straightness(self, coords):
        """直線性を計算（0=直線、1=曲線）"""
        if len(coords) < 3:
            return 0.0
        
        # 始点と終点を結ぶ直線の長さ
        start_point = coords[0]
        end_point = coords[-1]
        straight_distance = np.sqrt((end_point[0] - start_point[0])**2 + 
                                  (end_point[1] - start_point[1])**2)
        
        # 実際のパスの長さ
        actual_distance = self._calculate_path_length(coords)
        
        if straight_distance == 0:
            return 1.0
        
        # 曲線度 = (実際の長さ - 直線距離) / 直線距離
        curvature = (actual_distance - straight_distance) / straight_distance
        return min(curvature, 1.0)
    
    def _create_precise_polygons(self):
        """精密なポリゴンを各laneletから作成"""
        print("🔧 Creating precise polygons from left/right boundaries...")
        
        # 走行レーンのポリゴン
        for lanelet in self.driving_lanes:
            polygon = self._create_lanelet_polygon(lanelet)
            if polygon:
                polygon_path = path.Path(polygon[:-1])  # 最後の重複点を除く
                self.driving_lane_polygons.append({
                    'id': lanelet['id'],
                    'polygon': polygon,
                    'path': polygon_path
                })
        
        # 接続道路のポリゴン
        for lanelet in self.connecting_roads:
            polygon = self._create_lanelet_polygon(lanelet)
            if polygon:
                polygon_path = path.Path(polygon[:-1])  # 最後の重複点を除く
                self.connecting_road_polygons.append({
                    'id': lanelet['id'],
                    'polygon': polygon,
                    'path': polygon_path
                })
        
        print(f"  ✅ Created {len(self.driving_lane_polygons)} driving lane polygons")
        print(f"  ✅ Created {len(self.connecting_road_polygons)} connecting road polygons")
    
    def _create_lanelet_polygon(self, lanelet):
        """個別laneletからポリゴンを作成"""
        left_coords = lanelet['left']
        right_coords = lanelet['right']
        
        if not left_coords or not right_coords:
            return None
        
        # ポリゴン構築: 左境界 → 右境界（逆順）
        polygon_points = []
        
        # 左境界を順方向で追加
        polygon_points.extend(left_coords)
        
        # 右境界を逆方向で追加
        polygon_points.extend(reversed(right_coords))
        
        # 重複点を除去
        cleaned_points = self._remove_adjacent_duplicates(polygon_points)
        
        # ポリゴンを閉じる
        if len(cleaned_points) > 2 and cleaned_points[0] != cleaned_points[-1]:
            cleaned_points.append(cleaned_points[0])
        
        return cleaned_points if len(cleaned_points) >= 4 else None
    
    def _remove_adjacent_duplicates(self, points, tolerance=0.1):
        """隣接する重複点を除去"""
        if not points:
            return []
        
        cleaned = [points[0]]
        for point in points[1:]:
            last_point = cleaned[-1]
            dist = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            if dist > tolerance:
                cleaned.append(point)
        
        return cleaned
    
    def is_in_driving_lane(self, x, y):
        """走行レーン内かどうかを判定（精密ポリゴン使用）"""
        for polygon_data in self.driving_lane_polygons:
            if polygon_data['path'].contains_point((x, y)):
                return True
        return False
    
    def is_in_connecting_road(self, x, y):
        """接続道路内かどうかを判定（精密ポリゴン使用）"""
        for polygon_data in self.connecting_road_polygons:
            if polygon_data['path'].contains_point((x, y)):
                return True
        return False
    
    def is_in_any_lane(self, x, y):
        """いずれかのレーン内かどうかを判定"""
        return self.is_in_driving_lane(x, y) or self.is_in_connecting_road(x, y)
    
    def classify_point_precise(self, x, y):
        """点の位置を精密分類（接続レーンも走行レーンとして扱う）"""
        in_driving = self.is_in_driving_lane(x, y)
        in_connecting = self.is_in_connecting_road(x, y)
        
        if in_driving or in_connecting:
            return "DRIVING_LANE"  # 走行レーンまたは接続道路は全て走行レーン扱い
        else:
            return "OUTSIDE"
    
    def get_containing_lanes(self, x, y):
        """点を含むレーンのIDリストを取得"""
        containing_lanes = {
            'driving': [],
            'connecting': []
        }
        
        # 走行レーンをチェック
        for polygon_data in self.driving_lane_polygons:
            if polygon_data['path'].contains_point((x, y)):
                containing_lanes['driving'].append(polygon_data['id'])
        
        # 接続道路をチェック
        for polygon_data in self.connecting_road_polygons:
            if polygon_data['path'].contains_point((x, y)):
                containing_lanes['connecting'].append(polygon_data['id'])
        
        return containing_lanes
    
    def batch_classify_points(self, points):
        """複数点をバッチで精密分類"""
        results = []
        for x, y in points:
            classification = self.classify_point_precise(x, y)
            results.append(classification)
        return results
    
    def generate_test_samples(self, num_samples=500):
        """テスト用のランダムサンプルを大量生成"""
        samples = []
        
        # 全ポリゴンから範囲を取得
        all_points = []
        for polygon_data in self.driving_lane_polygons + self.connecting_road_polygons:
            all_points.extend(polygon_data['polygon'][:-1])
        
        if all_points:
            points_array = np.array(all_points)
            min_coords = np.min(points_array, axis=0)
            max_coords = np.max(points_array, axis=0)
            # マージンを追加
            margin_x = (max_coords[0] - min_coords[0]) * 0.2
            margin_y = (max_coords[1] - min_coords[1]) * 0.2
            extended_min = min_coords - [margin_x, margin_y]
            extended_max = max_coords + [margin_x, margin_y]
        else:
            extended_min = np.array([89580, 43100])
            extended_max = np.array([89720, 43210])
        
        # 多様なサンプル生成戦略
        strategies = [
            # 1. 完全ランダム（境界領域全体）
            lambda: (
                random.uniform(extended_min[0], extended_max[0]),
                random.uniform(extended_min[1], extended_max[1])
            ),
            # 2. レーン内密集（走行レーン近辺）
            lambda: (
                random.uniform(min_coords[0] + 10, max_coords[0] - 10),
                random.uniform(min_coords[1] + 10, max_coords[1] - 10)
            ),
            # 3. 境界付近（エッジケース）
            lambda: self._generate_boundary_near_point(),
            # 4. 外側領域
            lambda: self._generate_outside_point(extended_min, extended_max, min_coords, max_coords),
            # 5. 接続道路付近
            lambda: self._generate_connecting_area_point()
        ]
        
        # 各戦略で一定数のサンプルを生成
        samples_per_strategy = num_samples // len(strategies)
        remainder = num_samples % len(strategies)
        
        for i, strategy in enumerate(strategies):
            current_samples = samples_per_strategy + (1 if i < remainder else 0)
            for _ in range(current_samples):
                try:
                    sample = strategy()
                    if sample:
                        samples.append(sample)
                except Exception:
                    # フォールバック：完全ランダム
                    x = random.uniform(extended_min[0], extended_max[0])
                    y = random.uniform(extended_min[1], extended_max[1])
                    samples.append((x, y))
        
        return samples
    
    def _generate_boundary_near_point(self):
        """境界付近のポイントを生成"""
        if not self.driving_lane_polygons:
            return None
        
        # ランダムなポリゴンを選択
        polygon_data = random.choice(self.driving_lane_polygons + self.connecting_road_polygons)
        polygon_points = polygon_data['polygon'][:-1]
        
        if len(polygon_points) < 2:
            return None
        
        # ランダムな境界エッジを選択
        edge_start = random.choice(polygon_points)
        
        # エッジ付近にノイズを追加
        noise_x = random.uniform(-3, 3)
        noise_y = random.uniform(-3, 3)
        
        return (edge_start[0] + noise_x, edge_start[1] + noise_y)
    
    def _generate_outside_point(self, extended_min, extended_max, min_coords, max_coords):
        """外側領域のポイントを生成"""
        side = random.choice(['left', 'right', 'top', 'bottom'])
        
        if side == 'left':
            x = random.uniform(extended_min[0], min_coords[0])
            y = random.uniform(extended_min[1], extended_max[1])
        elif side == 'right':
            x = random.uniform(max_coords[0], extended_max[0])
            y = random.uniform(extended_min[1], extended_max[1])
        elif side == 'top':
            x = random.uniform(extended_min[0], extended_max[0])
            y = random.uniform(max_coords[1], extended_max[1])
        else:  # bottom
            x = random.uniform(extended_min[0], extended_max[0])
            y = random.uniform(extended_min[1], min_coords[1])
        
        return (x, y)
    
    def _generate_connecting_area_point(self):
        """接続道路エリア付近のポイントを生成"""
        # 右側エリア（車庫付近）に集中的にサンプル配置
        x = random.uniform(89675, 89700)
        y = random.uniform(43140, 43165)
        
        # ランダムノイズ追加
        x += random.uniform(-5, 5)
        y += random.uniform(-5, 5)
        
        return (x, y)
    
    def run_precise_test(self, num_samples=30):
        """精密内外判定テスト実行"""
        print(f"\n🧪 Running precise polygon-based test with {num_samples} random points...")
        
        # ランダムサンプル生成
        test_samples = self.generate_test_samples(num_samples)
        
        # バッチ分類
        start_time = time.time()
        classifications = self.batch_classify_points(test_samples)
        end_time = time.time()
        
        # 詳細分析のためのサンプル
        detailed_samples = test_samples[:5]
        
        # 結果集計
        category_counts = {}
        for classification in classifications:
            category_counts[classification] = category_counts.get(classification, 0) + 1
        
        # 結果表示
        print(f"\n=== 🎯 Precise Polygon Test Results ===")
        print(f"⚡ Processing time: {(end_time - start_time)*1000:.2f}ms for {num_samples} points")
        print(f"⚡ Average time per point: {(end_time - start_time)*1000/num_samples:.3f}ms")
        
        # 大量の場合は最初の10点のみ詳細表示
        show_details = min(10, num_samples)
        if show_details > 0:
            print(f"{'Sample':<8} {'Coordinates':<20} {'Classification':<16} {'Details'}")
            print("-" * 70)
            
            for i in range(show_details):
                x, y = test_samples[i]
                classification = classifications[i]
                containing = self.get_containing_lanes(x, y)
                details = f"D:{len(containing['driving'])}, C:{len(containing['connecting'])}"
                print(f"#{i+1:2d}      ({x:6.1f}, {y:6.1f})      {classification:<16} {details}")
            
            if num_samples > show_details:
                print(f"... ({num_samples - show_details} more samples)")
        
        # サンプル分布の詳細分析
        print(f"\n📈 Sample Distribution Analysis (Connecting roads treated as DRIVING_LANE):")
        strategy_names = ["Random", "Lane-Dense", "Boundary-Near", "Outside", "Connecting-Area"]
        samples_per_strategy = num_samples // 5
        
        for i, name in enumerate(strategy_names):
            start_idx = i * samples_per_strategy
            end_idx = start_idx + samples_per_strategy if i < 4 else num_samples
            strategy_classifications = classifications[start_idx:end_idx]
            strategy_counts = {}
            for c in strategy_classifications:
                strategy_counts[c] = strategy_counts.get(c, 0) + 1
            
            print(f"  {name:<15}: ", end="")
            for category, count in strategy_counts.items():
                percentage = count / len(strategy_classifications) * 100
                print(f"{category}({count}/{percentage:.1f}%) ", end="")
            print()
        
        print(f"\n📊 Precise Classification Summary:")
        for category, count in category_counts.items():
            percentage = count / num_samples * 100
            print(f"  {category:<20}: {count:2d} points ({percentage:4.1f}%)")
        
        return test_samples, classifications
    
    def visualize_test_cases(self, test_samples, classifications):
        """テストケースを可視化"""
        plt.figure(figsize=(20, 14))
        
        # 色とマーカーの定義（接続道路も走行レーン扱い）
        colors = {
            'DRIVING_LANE': 'blue',
            'OUTSIDE': 'gray'
        }
        markers = {
            'DRIVING_LANE': 'o',
            'OUTSIDE': 'x'
        }
        
        # 走行レーンポリゴン（薄く表示）
        for i, polygon_data in enumerate(self.driving_lane_polygons):
            polygon = polygon_data['polygon']
            x_coords = [coord[0] for coord in polygon]
            y_coords = [coord[1] for coord in polygon]
            plt.fill(x_coords, y_coords, color='lightblue', alpha=0.2, edgecolor='blue', linewidth=1)
        
        # 接続道路ポリゴン（薄く表示）
        for i, polygon_data in enumerate(self.connecting_road_polygons):
            polygon = polygon_data['polygon']
            x_coords = [coord[0] for coord in polygon]
            y_coords = [coord[1] for coord in polygon]
            plt.fill(x_coords, y_coords, color='lightgreen', alpha=0.2, edgecolor='green', linewidth=1)
        
        # テストポイントをプロット
        for (x, y), classification in zip(test_samples, classifications):
            plt.scatter(x, y, c=colors[classification], marker=markers[classification], 
                       s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # 凡例
        legend_elements = []
        for category in colors.keys():
            count = classifications.count(category)
            legend_elements.append(
                plt.Line2D([0], [0], marker=markers[category], color='w', 
                          markerfacecolor=colors[category], markersize=10, 
                          label=f'{category} ({count} points)')
            )
        
        plt.xlabel('Local X (m)')
        plt.ylabel('Local Y (m)')
        plt.title('Test Case Visualization - Point Classification Results')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('test_case_visualization.png', dpi=300, bbox_inches='tight')
        print("📊 Test case visualization saved as 'test_case_visualization.png'")

    def visualize_precise_boundaries(self):
        """精密境界を可視化"""
        plt.figure(figsize=(18, 12))
        
        # 走行レーンポリゴン
        for i, polygon_data in enumerate(self.driving_lane_polygons):
            polygon = polygon_data['polygon']
            x_coords = [coord[0] for coord in polygon]
            y_coords = [coord[1] for coord in polygon]
            
            color = plt.cm.Blues(0.3 + 0.1 * (i % 7))
            plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)
            plt.fill(x_coords, y_coords, color=color, alpha=0.15)
            
            # ラベル（最初の数個のみ）
            if i < 5:
                center_x = np.mean(x_coords[:-1])
                center_y = np.mean(y_coords[:-1])
                plt.text(center_x, center_y, f"D{polygon_data['id']}", 
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.1", facecolor='lightblue', alpha=0.7))
        
        # 接続道路ポリゴン
        for i, polygon_data in enumerate(self.connecting_road_polygons):
            polygon = polygon_data['polygon']
            x_coords = [coord[0] for coord in polygon]
            y_coords = [coord[1] for coord in polygon]
            
            color = plt.cm.Greens(0.3 + 0.1 * (i % 7))
            plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)
            plt.fill(x_coords, y_coords, color=color, alpha=0.15)
            
            # ラベル（最初の数個のみ）
            if i < 5:
                center_x = np.mean(x_coords[:-1])
                center_y = np.mean(y_coords[:-1])
                plt.text(center_x, center_y, f"C{polygon_data['id']}", 
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.1", facecolor='lightgreen', alpha=0.7))
        
        # 凡例用の要素
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.5, label=f'Driving Lanes ({len(self.driving_lane_polygons)})'),
            Patch(facecolor='lightgreen', alpha=0.5, label=f'Connecting Roads ({len(self.connecting_road_polygons)})')
        ]
        
        plt.xlabel('Local X (m)')
        plt.ylabel('Local Y (m)')
        plt.title('AI Challenge - Precise Lane Polygons (Left/Right Boundary Based)')
        plt.legend(handles=legend_elements, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('precise_inout_distinguisher.png', dpi=300, bbox_inches='tight')
        print("📊 Precise polygon visualization saved as 'precise_inout_distinguisher.png'")

if __name__ == '__main__':
    print("🚀 Precise Lane In/Out Distinguisher - Main Module")
    print("=" * 55)
    print("📋 This module provides the PreciseInOutDistinguisher class")
    print("🔧 Run test.py for testing and visualization")
    print()
    
    # 簡単な初期化テスト
    try:
        distinguisher = PreciseInOutDistinguisher()
        print(f"✅ Successfully initialized with {len(distinguisher.driving_lanes)} driving lanes and {len(distinguisher.connecting_roads)} connecting roads")
        
        # 簡単な使用例
        test_point = (89650.0, 43150.0)
        classification = distinguisher.classify_point_precise(*test_point)
        print(f"📍 Sample point {test_point}: {classification}")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
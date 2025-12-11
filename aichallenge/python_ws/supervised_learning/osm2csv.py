import xml.etree.ElementTree as ET
import csv
import argparse 

def parse_osm_to_csv(file_path, output_csv_path):

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        nodes = {}
        for node in root.findall('node'):
            node_id = node.get('id')
            lat = node.get('lat')
            lon = node.get('lon')
            local_x = None
            local_y = None
            ele = None
            for tag in node.findall('tag'):
                if tag.get('k') == 'local_x':
                    local_x = tag.get('v')
                elif tag.get('k') == 'local_y':
                    local_y = tag.get('v')
                elif tag.get('k') == 'ele':
                    ele = tag.get('v')
            nodes[node_id] = {
                'lat': lat,
                'lon': lon,
                'local_x': local_x,
                'local_y': local_y,
                'ele': ele
            }

        ways = {}
        for way in root.findall('way'):
            way_id = way.get('id')
            node_refs = [nd.get('ref') for nd in way.findall('nd')]
            ways[way_id] = node_refs

        output_data = []
        for relation in root.findall('relation'):
            is_lanelet = False
            for tag in relation.findall('tag'):
                if tag.get('k') == 'type' and tag.get('v') == 'lanelet':
                    is_lanelet = True
                    break
            
            if is_lanelet:
                relation_id = relation.get('id')
                for member in relation.findall('member'):
                    role = member.get('role')
                    if role in ['left', 'right']:
                        way_id = member.get('ref')
                        if way_id in ways:
                            node_sequence = ways[way_id]
                            for i, node_id in enumerate(node_sequence):
                                if node_id in nodes:
                                    node_info = nodes[node_id]
                                    output_data.append({
                                        'lanelet_id': relation_id,
                                        'way_id': way_id,
                                        'boundary_type': role,
                                        'node_id': node_id,
                                        'sequence_order': i + 1,
                                        'local_x': node_info['local_x'],
                                        'local_y': node_info['local_y'],
                                        'elevation': node_info['ele'],
                                        'latitude': node_info['lat'],
                                        'longitude': node_info['lon']
                                    })

        if not output_data:
            print("左/右の境界線を持つlaneletデータが見つかりませんでした。")
            return

        headers = [
            'lanelet_id', 'way_id', 'boundary_type', 'node_id', 
            'sequence_order', 'local_x', 'local_y', 'elevation', 
            'latitude', 'longitude'
        ]
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(output_data)
        
        print(f"CSVファイルを正常に作成しました: {output_csv_path}")
        print(f"抽出された合計点数: {len(output_data)}")

    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりませんでした。")
    except ET.ParseError:
        print(f"エラー: XMLファイル '{file_path}' の解析に失敗しました。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lanelet2の.osmファイルを解析し、左右のレーン境界線をCSVファイルに出力します。')
    parser.add_argument('input_file', type=str, help='入力するLanelet2 .osmファイルのパス')
    parser.add_argument('output_file', type=str, help='出力するCSVファイルのパス')
    args = parser.parse_args()

    parse_osm_to_csv(args.input_file, args.output_file)
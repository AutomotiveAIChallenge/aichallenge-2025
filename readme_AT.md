# aichallenge-2025

## データの記録

### 1. dockerを起動
```bash
## dockerを起動
bash docker_run.sh dev gpu
## docker内部でターミネータに入る
terminator
```

### 2. systemを起動 
recordオプションで記録ノードを起動するか切り替える。
記録ノードはコントローラで記録の開始と終了を制御するものである。
[記録ノードのyaml](aichallenge/workspace/src/aichallenge_submit/additinal_modules/tools/bag_manager_py/config/bag_manager.param.yaml)で録画したいTopicを選択する。
```bash
source workspace/install/setup.bash
ros2 launch original_launch system.launch.xml record:=true
```

## 学習プロセス

### 1. Rosbagから学習用にデータを抽出
[extract_data_from_bag.yaml](aichallenge/python_ws/supervised_learning/config/extract_data_from_bag.yaml)で抽出したいTopicや抽出したいラップ番号の指定が可能。
```bash
python3 extract_data_from_bag.py --output_dir ./datasets/ --search_dir /path/to/record/sequence/
```

### 2. 学習
[train.yaml](aichallenge/python_ws/supervised_learning/config/train.yaml)を調整することで学習パラメータを変更可能。また、hydraを使っているので、実行時に変更することも可能。
```bash
## 純粋な実行
python3 train.py

## もし、hydraでパラメータを書き換えたいとき
python3 train.py data.root=./datasets/sequence
```
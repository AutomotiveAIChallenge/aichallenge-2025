# Python Workspace

## 学習用データの作成
以下２つのTopicを含むrosbagを記録した後, extract_data_from_bag.pyを実行します。
sensor_msgs/msg/LaserScan
autoware_auto_control_msgs/msg/AckermannControlCommand

```bash
python3 extract_data_from_bag.py --bags-dir /path/to/record/ --outdir ./datasets/
```

## 学習
loss.accel_weightを0.0にすることで、ステアのみ学習を行うことが可能です。
アクセルの学習がうまく行かなかったため、まずはステアのみで学習することを推奨します。
```bash
python3 train.py \
data.train_dir=/path/to/train_dir \
data.val_dir=/path/to/val_dir \
model.name='TinyLidarNet' \
loss.steer_weight=1.0 \
loss.accel_weight=0.0 \ 
```

## 重みの形式変換
採点環境において実行できるように、pytorchではなくnumpyを用います。そのため、`.pth`から`.npy/.npz`に重みを変換します。
```bash
python3 convert_weight.py --model tinylidarnet --ckpt ./ckpts/weight.pth
```

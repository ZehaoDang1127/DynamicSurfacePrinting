# 从 cloth ConvLSTM 初始化 D405 模型

`prediction train --initialize-from` 支持用现有模型的权重初始化新的训练。可以将 cloth 模型作为起点，再用真实 D405 的打印机坐标高度图训练和验证。**cloth 权重本身不是当前打印装置可直接使用的模型。**

迁移只复制权重。目标数据决定网格、相机变换、坐标系和归一化；目标配置决定历史帧数、预测提前量和帧间隔。优化器从零开始。源模型和目标模型的 `hidden_channels`、`kernel_size` 必须完全一致，不支持静默丢弃不匹配层。

当前模型是全卷积网络，因而可以从 cloth 的 32×32 图迁移到不同尺寸的 D405 图。尺寸兼容仅表示参数能装入网络，不代表跨场景预测精度已经合格。

## 1. 准备目标记录和划分

在实际使用的 D405、相机固定姿态、打印机坐标变换、网格和采样频率下记录多个完整试验。每个目录应包含 `manifest.json` 和高度图帧文件。所有目标记录必须使用同一几何配置。

用整段试验划分训练、验证和测试集；相邻帧不能分到不同集合。目标应包含实际要预测的动态表面和工作范围。现有两段约3秒的静态 shadow 记录可用于软件检查，不能据此声称模型已经学会动态打印。

例如保存 `configs/d405_splits.json`：

```json
{
  "train": ["../output/d405_train_01", "../output/d405_train_02"],
  "validation": ["../output/d405_validation_01"],
  "test": ["../output/d405_test_01"]
}
```

把示例路径替换为已完成的实际记录。路径相对划分文件解析。训练仅用 `train` 计算归一化，并用 `validation` 选择最佳epoch；测试数据不参与参数更新或最佳epoch选择。

## 2. 选择匹配架构和目标时序

现有完整 cloth 权重是：

```text
output/tds_cloth/convlstm_full_20261012_200ms/model/best.pt
```

它使用 `[32, 64]` 隐藏通道、3×3卷积、5帧历史、5 Hz输入，预测200 ms之后的一张图。以下目标配置保持权重结构，改为30 Hz、10帧历史和未来600 ms的**单张预测图**：

```json
{
  "history": 10,
  "horizon": 18,
  "expected_dt_ms": 33.333333333333336,
  "timing_tolerance": 0.2,
  "min_valid_fraction": 0.1,
  "hidden_channels": [32, 64],
  "kernel_size": 3,
  "batch_size": 8,
  "learning_rate": 0.0001,
  "epochs": 50,
  "patience": 8,
  "grad_clip": 1.0,
  "seed": 42
}
```

可将它保存为 `configs/d405_transfer.json`。这是训练配置示例，600 ms不是实测打印延迟：实际提前量应根据相机、推理、通信、已提交运动及喷嘴行程的时序测量选择。`horizon=18` 表示预测第18帧时刻的一张图，不输出18张图，也不自动支持其间所有时刻。一个固定提前量模型不能直接满足控制器任意到达时间的请求。

若使用现有轻量 cloth 权重 `output/tds_cloth/benchmark/horizon_200ms/model/best.pt`，目标隐藏通道需改为 `[16, 16]`。

## 3. 在训练计算机运行

以下是 Bash 命令，从项目根目录执行。先激活已安装 `requirements-prediction.txt` 的环境；训练可在工作站完成，树莓派主要承担部署推理。

```bash
python -m prediction inspect \
  --splits configs/d405_splits.json \
  --config configs/d405_transfer.json

python -m prediction train \
  --initialize-from output/tds_cloth/convlstm_full_20261012_200ms/model/best.pt \
  --splits configs/d405_splits.json \
  --config configs/d405_transfer.json \
  --output output/d405_transfer_model_01 \
  --device auto --threads 4
```

在树莓派训练时使用 `--device cpu`；先评估内存和耗时。源权重需要实际存在于执行训练的机器上，部署ZIP不一定包含工作站的历史实验目录。每次训练使用新的输出目录。省略 `--initialize-from` 时保持原有随机初始化行为。

训练输出为 `best.pt`、`history.json`、`run.json`。后两种元数据及checkpoint中的 `initialization` 会记录源文件路径、SHA256、源配置和源几何，明确标记 `pretrained_weights_only`。checkpoint顶层的 `config`、`geometry`、`shape`、`normalization` 则来自新目标任务。源权重文件不被修改。

## 4. 验证并部署目标模型

```bash
python -m prediction evaluate \
  --checkpoint output/d405_transfer_model_01/best.pt \
  --splits configs/d405_splits.json --split test \
  --output output/d405_transfer_evaluation_01 \
  --device auto --threads 4
```

比较目标测试集上的ConvLSTM和基线误差，并检查预测支持区域、实际推理耗时和控制器需要的时间范围。迁移可能有帮助，也可能不如随机初始化；可另训同架构、同划分、相同训练预算的随机初始化版本作为对照。

将通过目标任务验证的新 `best.pt` 和它匹配的相机配置部署到树莓派。现有推理加载器仍检查几何、图尺寸和时序，不会因为使用了迁移学习就跳过这些检查。软件训练成功不等于相机标定、到达时序、机械跟踪或实体沉积已经完成验证。

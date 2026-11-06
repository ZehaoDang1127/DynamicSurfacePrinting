# Raspberry Pi：ConvLSTM 分段闭环实验版

版本：2026-10-27。ZIP：`dynamic_surface_printer_pi_convlstm_20261027.zip`。

这个版本包含 D405 连续采集、相机→打印机坐标变换、ConvLSTM 推理、喷头 XYZ/E 规划、Moonraker 短段提交和状态监控。它每完成一段就重新取最新相机历史、预测和规划；不是预先生成整条固定高度 G-code。

**当前交付是可测试的实验软件，尚不能用随包配置直接进行可靠的真实动态打印。** 包内已有的布料模型可以用于微调，但没有当前 D405 场景训练完成的模型；现有标定误差也大于普通首层间距。`live-check` 会列出这些未满足的条件，不能通过改文件名或把就绪标志全部改为 true 来代替实测。

## 1. 已实现的链路与实际含义

```text
D405 深度点（米）
  → 转成毫米，再应用相机到打印机的刚体变换 R、t
  → 打印机 XY 网格上的表面 Z 高度图
  → 最新 N 帧输入 ConvLSTM
  → 预测未来固定时刻的高度图
  → 在观测时刻与预测时刻之间做显式线性时间插值
  → 按预计喷头到达时间求表面高度
  → Z喷头 = Z预测表面 + 指定垂直间距
  → 计算下一小段的 XYZ、速度和相对挤出量 E
  → G1 + M400，确认结束状态后重新预测下一段
```

XY 来自 `profiles/live/path.json` 的打印图案，Z 随表面预测更新。没有实现布料横向变形时的材料点追踪，也没有改变喷嘴方向。挤出量按该时刻的空间表面路径长度计算，避免把表面整体上下运动误计成需要更多材料。

每段 `M400` 会等待队列排空，因此可能产生停顿、堆料或拉丝；这不是已验证的平滑连续跟随。相机线程持续更新有界历史，推理/规划不阻塞状态监控。普通 Moonraker 接口没有绝对执行时间调度保证，本程序使用声明的实测延迟范围并检测超时。网络回复不代表独立物理位置测量。

## 2. 关于布料模型和喷嘴间距

**布料模型可以作预训练起点。** 随包提供 `models/pretrained/cloth_600ms_16x16.pt`：16/16通道、布料数据、5 Hz、20 mm网格、预测600 ms。当前 D405 配置是30 Hz、2 mm网格、打印机坐标。模型全卷积结构可以迁移权重，但训练分布、时间设置和坐标几何必须用目标数据重新建立。

新增 `prediction train --initialize-from` 只复制网络参数；目标训练集重新计算归一化，优化器重新开始。不会继承布料模型的相机几何或假称模型已适配 D405。详细说明见包内 `docs/earlier_prediction/TRANSFER_LEARNING.md`。

对于原配0.4 mm喷嘴，OrcaSlicer 的 SV08 标准配置采用普通层高 **0.20 mm**、首层高度 **0.25 mm**。本包将0.25 mm作为首层垂直间距的候选目标，不把它视为柔性材料上已验证的沉积配方。
来源：[OrcaSlicer SV08 配置](https://github.com/OrcaSlicer/OrcaSlicer/blob/main/resources/profiles/Sovol/process/0.20mm%20Standard%20%40Sovol%20SV08%200.4%20nozzle.json)、[Prusa 首层校准](https://help.prusa3d.com/article/first-layer-calibration-i3_112364)。

本项目2026-10-25的四点深度标定三维拟合 RMS 为0.816 mm、最大残差1.140 mm、留一验证最大误差2.385 mm。用保存的四组点及矩阵重新计算，Z方向拟合残差RMS约0.322 mm、最大绝对值约0.333 mm；这些仍是拟合内残差，独立第五点尚未验证。现有配置中的1.2 mm标定预算和2 mm允许误差被原样保留，不能满足0.25 mm间距。因此执行检查会拒绝它；需要实际提高并独立验证标定精度，不能只调小 JSON 中的误差数字。

## 3. 在 Pi 上安装

以下 Bash 命令在 Raspberry Pi 本机终端执行。需要64位 Raspberry Pi OS / Debian、足够磁盘空间和首次安装联网。先在新目录解压，保留旧失败任务目录。

```bash
cd ~/Downloads
unzip dynamic_surface_printer_pi_convlstm_20261027.zip
cd Dynamic_Surface_Printer_Pi_ConvLSTM_20261027
bash setup_pi.sh --convlstm
bash run_pi.sh doctor --convlstm
bash run_pi.sh self-test
bash run_pi.sh live-self-test
```

`--convlstm` 创建独立 Python 3.12 环境，安装 PyTorch CPU、NumPy、httpx、RealSense及USB规则。ARM64 Python依赖wheel随包提供；系统包、Python运行时和规则首次准备仍需联网。若已有USB规则，安装脚本先备份。安装后重新插拔D405。

安装完成后不要直接移动虚拟环境所在目录；搬到新目录时重新安装。已有合适的环境可以使用：

```bash
PI_PYTHON=/absolute/path/to/venv/bin/python bash run_pi.sh doctor --convlstm
```

`live-self-test` 使用虚拟打印机，检查软件逻辑，不连接真实相机或打印机。Pi上的实际推理速度需要用真实模型测量。

## 4. 确认相机和坐标配置

```bash
bash run_pi.sh doctor --camera --convlstm
```

随包的 `profiles/live/camera.json` 是当前保存的 D405 标定配置，固定serial、640×480@30 FPS、14×22打印机坐标网格和非单位刚体变换。它仍是上述未独立验证的暂定标定。

先保持相机、夹具与坐标参考一致，完成独立点/区域检查，再更新实际使用的变换。`camera_to_output` 的平移单位为毫米。不要把 `configs/d405_camera_frame.json` 的单位阵相机深度直接当成打印机Z。

标定工具和流程见 `docs/geometry_calibration/CALIBRATION_WORKFLOW.md`、`docs/printer_control/STATIONARY_LINE_CHECK.md`；其中历史 `output/` 路径需要换成你保存的实际标定资料位置。运行这些额外工具时，先安装 `requirements-calibration.txt`（需要联网和OpenCV；本包默认运行闭环无需OpenCV），例如 `.venv-pi312/bin/python -m pip install -r requirements-calibration.txt`。

相机配置改变后，控制器的frame、grid及camera hash必须一致，模型也需要匹配新的几何。可计算hash：

```bash
.venv-pi312/bin/python -m controller camera-hash \
  --config profiles/live/controller.json \
  --camera-config profiles/live/camera.json
```

把实际结果记录到 `hardware.camera_config_sha256`，保留对应的标定记录。计算hash本身不验证精度。

## 5. 录制真实数据

完成坐标检查后，使用同一份相机配置记录多次独立的动态表面试验。每次使用新目录。示例单次1800帧在30 Hz下名义约60秒，实际时长以记录为准；这不是“录这些帧就足够”的训练保证。

```bash
.venv-pi312/bin/python -m surface_mapper \
  --config profiles/live/camera.json --source realsense \
  --frames 1800 --save-depth --output output/d405_train_01
```

分别再录 `d405_train_02`、`d405_validation_01`、`d405_test_01` 等完整独立试验，覆盖拟使用的运动、工作区和速度。不要把相邻帧随机分成训练/测试；不要把已有两段约3秒的静态shadow记录当成动态部署模型的充分数据。

采集只读取相机，不自动驱动夹具或打印机。停止其他占用相机的程序后再运行。

## 6. 从布料模型微调

可在笔记本/工作站训练，再把结果复制到Pi；以下是可以在Pi执行的CPU命令，但耗时取决于数据规模。复制所需配置并填写实际完整记录目录：

```bash
cp configs/splits_d405.example.json configs/d405_splits.json
cp configs/convlstm_d405_transfer.example.json configs/d405_training.json
nano configs/d405_splits.json
```

示例架构为16/16通道，与随包预训练权重匹配。默认10帧历史、30 Hz、`horizon=30`，即预测1秒后的单张图。**1秒是待测量后调整的工程起点，不是已测打印延迟，也不是30张未来图。** 改变目标提前量必须重新训练/验证，不能只改已训练checkpoint的元数据。

```bash
.venv-pi312/bin/python -m prediction train \
  --splits configs/d405_splits.json --config configs/d405_training.json \
  --initialize-from models/pretrained/cloth_600ms_16x16.pt \
  --output output/d405_model_01 --device cpu --threads 4

.venv-pi312/bin/python -m prediction evaluate \
  --checkpoint output/d405_model_01/best.pt \
  --splits configs/d405_splits.json --split test \
  --output output/d405_evaluation_01 --device cpu --threads 4
```

检查路径区域、对应预测提前量的误差、覆盖率和最差情况，并与基线比较。独立测试集用于最终评估；反复据测试集调参后，它就不能继续作为未使用的最终测试证据。微调可能改善也可能降低性能，需要实测。

评估完成后，将选定且与相机匹配的权重放到：

```text
models/d405/best.pt
```

控制器示例已经指向该路径。包内没有这个文件，是因为当前没有完成训练和评估的D405权重，不是漏打包。

## 7. 先独立运行实时 ConvLSTM

```bash
.venv-pi312/bin/python -m prediction live \
  --checkpoint models/d405/best.pt \
  --camera-config profiles/live/camera.json \
  --source realsense --frames 300 --device cpu --threads 4 \
  --output output/d405_convlstm_live_01
```

检查输出预测、实际推理耗时、帧间隔和重置次数。这个命令没有打印机连接。预测速度必须为后续到达时间规划留下足够余量；不能把相机设为30 FPS就认定整条闭环达到30 Hz。

## 8. 设置闭环并检查

两份主要设置：

| 文件 | 内容 |
|---|---|
| `profiles/live/controller.json` | 相机几何身份、模型路径、网格、间距、速度/加速度/流量边界、误差预算、Moonraker地址 |
| `profiles/live/settings.json` | 实际初始XYZ、曝光/USB延迟、时钟不确定度、命令开始延迟、调度误差范围、温度及运行上限、实际验证记录标志 |

`null` 表示尚未测量。时序字段必须包括当前Pi、网络和固件的实测结果；未知不能填0来跳过检查。默认地址是 `http://192.168.50.2:7125`，按实际有线网络修改。需要API key时使用 `MOONRAKER_API_KEY` 环境变量。

`initial_xyz_mm` 必须对应第一路径点的真实喷嘴XYZ。默认路径是X170→190、Y83。程序不执行自动接近、回零或从旧Z150位置盲目下降。坐标参考因故障失效时需要单独重新建立。

```bash
bash run_pi.sh live-check
```

这是离线配置/模型检查，不连接打印机。默认应报告缺失D405模型、缺失测量、未验证标定等问题。它会保存 `checks.json`；`READY`只表示所提供输入通过软件检查，不证明填写的测量已被独立证实。

时间插值模式 `linear_to_horizon` 会将最新实测图与ConvLSTM未来终点之间线性连接。插值期间的误差需要单独评估并计入 `interpolation_error_mm`；不允许超出模型提前量外推，也不自动降低误差预算。

## 9. 相机→预测→喷嘴坐标预览

在相机配置、匹配模型、起始XYZ和时序字段齐全后运行：

```bash
bash run_pi.sh live-shadow
```

它使用同一套采集、预测和规划逻辑产生分段候选坐标，但不连接打印机，不加热、不运动、不挤出。目录内的 `events.jsonl`、观察帧、预测图和报告用于核对。缺少支持、超出模型提前量、推理超时、间距预算不足时会停止并记录原因。

## 10. 条件满足后执行实验性分段打印

只有真实模型、独立标定精度、时序范围、材料配方、坐标参考和物理停止方式均得到验证，且实际起点已建立时，才使用此入口。当前随包配置并不满足这些条件。

```bash
bash run_pi.sh live-print --execute
```

未加 `--execute` 时仍为无硬件执行的shadow流程。显式执行会读取打印机状态，确认相机数据与初始位置，准备喷嘴温度，逐段发送XYZ/E并等待M400。每段后重新获取最新历史并规划，任何已发送请求若回复不确定都不会自动重发。

程序会按所规划短段设置并读回实际Klipper加速度和受支持的加减速平滑参数，避免仅发送F速度却假设另一套加速度。未知或不匹配的固件状态字段会被拒绝。正常结束恢复本次修改的运动限制；异常后保持停止，需核对当前状态再操作。

程序监控模式、偏移、固件会话、温度、坐标、相机支持和时效。故障会保存被拒绝的状态样本并尝试专用急停；不会自动重启、恢复任务或在坐标不确定时抬升。正常结束关闭喷嘴加热；默认无自动接近或返回动作。不要将该软件监控当成独立物理间距传感器。

真实表面在停顿期间仍可能运动，因此每段停稳的沉积质量、开头/结尾行为都需要实验确认。需要平滑连续跟随时，下一步是验证有明确排程观测的连续队列接口，而不是去掉M400后继续沿用当前时间估计。

协议参考：[Moonraker G-code API](https://moonraker.readthedocs.io/en/latest/external_api/printer/#run-a-gcode-command)、[Klipper G-code](https://www.klipper3d.org/G-Codes.html)、[Klipper状态语义](https://www.klipper3d.org/Status_Reference.html#motion_report)。

## 11. 输出与验证范围

每次运行使用新的 `output/pi_*` 目录。保留 `checks.json`、`live_settings.json`、`camera_config.json`、`config.json`、`events.jsonl`、`last_status_snapshot.json` 和 `report.json`，以便将模型、时间和指令对应起来。原静态任务诊断可继续使用 `bash run_pi.sh diagnose --session /旧失败目录的绝对路径`。

本包在开发机通过软件和虚拟硬件测试；ARM64/Python3.12依赖文件已按目标平台下载。尚未在你的Pi、相机和打印机组合上进行真实闭环运动/沉积验证；不存在可据此承诺的打印频率或精度。之前的全零/偏离路径停止事件也尚未确认根因，当前程序保留状态快照和原有拒绝行为。

具体检查结果见 `docs/raspberry_pi/PI_CONVLSTM_VALIDATION_20261027.md`：完整回归177项通过、1项因测试环境缺少RealSense SDK而跳过；解压后的15项虚拟闭环测试也已通过。

# 树莓派上的 D405 采集与表面预测

更新：2026-10-15。目标：把采集、生成高度图和预测下一时刻表面放到树莓派上。
这一步先验证预测输出，不驱动打印机。训练可以继续在笔记本/工作站上进行。

| 设备 | 当前任务 |
|---|---|
| 笔记本/工作站 | 编辑代码、训练模型、把部署包和选定的权重复制到树莓派 |
| 树莓派 5 | 连接 D405，运行采集、高度图投影、基线或 ConvLSTM 推理 |
| 打印机 H616 | 保持现有 Moonraker/Klipper 服务；独立的预测命令不会发送运动指令 |

**下面所有命令都在树莓派本机 Terminal 运行。**
你的树莓派本机提示符是 `haimlab@raspberrypi`。在项目根目录执行 Python 命令。

## 1. 复制并安装项目

将 `pi_controller_setup_20261015_02.zip` 从笔记本复制到树莓派。
该版本补入 D405 示例相机配置、ConvLSTM 示例配置和本指南。
在树莓派解压；如果已有旧版项目，先保留它，在新目录解压新版，避免覆盖本机配置和记录。

进入解压出的 `Dynamic_Surface_Printer` 目录后执行：

```bash
python3 -m venv .venv-controller
source .venv-controller/bin/activate
python -m pip install -r requirements-controller.txt
```

这安装 NumPy 和 Moonraker 客户端。Windows 的 `.venv` / `.venv-tds` 不复制到树莓派。
项目目前在 Linux 上不通过 `requirements.txt` 自动安装 RealSense Python 绑定，
因此完成本节并不意味着相机驱动已经准备好。

## 2. 先用虚拟表面验证程序

无需接相机、无需训练模型、无需连接打印机，先运行：

```bash
python -m prediction baseline-live \
  --source synthetic \
  --camera-config configs/d405_camera_frame.json \
  --history 5 --frames 90 \
  --output output/pi_synthetic_prediction_01
```

这是现有的 NumPy 常加速度基线。它检查预测软件链路，不证明真实表面预测准确率。
应看到 `forecast=...` 日志，结束摘要中 `complete` 为 `true`、`forecasts` 大于 0。
每次使用新输出目录。

## 3. 接入 D405 并检查 SDK

这台树莓派已确认使用 ARM64、Debian 13 和 Python 3.13.5。
对应的 Python 3.12 独立环境安装步骤见
[pyrealsense2 安装指南](PI_REALSENSE_INSTALL.md)。采用该步骤后，
本指南后面的命令统一使用 `.venv-pi312` 环境。

把 D405 接到树莓派的合适 USB 接口，使用稳定的数据线并保证树莓派供电正常。
在上面已经激活的 Python 环境中检查：

```bash
python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().query_devices()])"
```

应枚举到 D405。如果导入失败，需要在树莓派上安装匹配系统架构、Python 版本的
librealsense / `pyrealsense2`。如果当前平台没有兼容的预编译包，则需要构建绑定。
官方 Python 安装及构建说明见
[RealSense Python wrapper](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/readme.md)。

为选择相机依赖的安装方式，可先获取：

```bash
uname -m
python --version
cat /etc/os-release
```

系统里安装过 SDK，也仍需确认当前这个虚拟环境能导入绑定。
能导入但枚举为空时，检查 USB 连接、Linux 设备权限及实际 SDK 安装；
已枚举但无法打开流时，还需检查其他程序是否正在使用相机和当前流模式是否可用。

## 4. 先采集真实高度图

SDK 和相机可用后，运行短采集：

```bash
python -m surface_mapper \
  --config configs/d405_camera_frame.json \
  --source realsense --frames 90 --print-every 30 \
  --output output/pi_d405_capture_01
```

检查能否持续采集、覆盖率和实际帧率，并查看生成的记录和最新高度图。
该相机配置是相机坐标示例，变换矩阵为单位阵；它输出的是相机深度，尚未标定为打印机 Z。
它可以用于检查采集和预测，不能直接当作喷嘴目标高度。

## 5. 对真实表面运行基线预测

采集检查结束后，再运行：

```bash
python -m prediction baseline-live \
  --source realsense \
  --camera-config configs/d405_camera_frame.json \
  --history 5 --frames 300 \
  --output output/pi_d405_baseline_01
```

这个命令只预测并写日志，不发送 Moonraker 指令。
默认预测提前量是相机配置的一帧间隔；30 FPS 配置对应约 33.3 ms。
检查预测有效区域、帧丢失/历史重置和实际耗时，不把配置中的 30 FPS 当成实测帧率。
相机采集与预测命令依次运行，避免两个进程同时打开相机。

## 6. 切换到已有 ConvLSTM 权重

在树莓派同一环境安装预测依赖：

```bash
python -m pip install -r requirements-prediction.txt
```

复制在匹配相机几何、网格和时间间隔上训练的 checkpoint。
下面 `models/d405_best.pt` 是你自行复制的模型文件示例，不包含在部署包中。
如果训练时使用另一份相机配置，这里应使用那份匹配的配置。

```bash
python -m prediction live \
  --checkpoint models/d405_best.pt \
  --camera-config configs/d405_camera_frame.json \
  --source realsense --frames 300 \
  --device cpu \
  --output output/pi_d405_convlstm_01
```

现有 live 命令严格检查 checkpoint 几何和相机时间间隔。
布料数据集、虚拟 demo 或其他相机设置的权重不能据此当作已验证的 D405 打印模型。
在树莓派上测量实际推理耗时，再决定模型规模、输入区域和运行频率。

## 7. 后续接入打印机控制

独立的 `surface_mapper` / `prediction` 命令负责测量和预测；
`controller` 包承担预测适配、路径规划和打印机通信。
当前已实现虚拟动态控制、只读发现和受限制的静态冷态运动试验，
动态表面实机打印仍需坐标变换、运动时序、夹具和沉积验证。

基底高度可以继续保持 `null`，不影响独立采集/预测或 Moonraker 只读发现。
在实机控制阶段，再测量基底高度并将相机结果标定到打印机坐标。

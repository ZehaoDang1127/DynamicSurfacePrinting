# 树莓派 ARM64 / Debian 13 安装 pyrealsense2

更新：2026-10-15。适用这台已确认的设备：`aarch64`、Debian 13.2、Python 3.13.5，
项目位于 `~/Desktop/Dynamic_Surface_Printer`，现有环境为 `.venv-controller`。
以下是树莓派本机操作步骤，尚未代替你在设备上执行。

## 为什么新建 Python 3.12 环境

截至本次查询，官方 `pyrealsense2==2.59.1.11053` 发布了 CPython 3.12 的
`manylinux2014_aarch64` wheel，没有发布 CPython 3.13 的 ARM64 wheel。
本流程用 uv 安装独立 Python 3.12，创建 `.venv-pi312`。
系统 Python 不改动；原来的 `.venv-controller` 保留，
只在其中安装用于创建新环境的 uv。
[官方 PyPI 发布文件](https://pypi.org/project/pyrealsense2/2.59.1.11053/#files)。

## 1. 在树莓派本机进入项目

使用提示符为 `haimlab@raspberrypi` 的本机 Terminal，在这个窗口执行：

```bash
cd ~/Desktop/Dynamic_Surface_Printer
source .venv-controller/bin/activate
python -m pip install --upgrade uv
python -m uv python install 3.12
python -m uv venv --python 3.12 --seed .venv-pi312
```

`--seed` 为新环境安装 pip。uv 的 Python 运行时下载需要网络连接。
如果 `.venv-pi312` 已经存在并且是之前创建的正确环境，直接使用下一节的激活命令。
[uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)、
[uv Python 管理说明](https://docs.astral.sh/uv/guides/install-python/)。

## 2. 激活新环境，安装项目与相机绑定

```bash
source .venv-pi312/bin/activate
python --version
python -m pip install -r requirements-controller.txt
python -m pip install --only-binary=:all: "pyrealsense2==2.59.1.11053"
```

确认版本输出为 Python 3.12.x。环境提示符应变成 `(.venv-pi312)`。
`--only-binary` 要求使用预编译包；如果提示没有匹配的包，先确认此窗口中的
`python --version`、`uname -m` 和 `python -m pip --version`，不继续切换到其他不匹配的 wheel。

检查导入和环境：

```bash
python -c "import sys, pyrealsense2 as rs; print(sys.executable); print(rs.__file__)"
```

解释器和模块路径应属于当前 `.venv-pi312`。

## 3. 接相机并枚举

把 D405 用可靠的数据线直接接到树莓派，然后在同一个环境运行：

```bash
python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().query_devices()])"
```

应看到包含 D405 的设备列表。导入成功只说明 Python 模块已安装；
枚举成功和实际采集成功还分别需要 USB 连接、权限与流配置正常。

## 4. 需要时设置 Linux USB 权限

如果导入正常，但枚举为空或提示 USB 权限错误，先检查数据线和连接。
可以按官方 SDK 的方式安装 RealSense udev 规则。

先下载对应 SDK tag 的规则文件：

```bash
curl -fL https://raw.githubusercontent.com/realsenseai/librealsense/v2.59.1/config/99-realsense-libusb.rules \
  -o /tmp/99-realsense-libusb.rules
```

下载成功后再安装。如果 `/etc/udev/rules.d/99-realsense-libusb.rules` 已存在，
先保存一份原文件备份，再更新它。

```bash
sudo install -m 0644 /tmp/99-realsense-libusb.rules /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb
```

重新插拔 D405，然后再次运行枚举命令。
[官方规则文件](https://github.com/realsenseai/librealsense/blob/v2.59.1/config/99-realsense-libusb.rules)、
[官方 Linux 权限安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md)。

## 5. 采集一次真实数据

在项目根目录、同一个 `.venv-pi312` 环境执行。这里的“项目根目录”
是直接包含 `surface_mapper` 和 `configs` 文件夹的目录。
先确认源代码已经完整解压：

```bash
cd ~/Desktop/Dynamic_Surface_Printer
ls surface_mapper/__main__.py configs/d405_camera_frame.json
```

两项文件均存在时，执行：

```bash
python -m surface_mapper \
  --config configs/d405_camera_frame.json \
  --source realsense --frames 90 --print-every 30 \
  --output output/pi_d405_capture_01
```

每次使用新的输出目录。确认实际帧率和有效区域，再按
[树莓派预测指南](PI_PREDICTION_SETUP.md) 运行基线预测。
该相机示例仍输出相机坐标深度，不代表完成了打印机坐标标定。

### 如果报 No module named surface_mapper

`surface_mapper` 是本项目源代码中的目录。先检查当前目录是否直接包含它；
有些解压操作会生成额外一层 `Dynamic_Surface_Printer` 文件夹。
可以在树莓派本机寻找：

```bash
find ~/Desktop -maxdepth 5 -type d -name surface_mapper
```

进入查到的 `surface_mapper` 的上一级目录，再运行 `python -m surface_mapper`。
如果完全找不到，重新复制并完整解压部署包中的项目源代码。
只有安装依赖文件并不会自动创建本项目的源代码目录。

## 此后每次运行

重新打开 Terminal 后，使用新的环境：

```bash
cd ~/Desktop/Dynamic_Surface_Printer
source .venv-pi312/bin/activate
```

接相机的采集、预测和控制器应使用这个已经安装相机绑定的环境。
如果后续运行 ConvLSTM，再在此环境安装 `requirements-prediction.txt`。
独立采集/预测不会向打印机发送运动指令。

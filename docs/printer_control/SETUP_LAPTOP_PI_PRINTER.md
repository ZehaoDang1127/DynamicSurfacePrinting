# 笔记本、树莓派、打印机：按设备划分的设置指南

更新：2026-10-15。目标：树莓派通过一根网线调用 SV08 上的 Moonraker。
这份指南说明操作位置和顺序；尚未替你在三台设备上执行设置。

## 三台设备各自负责什么

| 设备 | 设置时的任务 | 后续运行时的任务 | 本指南选择的有线地址 |
|---|---|---|---|
| Windows 笔记本 | 备份打印机配置、SSH 配置打印机、向树莓派传项目 | 开发、训练、查看结果；控制器部署到树莓派后，笔记本无需参与打印控制通信 | 不需要设置 |
| 树莓派 5 | 设置自己的有线 IP、安装项目、检查 Moonraker | 接 D405，运行采集、预测和路径控制器 | `192.168.50.1/24` |
| SV08 的 H616 | 设置自己的有线 IP、保持已有 Moonraker 和 Klipper 服务 | Moonraker 接收请求；Klipper 与打印机内部 MCU 通信 | `192.168.50.2/24` |

```text
笔记本 ── Wi-Fi / SSH ── 打印机 H616（设置和管理）
   └── 将项目文件复制到树莓派

D405 ── USB ── 树莓派 5 ── 一根网线 ── 打印机 H616 ── 原有内部连接 ── 两个 MCU
              预测、路径控制          Moonraker + Klipper
```

**固定使用两个操作窗口：**

- **窗口 A：笔记本 PowerShell，通过 SSH 登录打印机。登录成功后，这个窗口里的命令在打印机上执行。**
- **窗口 B：树莓派桌面中新开的本机 Terminal。这个窗口直接在树莓派上执行命令，全程不用它 SSH 到打印机。**

SSH 会改变命令的执行位置。在树莓派屏幕上看到的终端，也可能正在控制打印机。
本指南统一由笔记本 SSH 管理打印机，树莓派使用本机终端，以便区分。

## 1. 接线和准备：三台设备

1. 将树莓派 RJ45 网口与打印机 RJ45 网口用一根网线连接。
2. 树莓派和打印机各自正常供电，保持打印机空闲。
3. 笔记本和打印机暂时保留现有 Wi-Fi，继续使用当前能打开打印机网页的地址。
   树莓派如需下载安装软件，也可保留 Wi-Fi。
4. 本指南用的 `192.168.50.0/24` 应与两台设备已有网络分开。
   已有 Wi-Fi/其他接口若占用这个网段，应先选另一个未使用的私有网段，并统一替换两端地址。

此次设置保留打印机内部 MCU 接线、现有 Klipper/Moonraker 安装和固件。
基底高度仍可保持未知；本指南结束于只读连接检查。

## 2. 笔记本：备份，然后 SSH 登录打印机

先用笔记本浏览器打开打印机网页，在配置页面下载备份 `printer.cfg`、
被它引用的其他配置文件，以及 `moonraker.conf`。

打开 **Windows PowerShell（窗口 A）**，执行：

```powershell
$printerWifiIp = Read-Host "输入打印机当前网页的 IP 地址（只填 IP）"
ssh "sovol@$printerWifiIp"
```

输入的是当前能访问的无线 IP，不是尚待配置的 `192.168.50.2`。
原厂未修改时，用户名和密码均为 `sovol`；改过则使用自己的账户。
首次连接确认目标后按提示接受主机身份；输入密码时没有字符或星号显示。
[Sovol 原厂 SSH 登录说明](https://github.com/Sovol3d/SV08)。

登录完成后，窗口 A 就是**打印机终端**。你提供的 SSH 截图提示符是
`sovol@SPI-XI`；截图中 `eth0` 的连接配置名是 `Wired connection 1`。

## 3. 打印机：设置有线地址 192.168.50.2

**执行位置：窗口 A，即笔记本 SSH 登录后的打印机终端。**

先核对：

```bash
hostname
ip route
nmcli -f NAME,TYPE,DEVICE connection show
```

下面使用截图中已识别的打印机有线配置 `Wired connection 1`。
如果实际名称已变，替换引号中的名称，选择绑定有线网口的 `ethernet` 配置。

```bash
sudo nmcli connection modify "Wired connection 1" \
  ipv4.method manual \
  ipv4.addresses 192.168.50.2/24 \
  ipv4.gateway "" \
  ipv4.dns "" \
  ipv4.never-default yes \
  connection.autoconnect yes
sudo nmcli connection up "Wired connection 1"
ip -4 -br address show eth0
```

目标结果：打印机 `eth0` 显示 `192.168.50.2/24`。
这些命令设置的是有线接口；当前 Wi-Fi 仍可用于管理和上网。
如果之前误把该接口设成了 `.1`，上述命令也会将该配置改成正确的 `.2`。
[NetworkManager 配置参数](https://networkmanager.dev/docs/api/latest/nm-settings-nmcli.html)。

仍在**窗口 A / 打印机**检查服务：

```bash
systemctl is-active moonraker
systemctl is-active klipper
```

两项应返回 `active`。若服务名称或安装不同，用实际安装的服务检查。
服务已正常运行就保持原样。

## 4. 树莓派：在本机设置有线地址 192.168.50.1

**执行位置：树莓派桌面中新开的 Terminal（窗口 B）。这里不执行 SSH。**
如果原来的树莓派窗口还停留在打印机的 SSH 会话，关闭那个窗口，另开一个本机 Terminal。

需要确认设备时，可在窗口 B 运行：

```bash
tr '\0' '\n' < /proc/device-tree/model
```

应显示 Raspberry Pi 的型号。然后检查树莓派本机的网络：

```bash
ip route
nmcli device status
```

在使用 NetworkManager 的树莓派系统上，运行：

```bash
sudo nmtui
```

按以下顺序设置：

1. 选择 **Edit a connection**。
2. 选择树莓派自己的 **Ethernet / 有线连接**。如果没有，则添加 Ethernet 连接并选择实际有线接口。
3. 将 **IPv4 CONFIGURATION** 改成 **Manual**。
4. 地址填写 **`192.168.50.1/24`**。
5. Gateway / DNS 留空，勾选 **Never use this network for default route**。
6. 勾选 **Automatically connect**，保存。
7. 返回 **Activate a connection**，将该有线连接重新激活，然后退出。

树莓派的配置名称以它自己的列表为准，不需要与打印机相同。
[NetworkManager nmtui 文档](https://networkmanager.dev/docs/api/latest/nmtui.html)。

仍在**窗口 B / 树莓派本机**检查：

```bash
ip -4 -br address
ip route get 192.168.50.2
ping -c 5 192.168.50.2
```

目标结果：树莓派有线接口显示 `192.168.50.1/24`；到打印机的路由走有线接口，
源地址为 `192.168.50.1`；`ping` 收到回复。
两端都不在这条直连链路上填写网关或 DNS。

如果树莓派系统没有使用 NetworkManager，不要强行安装另一套网络管理器覆盖现有配置；
应按当前系统的网络管理方式设置表中同样的静态地址。

## 5. 树莓派：检查 Moonraker

**执行位置：窗口 B / 树莓派本机。**

```bash
curl --fail --show-error --silent --connect-timeout 3 \
  http://192.168.50.2:7125/server/info
curl --fail --show-error --silent --connect-timeout 3 \
  http://192.168.50.2:7125/printer/info
```

第一项检查 `result.klippy_connected` 为 `true`、`result.klippy_state` 为 `ready`；
第二项检查 `result.state` 为 `ready`。这些请求只读取状态，不回零、不移动、不加热。

Sovol 发布的安装已有以下 Moonraker 设置，通常无需改动：

```ini
[server]
host: 0.0.0.0
port: 7125
klippy_uds_address: /home/sovol/printer_data/comms/klippy.sock
```

来源：[Sovol SV08 Moonraker 配置](https://raw.githubusercontent.com/Sovol3d/SV08/main/home/sovol/printer_data/config/moonraker.conf)。
实际安装值以加载的配置为准。`klippy_uds_address` 始终是打印机 H616 的本机路径。
需要修改时，在笔记本的打印机配置网页编辑已有 `[server]` 字段并保留备份；
修改后在**窗口 A / 打印机**执行 `sudo systemctl restart moonraker`。

若返回 401/403，使用已有 API key。具体读取、设置环境变量与请求头的方法见
[Moonraker 指南第 4 节](MOONRAKER_ETHERNET_SETUP.md#4-从树莓派检查-http-api-和鉴权)。
其中“在 H616 运行”对应窗口 A，“在树莓派运行”对应窗口 B。

## 6. 笔记本传项目，树莓派安装并检查控制器

**笔记本负责传文件：**把本次准备的 `pi_controller_setup_20261015_02.zip`
通过 U 盘或现有文件传输方式复制到树莓派。包内包含当前控制器源代码、所需本地模块、
示例配置和本指南，不包含 Windows 虚拟环境、实验数据或模型权重。

**树莓派负责安装：**在窗口 B 中进入 ZIP 所在目录，然后执行：

```bash
unzip pi_controller_setup_20261015_02.zip
cd Dynamic_Surface_Printer
python3 -m venv .venv-controller
source .venv-controller/bin/activate
python -m pip install -r requirements-controller.txt
cp -n configs/controller.example.json configs/controller.json
nano configs/controller.json
```

如果树莓派已有当前项目，也可以直接在其根目录执行创建环境及后面的步骤。
若缺少 `unzip` 或 Python venv 支持，在 Raspberry Pi OS / Debian 上可安装
对应的 `unzip`、`python3-venv` 系统包。

修改已有 `hardware` 对象中的这两个字段：

```json
"backend": "moonraker",
"moonraker_url": "http://192.168.50.2:7125"
```

以上是字段片段，不是完整 JSON 文件。其他字段继续保留。
`substrate_height_mm` 保持 `null`，各验证标志保留实际状态。
在 nano 中按 Ctrl+O、回车保存，再按 Ctrl+X 退出。

仍在**窗口 B / 树莓派本机项目根目录**运行：

```bash
python -m controller discover \
  --config configs/controller.json \
  --output output/sv08_discovery_01
```

每次使用新的输出目录。成功后会保存 `output/sv08_discovery_01/discovery.json`。
这个命令只读取版本、状态和可用对象。

该部署包的默认预测器是 NumPy 速度基线。只读发现不需要相机驱动或模型权重。
后续 D405 采集需准备树莓派上的 RealSense 运行时和 Python 绑定；ConvLSTM 预测另需
匹配的 PyTorch 环境、checkpoint 与几何配置，并测量推理耗时。
当前控制器支持虚拟动态控制和受限制的静态冷态试验，实时动态表面实机打印仍有验证门槛。

采集和表面预测的具体安装检查、基线及 ConvLSTM 命令见
[树莓派 D405 采集与预测指南](../raspberry_pi/PI_PREDICTION_SETUP.md)。

## 完成的判据

| 在哪台设备检查 | 成功结果 |
|---|---|
| 打印机 H616，窗口 A | 有线地址 `.2`，Moonraker 和 Klipper 服务正常 |
| 树莓派，窗口 B | 有线地址 `.1`，到 `.2` 的路由走有线接口 |
| 树莓派，窗口 B | Moonraker 请求成功、Klipper 状态 `ready` |
| 树莓派，窗口 B | 控制器生成 `discovery.json` |

持久网络设置完成后，在方便的空闲时段重启两台设备，再确认地址和只读请求仍有效。
SSH 管理窗口可以在配置结束后关闭，Moonraker 和 Klipper 服务会继续运行。
控制器后续运行在哪台设备，其进程就必须在那台设备上保持运行；本方案是树莓派。

更细的授权、故障排查和延迟测量见 [Moonraker 参考指南](MOONRAKER_ETHERNET_SETUP.md)。

# 树莓派与 SV08 通过一根网线连接 Moonraker

**按设备操作的完整流程请先看 [笔记本、树莓派、打印机设置指南](SETUP_LAPTOP_PI_PRINTER.md)。**
该指南固定由笔记本 SSH 管理打印机、树莓派使用本机终端；本文件保留网络和鉴权参考。

更新：2026 年 10 月 14 日。当前选择是树莓派 5 负责采集、预测和路径规划，
通过网线直连 SV08 的 H616；Moonraker 和 Klipper 继续运行在 H616 上。
本指南是操作步骤，尚未在实际打印机上执行。

```text
D405 ─ USB ─ 树莓派 5：预测 + controller
                    │ 机身网口，一根网线
                    │ 有线接口固定 IP
             SV08 H616：Moonraker → Klipper
                                      │ 原有内部连接
                               主板 MCU + 工具头 MCU
```

Moonraker 在这条本地连接上接收请求，Klipper 负责运动规划和 MCU 通信。
控制链路不需要互联网。安装软件时可临时使用 Wi-Fi；模型训练仍可留在工作站。
这个方案使用打印机现有固件，不要求先迁移 Mainline 或重新接 MCU 的 USB。

## 1. 先准备进入 H616 的方法

### 暂时没有网线时可以先做什么

现在可以先完成软件准备和 Wi-Fi 只读联调，之后再切换到有线地址。
不需要先确定基底高度，也不需要先修改 H616 的有线网络配置。

1. 如果打印机已接入 Wi-Fi，让电脑或树莓派也接入可相互访问的同一局域网。
   用打印机当前的无线 IP 打开原有网页界面，下载备份 `printer.cfg`、
   被它包含的配置文件，以及 `moonraker.conf`。
2. 在浏览器访问 `http://实际无线IP:7125/printer/info`。
   返回 JSON 后查看 `result.state`，目标为 `ready`。
   若提示鉴权，使用第 4 节的已有 API key 方法。
   此时使用的是实际无线 IP，不是后面为网线直连选择的 `192.168.50.2`。
3. 将 `configs/controller.example.json` 另存为 `configs/controller.wifi.json`，
   在现有 `hardware` 对象中保留 `backend: "moonraker"`，将 `moonraker_url`
   设为 `http://实际无线IP:7125`。基底高度保持 `null`，验证标志保持实际状态。
4. 在当前 Windows 电脑的项目根目录和已有环境中运行只读检查：

```powershell
.\.venv-tds\Scripts\python.exe -m controller discover `
  --config configs/controller.wifi.json `
  --output output/sv08_wifi_discovery_01
```

它保存状态、版本及可用对象，不发送运动或加热命令。每次使用新输出目录。
如果树莓派尚未准备好，可以先用这台 Windows 电脑完成该检查；树莓派准备好后，
按第 5 节部署项目，用 Linux 环境执行相同的 `discover` 命令。

5. 即使没有打印机网络，也可先验证完整的虚拟控制流程：

```powershell
.\.venv-tds\Scripts\python.exe -m controller simulate `
  --scenario moving-plane --virtual-extrusion `
  --output output/controller_pre_cable_01
```

模拟不会连接打印机。它使用仅用于该次模拟的虚拟基底高度，不会改写硬件配置。
有网线后再设置两端固定 IP，并在有线配置中改用 H616 的有线地址。
Wi-Fi 检查结果证明接口可以通信，不能替代有线连接的时序测试。

先通过打印机现有 Wi-Fi 地址使用 SSH，或者使用已可用的本机终端进入 H616。
使用自己现有的账户和密码。树莓派也需要终端访问。
更改有线接口时，优先从 Wi-Fi 或本机终端操作；经该有线接口建立的 SSH 会断开。

如果目前完全无法进入 H616，先恢复一种登录方式。只插网线不会自动给两端分配
本指南中的固定 IP。不要为此重刷固件或更改 MCU 配置。

将树莓派的 RJ45 网口与打印机的 RJ45 网口用普通以太网线连接。

## 2. 设置两端固定 IP

以下地址是本指南选择的示例，不是 SV08 出厂地址。
先用两端的 `ip -br address` 和 `ip route` 确认现有网络没有占用
`192.168.50.0/24`；如与 Wi-Fi 等已有网段冲突，换一个未使用的私有网段，
同时修改下面两端地址与控制器 URL。

| 设置项 | 树莓派有线接口 | H616 有线接口 |
|---|---|---|
| IPv4 地址/前缀 | `192.168.50.1/24` | `192.168.50.2/24` |
| 子网掩码 | `255.255.255.0` | `255.255.255.0` |
| 网关 | 留空 | 留空 |
| DNS | 留空 | 留空 |
| 默认路由 | 不使用这个接口作为默认路由 | 不使用这个接口作为默认路由 |

网线只承载两台设备间的通信；Wi-Fi 如需上网，可继续保留自己的默认路由。

### 使用 NetworkManager 的系统：持久设置

分别在树莓派和 H616 的终端检查：

```bash
ip -br link
systemctl is-active NetworkManager
nmcli device status
nmcli -f NAME,TYPE,DEVICE connection show
```

只有在 NetworkManager 正在管理该有线接口时，使用下面的方法。
识别 `ethernet` 接口及其连接配置名称；不要修改 `wifi` 连接。
接口可能叫 `eth0`、`end0` 或其他名称，以本机输出为准。

运行：

```bash
sudo nmtui
```

在 **Edit a connection** 中选中对应的有线连接，将 **IPv4 CONFIGURATION**
设为 **Manual**，填入上表中该设备的地址，网关和 DNS 留空，勾选
**Never use this network for default route** 和 **Automatically connect**。
保存后在 **Activate a connection** 中重新激活该有线连接。
如果尚无有线配置，创建一个 Ethernet 配置，并绑定上面查到的实际有线接口。
这些操作分别在两台设备执行，只有地址末尾的 `.1` / `.2` 不同。
[NetworkManager nmtui 文档](https://networkmanager.dev/docs/api/latest/nmtui.html)。

没有 `nmtui`、但 `nmcli` 可用时，可以修改现有有线配置。以下 `ETH_PROFILE`
必须换成上面查到的实际连接名称。在树莓派运行：

```bash
ETH_PROFILE="实际有线连接名称"
sudo nmcli connection modify "$ETH_PROFILE" \
  ipv4.method manual ipv4.addresses 192.168.50.1/24 \
  ipv4.gateway "" ipv4.dns "" ipv4.never-default yes \
  connection.autoconnect yes
sudo nmcli connection up "$ETH_PROFILE"
```

在 H616 上执行同样的命令，将地址改为 `192.168.50.2/24`，并使用 H616
自己的有线连接名称。不要照抄树莓派的连接名称。
[NetworkManager nmcli 文档](https://networkmanager.dev/docs/api/latest/nmcli.html)。

### H616 没有使用 NetworkManager：先做临时连接验证

不要另装网络管理器来覆盖现有配置。用 `ip -br link` 查出实际有线接口，
在 H616 上给该接口添加临时地址：

```bash
ETH_IF="实际有线接口名"
sudo ip link set dev "$ETH_IF" up
sudo ip address add 192.168.50.2/24 dev "$ETH_IF"
```

这不会清除现有地址，通常在重启或网络服务重新配置后失效。
树莓派仍按前面的持久方式设置 `.1`，然后可以完成下面的 Moonraker 连通测试。
临时方式不作为长期运行配置。持久化必须按 H616 实际使用的网络管理器设置；
用下面命令区分服务，并保留现有配置，不能混写多套网络配置：

```bash
systemctl is-active NetworkManager systemd-networkd networking
```

### 确认网线确实承载这条连接

在树莓派运行：

```bash
ip -br address
ip route get 192.168.50.2
ping -c 5 192.168.50.2
```

路由输出应指向实际有线接口，并显示 `src 192.168.50.1`。
H616 的 `ip -br address` 应显示有线地址 `192.168.50.2/24`。
使用持久配置时，重启两台设备后再次检查地址和路由。

## 3. 检查 H616 上已有的 Moonraker

Sovol 发布的 SV08 配置已有以下设置，通常只需要核对，无需重装：

```ini
[server]
host: 0.0.0.0
port: 7125
klippy_uds_address: /home/sovol/printer_data/comms/klippy.sock
```

来源：[Sovol SV08 moonraker.conf](https://raw.githubusercontent.com/Sovol3d/SV08/main/home/sovol/printer_data/config/moonraker.conf)。
`host` 是服务器监听地址；树莓派连接时使用 `192.168.50.2`。
`klippy_uds_address` 是 **H616 本机**的 socket 路径，保持安装中实际有效的值，
不能改成树莓派路径或 IP。

在 H616 上运行：

```bash
systemctl status moonraker --no-pager
systemctl status klipper --no-pager
systemctl cat moonraker
```

用现有 Mainsail/Fluidd 的配置编辑器，或者根据服务启动参数找到实际加载的
`moonraker.conf`。Sovol 发布的布局是
`/home/sovol/printer_data/config/moonraker.conf`，实际安装若不同，以服务参数为准。

如果端口和监听地址已经正确，就继续下一步。如果需要修改，先下载配置备份，
只编辑现有 `[server]` 中的对应字段，保留其他内容。
同一个文件不能重复添加 `[server]` 或 `[authorization]` 段。
[Moonraker 配置说明](https://moonraker.readthedocs.io/en/latest/configuration/)。

仅在修改后、打印机空闲时重启 Moonraker：

```bash
sudo systemctl restart moonraker
```

## 4. 从树莓派检查 HTTP API 和鉴权

在树莓派运行下面两个只读请求：

```bash
curl --fail --show-error --silent --connect-timeout 3 \
  http://192.168.50.2:7125/server/info
curl --fail --show-error --silent --connect-timeout 3 \
  http://192.168.50.2:7125/printer/info
```

检查 JSON：`result.klippy_connected` 应为 `true`，
`result.klippy_state` 和打印机的 `result.state` 应为 `ready`。
收到 JSON 说明 API 可访问；Klipper 是否就绪还要看状态字段。
发现连接不要求先回零、移动或加热。

若返回 401/403，使用已有 API key。可以从已经获授权的客户端读取
`GET /access/api_key`；Sovol 原有配置中本机回环地址受信任时，也可在 H616
终端运行以下命令查看现有 key：

```bash
curl --fail --show-error --silent http://127.0.0.1:7125/access/api_key
```

如果该请求也要求登录，先使用已有 Moonraker 登录方式取得访问权限，再读取 key。
在树莓派的 Bash 终端安全输入 JSON `result` 中的 key 字符串：

```bash
read -rsp "Moonraker API key: " MOONRAKER_API_KEY
export MOONRAKER_API_KEY
curl --fail --show-error --silent --connect-timeout 3 \
  -H "X-Api-Key: $MOONRAKER_API_KEY" \
  http://192.168.50.2:7125/printer/info
```

Moonraker 的 HTTP API 使用 `X-Api-Key` 请求头；本项目也从该环境变量读取 key。
保持现有授权策略即可，Python 客户端无需修改浏览器的 `cors_domains`。
不要为了连通而开放所有客户端或公网端口。
[Moonraker 鉴权说明](https://moonraker.readthedocs.io/en/latest/external_api/authorization/)。

## 5. 配置并检查本项目控制器

把包含本次 `controller/` 代码的完整项目复制或同步到树莓派。
在树莓派的项目根目录创建 Linux Python 环境：

```bash
python3 -m venv .venv-controller
source .venv-controller/bin/activate
python -m pip install -r requirements-controller.txt
cp -n configs/controller.example.json configs/controller.json
```

修改 `configs/controller.json` 的现有 `hardware` 对象内这些字段，保留其他字段：

```json
"backend": "moonraker",
"klipper_socket_path": null,
"moonraker_url": "http://192.168.50.2:7125",
"api_key_env": "MOONRAKER_API_KEY"
```

这是对象内部的字段片段，不是完整配置文件。`substrate_height_mm` 保持 `null`，
各项硬件验证标志也继续保留实际状态。只读连接检查不需要安装相机驱动或模型。
实际 D405 采集还需要单独配置 Pi 上的 RealSense 运行时和 Python 绑定；
ConvLSTM 推理另需预测环境及匹配的 checkpoint。

运行：

```bash
python -m controller discover \
  --config configs/controller.json \
  --output output/sv08_discovery_01
```

每次使用新的输出目录。它会保存 `discovery.json`，包含版本、状态、可用对象及一次
状态查询往返时间；不发送运动或加热指令，也不自动修改打印机配置。

连接完成的判据是：有线地址和路由正确；API 可访问；Klipper 报告 `ready`；
控制器成功保存发现报告。随后再按 [硬件指南](HARDWARE_SETUP.md) 完成坐标标定、
实际基底高度和冷态运动设置。当前实现支持虚拟动态控制和受限制的静态冷态试验；
接通 Moonraker 本身不会开启动态表面实机打印。

## 6. 排查与延迟检查

| 现象 | 检查顺序 |
|---|---|
| 网线接上但无地址 | 两端必须各自配置固定 IP；直接连接没有路由器提供 DHCP |
| `ping` 不通 | 网口连接状态、两端地址/掩码、是否有冲突网段、是否走有线接口；也可能有 ICMP 策略限制 |
| `ping` 通，HTTP 失败 | H616 上 Moonraker 服务、实际监听地址/端口，以及是否有防火墙拦截 |
| HTTP 401/403 | 使用现有 API key 或已有授权方式 |
| `klippy_connected: false` | H616 上 Klipper 服务与实际 socket 路径 |
| API 通，Klipper 为 `error` / `shutdown` | 查看 Klipper 的 `state_message` 和日志，处理打印机本机错误 |
| 重启后失联 | 是否只加了临时 IP，持久配置是否绑定到正确有线接口 |

在树莓派运行 `ping -c 30 192.168.50.2` 可初步看丢包和往返波动。
HTTP 总耗时可以用下面的只读请求观察（需要鉴权时加同样的 `X-Api-Key` 请求头）：

```bash
curl --fail --show-error --silent --output /dev/null \
  --connect-timeout 3 --write-out 'HTTP total: %{time_total}s\n' \
  http://192.168.50.2:7125/printer/info
```

这些值包含不同层面的等待时间。它们和 `discovery.json` 的 `round_trip_s`
都不是喷嘴实际到达时间，也不能给出运动队列延迟。
后续动态跟踪应测量完整的采集、预测、提交、排队到执行链路，再确定预测提前量。
不要用未经实测的固定毫秒数替代运动时序标定。

# Raspberry Pi 运行指南

版本：2026-10-26。对应压缩包：`dynamic_surface_printer_pi_20261026.zip`。

这个包包含完整 Python 源码、树莓派安装与启动脚本、当前静态打印配置，以及本次增加的故障状态记录和只读诊断工具。默认安装即可进行软件自检、读取打印机状态、收集故障资料和生成路径预览；接入 D405 时再安装相机依赖。

**先完成第 1–5 步。下面的自检、诊断、状态和预览命令不会移动打印机、加热或启动打印。** 本次更新补齐故障发生前的状态记录；尚未确认上次停止的根因，也没有解除原有路径偏差检查或急停行为。不要为了补采数据重新运行上次的热打印任务。

## 1. 在树莓派解压，保留旧目录

以下命令全部在 **Raspberry Pi 本机 Terminal** 执行，也可以在已经 SSH 登录树莓派的终端执行。不要在打印机 H616 的 SSH 会话中执行安装。

将 ZIP 复制到树莓派的 `~/Downloads` 后运行：

```bash
cd ~/Downloads
unzip dynamic_surface_printer_pi_20261026.zip
cd Dynamic_Surface_Printer_Pi_20261026
ls setup_pi.sh run_pi.sh controller/stationary.py
```

如果文件放在其他位置，将第一行改为实际目录。`ls` 应能找到三个文件；如果没有找到，检查是否进入了额外一层解压目录。

使用新目录保存这个版本。保留旧程序目录、旧配置和失败任务的整个输出文件夹，后续诊断需要读取它们。不要用新包覆盖旧的 `output/`。

目标环境为 Raspberry Pi 5 的 **64 位 Linux（aarch64）**。现有项目记录的设备是 Debian 13.2、系统 Python 3.13.5。本包使用独立 Python 3.12 环境，避免系统 Python 与相机 wheel 不匹配。首次安装需要互联网；打印机的直连网线本身不提供互联网，可保留树莓派的 Wi-Fi。

## 2. 一次性安装

在解压后的项目目录执行：

```bash
bash setup_pi.sh
```

安装脚本通过系统包管理器准备 `python3-venv` 和证书等前置依赖，再准备独立 Python 3.12（已安装则复用，否则由 uv 下载），在项目内创建 `.venv-pi312`，从随包的 `wheelhouse/` 安装控制器所需的 NumPy 和 httpx。遇到 `sudo` 提示时输入树莓派账户密码；密码输入通常没有字符显示。脚本不会安装或重启打印机的 Klipper/Moonraker 服务。

安装完成后保持项目目录位置不变；虚拟环境包含绝对路径。若需要搬到另一个目录，在新目录重新安装。脚本可以重复运行，不会重写你的试验配置或历史输出。

后续使用启动脚本即可，不需要手动 `source` 激活环境：

```bash
bash run_pi.sh
bash run_pi.sh doctor
bash run_pi.sh self-test
```

第一条显示可用命令；`doctor` 检查本机解释器和依赖；`self-test` 检查软件链路。自检成功说明这个环境能够运行本包的软件，不代表已经验证相机、打印机通信或真实运动。

如果已有能运行本项目的 Linux 环境，可以明确指定其 Python：

```bash
PI_PYTHON=/absolute/path/to/your/venv/bin/python bash run_pi.sh doctor
```

将示例绝对路径替换为实际解释器路径。启动脚本默认使用本包自己的 `.venv-pi312`，不要复制 Windows 的 `.venv` 到树莓派。

## 3. 收集上次失败任务的资料

找到旧项目中发生故障的输出文件夹。它通常包含 `execution.json`、`job.json` 和 `execution_events.jsonl`。传给 `--session` 的是**包含这些文件的目录**，不是 ZIP 文件、单个 JSON 文件或整个旧项目目录。

在新包的项目目录运行：

```bash
bash run_pi.sh diagnose \
  --session /absolute/path/to/old/project/output/failed_run
```

把 `/absolute/path/to/old/project/output/failed_run` 替换为树莓派上的真实路径。路径含空格时加双引号。新旧项目目录不同时，使用绝对路径最清楚；启动脚本中的相对路径按本包项目根目录解释。

该命令读取指定任务的已有记录，并通过 HTTP GET 读取打印机的当前状态和受长度限制的 Klipper 日志末尾。它不执行恢复、重启、回零、加热、上传或打印。终端会打印新诊断目录的位置，结果保存在本包的 `output/` 下，每次生成新的目录。

暂时无法连接打印机时，仍可以先收集本地记录：

```bash
bash run_pi.sh diagnose --offline \
  --session /absolute/path/to/old/project/output/failed_run
```

如果旧任务缺少 `last_status_snapshot.json`，这是旧版未保存该文件的结果。新版本不能追溯补出当时的实际触发样本；现在查询到的坐标是**事后状态**。没有旧目录时也可以不写 `--session` 运行诊断，但新解压的包通常没有可以自动找到的历史失败记录。

## 4. 检查打印机连接

当前配置文件是：

```text
profiles/stationary/controller.json
```

其中 `hardware.moonraker_url` 默认是：

```text
http://192.168.50.2:7125
```

这是现有方案中打印机 H616 的有线地址，树莓派对应地址为 `192.168.50.1/24`。如果这条链路尚未配置，按包内 `docs/printer_control/SETUP_LAPTOP_PI_PRINTER.md` 的网络部分操作；其中旧包名和旧安装环境步骤由本指南替代。

如果实际使用另一个地址，编辑 `profiles/stationary/controller.json` 中的 `hardware.moonraker_url`。旧记录中的 `100.65.32.176` 只有在当前树莓派实际能访问时才可使用，不要仅因为曾出现在旧文件中就照填。

连接准备好后执行：

```bash
bash run_pi.sh status
```

查看返回的固件状态、坐标、归零标志和温度，以及输出目录内的完整记录。`ready` 表示当前固件状态；它不能单独证明上次故障后的物理位置或坐标参考仍然可靠。

诊断命令的地址可单独覆盖：

```bash
bash run_pi.sh diagnose --url http://192.168.50.2:7125 \
  --session /absolute/path/to/old/project/output/failed_run
```

如果 Moonraker 返回 401/403，可在当前终端设置已有 API key，再运行状态或诊断命令：

```bash
read -rsp 'Moonraker API key: ' MOONRAKER_API_KEY; echo
export MOONRAKER_API_KEY
bash run_pi.sh status
```

这里输入已有密钥；不要把密钥写入配置、截图或发给他人的日志。

## 5. 生成静态路径预览

```bash
bash run_pi.sh preview
```

该命令读取 `profiles/stationary/controller.json` 和 `profiles/stationary/plan.json`，生成本地 G-code 预览及 `job.json`，不连接打印机执行任务。结果目录由终端给出。

随包的配置和计划复制自此前 2026-10-25 的静态试验，供核对原任务：要求起始位置 X190/Y90/Z170 mm；预览线为 X170–190、Y83、Z150 mm，旅行高度 Z170 mm，喷嘴设定 220°C。它们是原计划参数，不是当前位置或已测表面间隙。连接地址改为 `.2`；实物就绪标志均保留为 `false`。先收集故障资料，查清触发偏差的样本，再决定后续实体试验。

`run_pi.sh` 没有执行打印的快捷命令。底层控制器仍保留原有执行接口，但本指南的流程到诊断、状态读取和预览为止，不恢复失败任务。

## 6. 可选：在树莓派连接 D405

如果当前只处理打印停止故障，可以跳过本节。需要相机采集或表面预测时，在同一个新包目录执行：

```bash
bash setup_pi.sh --camera
```

相机安装使用 Python 3.12 和 `pyrealsense2==2.59.1.11053`。已从官方 PyPI 下载对应 ARM64/Python 3.12 wheel 和控制器依赖，放入包内 `wheelhouse/`；安装这些 Python 依赖时不再联网。首次准备系统包、Python 运行时以及下载相机 udev 规则仍需要互联网，因此这不是完全离线安装包。参考：[RealSense 发布文件](https://pypi.org/project/pyrealsense2/2.59.1.11053/#files)、[uv Python 管理说明](https://docs.astral.sh/uv/guides/install-python/)。

`--camera` 还会安装 USB 相关系统依赖和 RealSense udev 规则，已有规则会先备份。完成后重新插拔 D405，关闭其他占用相机的软件，然后依次运行：

```bash
bash run_pi.sh doctor --camera
bash run_pi.sh capture --frames 90
bash run_pi.sh baseline --frames 300
```

采集结束后再运行预测，避免两个程序同时打开相机。基线使用 NumPy 计算，不需要 PyTorch 或训练权重。默认相机配置是 `configs/d405_camera_frame.json`，输出的是相机坐标深度；单位阵变换并未将它标定为打印机 Z，不能直接作为喷嘴高度。

相机枚举为空或 USB 权限报错时，按 `docs/raspberry_pi/PI_REALSENSE_INSTALL.md` 的 udev 规则部分排查，再重新插拔相机。该旧文档中的手动创建环境步骤由本包安装脚本替代。软件依赖已安装、设备能枚举、真实流能持续采集是三个不同的检查阶段。

本包默认没有训练好的 ConvLSTM 权重。已有相机标定也只有在相机、夹具和坐标参考保持一致时才可复用；本次诊断不自动认定原标定已经满足实体打印精度要求。

## 7. 每次打开新终端后的操作

```bash
cd ~/Downloads/Dynamic_Surface_Printer_Pi_20261026
bash run_pi.sh doctor
```

然后选择本次需要的 `diagnose`、`status`、`preview`、`capture` 或 `baseline` 命令。换了存放位置就调整 `cd`。终端始终显示每次任务的输出路径，保留这些结果用于核对。

## 输出文件与排错

| 情况 | 检查方法 |
|---|---|
| `No such file or directory` | 确认当前目录直接包含 `setup_pi.sh`、`run_pi.sh` 和 `controller/`。 |
| `No module named ...` | 先完成安装；用 `doctor` 确认解释器。不要把系统 Python、旧环境和新环境混用。 |
| 下载依赖失败 | 检查树莓派互联网、DNS 和系统时间；修复网络后重新运行安装脚本。 |
| `No matching distribution` | 检查 `uname -m` 为 `aarch64`，以及实际解释器为 Python 3.12；不要强装 x86 或其他 Python 版本的 wheel。 |
| Moonraker 超时或连接拒绝 | 检查网线、两端 IP、端口 7125 和 H616 上已有服务。先解决通信，不运行打印来测试通信。 |
| Moonraker 401/403 | 使用已有 API key；检查当前终端的环境变量。 |
| 找不到历史故障 | 给 `diagnose --session` 指定旧失败任务目录的绝对路径。 |
| `historical_triggering_pose_available` 为 `false` | 旧版没有保存触发样本；当前状态不能替代该历史样本。 |
| 相机打不开 | 关闭其他采集程序，检查数据线、供电、USB 权限和流模式。 |

诊断目录内主要检查 `diagnostic.json`、可用时的 `execution_events_tail.txt` 和 `klippy_tail.log`。新版底层静态任务在检查状态前保存 `last_status_snapshot.json`，事件记录也包含当时报告的 XYZ、最近路径位置和距离，因此后续获准的实体试验若再失败，可以保留实际被拒绝的样本。

ZIP 包含源码、示例/试验配置、安装启动脚本、说明和 ARM64 Python 依赖 wheel，不包含 Windows 虚拟环境、历史大型实验数据或模型权重。软件检查在开发机完成；本包尚未在你的 Raspberry Pi、D405 和 SV08 实机组合上执行验证。树莓派上的自检、只读诊断和相机短采集是部署后的验证步骤。

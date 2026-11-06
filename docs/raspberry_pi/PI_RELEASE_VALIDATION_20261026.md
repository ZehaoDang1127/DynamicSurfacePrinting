# 部署包验证记录 — 2026-10-26

验证对象是本次 ZIP 解压后的独立项目目录，运行时不依赖原项目的源码路径。
开发机为 Windows，解释器为 Python 3.11.9，NumPy 2.4.6、httpx 0.28.1。
这些是软件验证结果，不代表 Raspberry Pi、D405 或 SV08 的实机验证。

| 检查 | 结果 |
|---|---|
| `test_pi_launcher.py` | 5 项通过；包括错误历史目录拒绝和预览无法启用执行 |
| `test_stationary_print.py` | 10 项通过；模拟打印机验证故障样本在急停前保存、启动不重复及原有检查 |
| `self-test` | 12 帧合成采集、8 个预测、7 个评分窗口；虚拟路径完成；静态预览成功 |
| 随包静态计划的 `preview` | 20 mm 线段预览成功，`uploaded=false`，未连接打印机 |
| `diagnose --offline` | 成功生成独立诊断目录；新包无历史运行时明确记录缺少历史故障 |
| Bash 语法 | `setup_pi.sh` 与 `run_pi.sh` 通过 `bash -n` |
| 压缩包 | CRC、文件 SHA256 清单、脚本 LF 换行均通过检查 |
| ARM64 依赖 | pip 按 CPython 3.12 / manylinux2014_aarch64 成功解析并下载 NumPy 2.2.6、httpx 0.28.1、RealSense 2.59.1.11053 及依赖；wheel 随包提供 |

未执行：Pi 上的 apt/uv 安装、ARM64 本机导入和性能测试、真实 D405 采集、真实 Moonraker 通信、打印机运动或加热。
本次仅发布诊断记录更新及部署工具；先前路径偏差停止的根因仍未确认。

部署后按中文指南运行 `doctor`、`self-test` 和只读诊断。需要相机时再运行短采集。

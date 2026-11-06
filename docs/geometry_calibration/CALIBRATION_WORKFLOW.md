# 四点外参标定与喷嘴定位

这两个程序可以在四点照片拍摄前先测试。工作目录为项目根目录
`D:\HAIM_Lab\Dynamic_Surface_Printer`。真实 XYZ 和图像坐标缺失时，程序会报错，不会补入示例值。

- `python -m CameraCalibration.transform`：点选标记、PnP、正反变换、重投影。
- `python -m controller.nozzle`：按打印机 XYZ 定位喷嘴，先抬升再横移，默认只输出预览。

喷嘴定位目前是**冷态、不挤出**的定位和空跑。它用于验证四个标定点的位置或其他已知 XYZ；四点实际挤出打印需要另行确定材料、温度、挤出量和喷嘴间隙。
现有动态控制器的打印门槛没有修改。

## 1. 安装与离线演示

```powershell
Set-Location D:\HAIM_Lab\Dynamic_Surface_Printer
& .\.venv\Scripts\python.exe -m pip install -r requirements-calibration.txt

# 使用当前内参和已知倾斜姿态生成合成像素，验证 PnP 和逆变换。
& .\.venv\Scripts\python.exe -m CameraCalibration.transform demo --output output/transform_demo

# 使用虚构的静止夹具和四点坐标预览喷嘴路径；不连接打印机。
& .\.venv\Scripts\python.exe -m controller.nozzle demo --output output/nozzle_demo
```

每个输出路径必须是新的。演示数据和演示 G-code 都标有 `synthetic_demo` 或 `SYNTHETIC DEMO`，不能作为实测标定或硬件任务。

## 2. 四点数据准备

当前默认读取 `CameraCalibration/results/calibration.json`，对应 **848×480 彩色图像**。
也可以用 `--intrinsics` 指向相同 JSON 格式的其他内参文件。

```powershell
& .\.venv\Scripts\python.exe -m CameraCalibration.transform init --output output/calibration_points.json
```

在生成的文件里为 P1–P4 填入实际 `printer_xyz_mm: [X,Y,Z]`。
P1–P4 是对应关系标签，不能按图像上下左右猜测编号。Z 是相机看到的标记特征高度，而非自动等于打印标记时的喷嘴 Z。
方块的理想点位为 `(X0,Y0,Z0)`、`(X0+L,Y0,Z0)`、`(X0+L,Y0+L,Z0)`、`(X0,Y0+L,Z0)`。

拍摄时保持相机安装与打印机坐标基准不变，移开喷嘴以免遮挡。使用原始、未裁剪、未缩放、未去畸变的完整彩色照片。
这条路径直接把原始像素、原始 K 和畸变参数一起交给 OpenCV。
不要将 `results/undistorted/` 中的去畸变照片与原始 K/畸变参数混用。

照片拍好后：

```powershell
& .\.venv\Scripts\python.exe -m CameraCalibration.transform pick `
  --image output/four_dots.png --points output/calibration_points.json `
  --output output/calibration_points_clicked.json
```

窗口按数据文件中的 P1、P2、P3、P4 顺序接收左键点击；右键或 U 撤销；Enter 保存；Esc 退出且不保存。
图片按原始分辨率显示，点击坐标就是完整图片的 uv。也可以手动填写每项 `image_uv_px: [u,v]`，跳过窗口。

## 3. 求变换并查看误差

```powershell
& .\.venv\Scripts\python.exe -m CameraCalibration.transform solve `
  --points output/calibration_points_clicked.json --image output/four_dots.png `
  --output output/printer_registration
```

输出：

- `transform.json`：`printer_to_camera`、`camera_to_printer`、候选姿态、每点误差和 RMS，平移单位 mm。
- `reprojection.png`：绿色圆圈为点击位置，红色十字为投影位置。

平面点使用 IPPE 并进行 LM 优化；非共面点使用 SQPNP 并进行 LM 优化。
所有点必须在相机前方。默认任一点重投影误差超过 1 px 时不接受结果，`--max-error-px` 可修改阈值。
平面候选的 RMS 差不超过 0.05 px 时，状态为 `needs_pose_selection`，顶层变换为空。
先根据相机朝向、深度和候选矩阵检查姿态，再用 `--pose-index N` 和新的输出目录明确选择候选；候选编号按误差排序。
像素拟合误差不能代替毫米精度验证。输出的 `metric_validation` 初始始终为 `not_performed`。

### 彩色坐标与深度坐标

当前内参是彩色图像标定结果，因此默认输出的是**彩色相机 ↔ 打印机**变换。
单独的彩色结果不会生成深度高度图配置，`surface_mapper_update` 为 null。

若从 RealSense SDK 取得深度到彩色的外参，可保存为：

```json
{
  "from_frame": "depth",
  "to_frame": "color",
  "units": "mm",
  "matrix": "这里填入实际的 4x4 刚体矩阵"
}
```

`matrix` 必须是真正的二维数字数组；SDK 外参平移若为米，先乘 1000。
加入 `solve --depth-to-color 实际文件.json` 后，程序计算
`T_printer_from_depth = T_printer_from_color @ T_color_from_depth`，并额外生成
`depth_camera_to_printer` 和 `surface_mapper_update`。
它不会覆盖现有相机配置；网格范围和模型训练几何仍需与新坐标系一致。

## 4. 准备喷嘴移动

复制 `configs/controller.example.json` 和 `configs/nozzle_motion.example.json` 为自己的配置文件。

控制器配置需要设置实际连接：

- 电脑通过 Moonraker：`hardware.backend: "moonraker"`，填写实际 `moonraker_url`。
- 树莓派与 Klippy 在同一 Linux 主机：`hardware.backend: "klipper_socket"`，填写现有的本地 API socket。
- 核对 `limits`，填写测得的 `hardware.fixture_top_z_mm`（完整夹具和障碍物的最高 Z）与冷态净空。

只读查询当前位置：

```powershell
& .\.venv\Scripts\python.exe -m controller.nozzle status --config configs/controller.local.json
```

它不发送 G-code。输出打印机 XYZ、G-code XYZ、原点偏移和归零状态。

移动计划里的参数：

- `start_xyz_mm`：执行开始时喷嘴的实际位置，硬件执行前会核对。
- `travel_z_mm`：横移高度，至少与开始 Z 和所有目标 Z 一样高。
- `waypoints[].xyz_mm`：喷嘴目标 XYZ，不是表面标记的 XYZ；定位到点上方时需要加净空。
- `xy_speed_mm_s` / `z_speed_mm_s`：分别设置横移和升降速度，不能超过控制器上限。
- `dwell_s`：每个点到达后停留时间；`return_to_start` 决定最后是否回到起点。

路径依次执行：原地升至 travel Z → 在 travel Z 横移 → 下降至目标 Z → 停留 → 升回 travel Z。
开始点、目标点和整个路径均需高于夹具最高处加净空；所有 XY 横移都在 travel Z 完成。

先生成预览，不连接打印机：

```powershell
& .\.venv\Scripts\python.exe -m controller.nozzle run `
  --config configs/controller.local.json --plan configs/nozzle_motion.local.json `
  --output output/nozzle_preview
```

输出 `motion_preview.json` 和 `motion_preview.gcode`。内部速度使用 mm/s，G-code 的 F 已换算为 mm/min。

实际定位在准备好机器后使用同一命令增加 `--execute`，同时改用新的输出目录。
硬件需已完成归零、停止其他任务、关闭加热并冷却、核实零偏移与高度补偿状态；程序不自动执行归零、清除偏移或改变补偿。
配置中 `stationary_fixture_verified`、`stop_test_verified`、`exclusive_control_verified` 需反映真实检查结果。
以打印机 XYZ 定位不依赖尚未完成的相机外参，不要求伪造 `calibration_id` 或相机 hash。

每段移动按请求超时和加速度限制切分，逐段 `M400` 等待队列排空并核对位置；回复丢失时不重试，提交过运动后发生故障会请求急停。
这些状态来自控制器报告，不代表独立物理测量。执行日志为 `motion_events.jsonl` 和 `execution.json`。

## 参考

实际深度误差检查与下一步静止直线试验见
[独立第五点测量](../printer_control/STATIONARY_LINE_CHECK.md)。
`python -m CameraCalibration.metric_check` 支持配对采集、点选和毫米误差报告；
不会重新拟合变换或启用打印机运动。

- [OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [Moonraker API](https://moonraker.readthedocs.io/en/latest/web_api/)
- [Klipper 状态字段](https://www.klipper3d.org/Status_Reference.html)
- [Klipper G-code](https://www.klipper3d.org/G-Codes.html)

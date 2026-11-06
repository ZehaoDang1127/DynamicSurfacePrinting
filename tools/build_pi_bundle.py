"""Build a clean Pi deployment ZIP from current sources and explicit local profiles.

Run from any directory. No Git checkout, printer connection, or network required.
Local trial outputs are included only through the two explicitly selected JSONs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = "Dynamic_Surface_Printer_Pi_20261026"


def build(destination, profile_dir, wheelhouse, live=False):
    files = {}
    for directory in ("controller", "surface_mapper", "prediction", "CameraCalibration", "pi_support", "tests"):
        for source in sorted((ROOT / directory).rglob("*.py")):
            files[source.relative_to(ROOT).as_posix()] = source.read_bytes()
    for source in sorted((ROOT / "configs").glob("*.json")):
        if source.name == "splits.json":
            continue
        files[source.relative_to(ROOT).as_posix()] = source.read_bytes()
    for source in sorted((ROOT / "docs").rglob("*.md")):
        files[source.relative_to(ROOT).as_posix()] = source.read_bytes()
    for name in ("setup_pi.sh", "run_pi.sh", "README.md", "tools/matlab/visualize_ca_live.m",
                 "tools/collect_stationary_failure.py", "tools/build_pi_bundle.py",
                 "CameraCalibration/results/calibration.json", "CameraCalibration/results/calibration.npz"):
        files[name] = (ROOT / name).read_bytes()
    for source in sorted(ROOT.glob("requirements*.txt")):
        files[source.name] = source.read_bytes()
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels or not any(p.name.startswith("pyrealsense2-") and "aarch64" in p.name for p in wheels):
        raise ValueError("A verified ARM64/Python 3.12 wheelhouse including pyrealsense2 is required")
    for source in wheels:
        files["wheelhouse/" + source.name] = source.read_bytes()

    guide = (ROOT / ("docs/raspberry_pi/CONVLSTM_LIVE_PI_GUIDE_ZH.md" if live else "docs/raspberry_pi/RASPBERRY_PI_GUIDE_ZH.md")).read_bytes()
    files["GUIDELINE_中文.md"] = guide
    files["START_HERE.md"] = (
        "# Raspberry Pi 部署包 — 2026-10-26\n\n"
        "先阅读 [中文操作指南](GUIDELINE_中文.md)。全部命令在树莓派本机终端执行。\n\n"
        "```bash\nbash setup_pi.sh\nbash run_pi.sh self-test\nbash run_pi.sh diagnose\n```\n\n"
        "已有可用环境可按指南使用 PI_PYTHON。首次安装需要联网。\n"
        "本次更新增加停机前状态快照；停机原因尚未确认。先收集旧失败目录的诊断资料。\n"
        "默认入口只读、预览或离线测试；不会启动打印。Pi 硬件运行尚未现场验证。\n"
    ).encode("utf-8")
    controller = json.loads((profile_dir / "controller.json").read_text(encoding="utf-8-sig"))
    controller["hardware"]["moonraker_url"] = "http://192.168.50.2:7125"
    for flag in ("stationary_fixture_verified", "stop_test_verified", "exclusive_control_verified"):
        controller["hardware"][flag] = False
    plan = json.loads((profile_dir / "plan.json").read_text(encoding="utf-8-sig"))
    plan["loaded_and_primed"] = False
    plan["path_clearance_verified"] = False
    for name, value in (("controller.json", controller), ("plan.json", plan)):
        files[f"profiles/stationary/{name}"] = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    files["output/.gitkeep"] = b""
    files["models/README.md"] = (
        "No trained checkpoint is bundled. NumPy baselines need no checkpoint.\n"
        "ConvLSTM requires separately installed PyTorch and a model matching the camera geometry and cadence.\n"
    ).encode()
    if live:
        if not any(p.name.startswith("torch-") and "aarch64" in p.name for p in wheels):
            raise ValueError("The live bundle requires ARM64 PyTorch in its wheelhouse")
        for name in ("camera.json", "path.json"):
            files["profiles/live/" + name] = (ROOT / "output/integrated_setup_20261025" / name).read_bytes()
        live_controller = json.loads((ROOT / "output/integrated_setup_20261025/controller.json").read_text(encoding="utf-8-sig"))
        live_controller["desired_gap_mm"] = 0.25
        live_controller["hardware"]["moonraker_url"] = "http://192.168.50.2:7125"
        live_controller["predictor"].update(kind="convlstm", checkpoint="models/d405/best.pt", device="cpu",
            history=10, max_horizon_s=1.0, temporal_policy="linear_to_horizon", interpolation_error_mm=0.1)
        # Keep the recorded millimetre-scale calibration budget; never shrink it
        # to make a 0.25 mm nominal gap appear qualified for actuation.
        files["profiles/live/controller.json"] = (json.dumps(live_controller, indent=2) + "\n").encode()
        files["profiles/live/settings.json"] = (ROOT / "configs/live_print.example.json").read_bytes()
        pretrained = ROOT / "output/tds_cloth/benchmark/horizon_600ms/model/best.pt"
        files["models/pretrained/cloth_600ms_16x16.pt"] = pretrained.read_bytes()
        files["models/d405/README.md"] = (
            "Place the evaluated checkpoint trained/fine-tuned on the matching D405 geometry here as best.pt.\n"
            "No deployment-qualified D405 checkpoint is available in this repository.\n").encode()
        files["models/README.md"] = (
            "The supplied models/pretrained/cloth_600ms_16x16.pt is for weight initialization only.\n"
            "Use prediction train --initialize-from with real D405 training/validation sessions.\n"
            "Do not copy the cloth file to models/d405/best.pt or change its metadata to bypass geometry checks.\n").encode()
        files["START_HERE.md"] = (
            "# ConvLSTM 分段闭环实验版 — 2026-10-27\n\n"
            "阅读 [完整中文指南](GUIDELINE_中文.md)。\n\n"
            "```bash\nbash setup_pi.sh --convlstm\nbash run_pi.sh doctor --convlstm\nbash run_pi.sh live-self-test\n```\n\n"
            "包含相机/坐标变换、ConvLSTM 时间适配、XYZ/E 分段提交和监控源码。\n"
            "当前真实 D405 模型缺失，已有标定误差大于候选 0.25 mm 间距，默认实机条件不满足。\n"
            "布料权重仅作预训练起点；先完成真实数据微调、精度和时序验证。\n"
            "这不是已通过实机验证的连续动态打印成品。已有诊断功能继续保留。\n").encode("utf-8")
    # Linux scripts and copied text files are UTF-8, LF-only, without a BOM.
    for name in files:
        if Path(name).suffix in {".sh", ".py", ".md", ".txt", ".json", ".m"}:
            files[name] = files[name].decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")
    metadata = {
        "release": "2026-10-27-convlstm-segmented-experimental" if live else "2026-10-26-stationary-diagnostics",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "target": "Raspberry Pi 5 / Raspberry Pi OS or Debian ARM64 / Python 3.12",
        "entry_guide": "GUIDELINE_中文.md",
        "profile_source": ("2026-10-25 provisional D405 registration; 0.25 mm candidate gap; measured readiness absent"
                           if live else "2026-10-25 stationary trial; URL set to direct Ethernet; readiness flags false"),
        "hardware_tested": False,
        "incident_root_cause_confirmed": False,
        "deployment_d405_checkpoint_included": False,
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    files["BUNDLE_MANIFEST.json"] = (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create avoids overwriting a previously handed-off archive.
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            bundle_root = "Dynamic_Surface_Printer_Pi_ConvLSTM_20261027" if live else BUNDLE_ROOT
            info = zipfile.ZipInfo(f"{bundle_root}/{name}", date_time=(2026, 10, 27 if live else 26, 0, 0, 0))
            info.create_system = 3
            mode = 0o755 if name.endswith(".sh") else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(f"{digest}  {destination.name}\n", encoding="ascii")
    print(json.dumps({"zip": str(destination.resolve()), "files": len(files),
                      "size_bytes": destination.stat().st_size, "sha256": digest}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "output/stationary_print_20261025")
    parser.add_argument("--wheelhouse", type=Path, default=ROOT / "output/pi_dependency_check_20261026")
    parser.add_argument("--live", action="store_true", help="package experimental ConvLSTM segmented control and pretrained initialization")
    args = parser.parse_args()
    build(args.output.resolve(), args.profile_dir.resolve(), args.wheelhouse.resolve(), args.live)


if __name__ == "__main__":
    main()

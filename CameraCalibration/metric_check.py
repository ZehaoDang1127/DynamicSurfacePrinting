"""Independent metric point checks before using registration for nozzle motion.

Capture reads the D405 only. Check never refits a transform or enables motion.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .transform import pick_points, read_json, rigid_matrix, write_json


def vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{name} requires {length} measured numeric values")
    result = np.array(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def capture(camera_config, output, frames=15):
    """Use matching D405 color/depth modes; store SDK raw-depth XYZ/color UV."""
    import pyrealsense2 as rs
    from surface_mapper.config import load_config
    app = load_config(camera_config)
    if not app.camera.serial:
        raise ValueError("camera config must identify the actual D405 serial")
    if type(frames) is not int or not 10 <= frames <= 60:
        raise ValueError("capture 10 to 60 stationary frames")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pipeline, settings = rs.pipeline(), rs.config()
    settings.enable_device(app.camera.serial)
    # The D405's shared imager cannot resolve arbitrary mixed depth/color modes.
    settings.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    settings.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    started = False
    try:
        profile = pipeline.start(settings)
        started = True
        cloud = rs.pointcloud()
        xyz, uv, times = [], [], []
        for i in range(app.camera.warmup_frames + frames):
            pair = pipeline.wait_for_frames(app.camera.timeout_ms)
            depth, color = pair.get_depth_frame(), pair.get_color_frame()
            if not depth or not color:
                raise RuntimeError("paired capture has a missing color or depth frame")
            if i < app.camera.warmup_frames:
                continue
            cloud.map_to(color)
            points = cloud.calculate(depth)
            xyz.append(np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3).copy())
            pixels = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2).copy()
            pixels *= [848, 480]
            uv.append(pixels)
            times.append([depth.get_timestamp(), color.get_timestamp()])
        image = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("could not encode the raw color image")
        encoded.tofile(output / "color.png")
        np.savez_compressed(output / "paired_depth.npz", xyz_depth_m=np.stack(xyz),
                            color_uv_px=np.stack(uv))
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        k = color_profile.get_intrinsics()
        extr = depth_profile.get_extrinsics_to(color_profile)
        matrix = np.eye(4)
        matrix[:3, :3] = np.array(extr.rotation).reshape(3, 3, order="F")
        matrix[:3, 3] = np.array(extr.translation) * 1000
        metadata = {
            "schema_version": 1, "serial": app.camera.serial, "paired_frames_saved": frames,
            "pixel_coordinates": "raw_full_image", "image_size": [848, 480],
            "depth_xyz_units": "m", "depth_to_color_units": "mm",
            "depth_to_color_mm": matrix.tolist(),
            "color": {"width": k.width, "height": k.height, "fx": k.fx, "fy": k.fy,
                "ppx": k.ppx, "ppy": k.ppy, "model": str(k.model), "coeffs": list(k.coeffs)},
            "timestamps_depth_color_ms": times,
            "color_image_sha256": hashlib.sha256((output / "color.png").read_bytes()).hexdigest(),
            "stationary_surface_required": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output / "metadata.json", metadata)
        return {"status": "captured", "frames": frames, "image": str(output / "color.png"),
                "output": str(output), "printer_commands_sent": 0}
    except Exception as exc:
        write_json(output / "capture_error.json", {"status": "failed", "reason": str(exc)})
        raise
    finally:
        if started:
            pipeline.stop()


def deproject_color(intrinsics, uv, z_mm):
    """Keep factory inverse distortion in the SDK's own camera model."""
    import pyrealsense2 as rs
    allowed = {"distortion.none": rs.distortion.none,
               "distortion.brown_conrady": rs.distortion.brown_conrady,
               "distortion.inverse_brown_conrady": rs.distortion.inverse_brown_conrady}
    if intrinsics["model"] not in allowed:
        raise ValueError("unsupported factory color distortion model for this point check")
    k = rs.intrinsics()
    k.width, k.height = intrinsics["width"], intrinsics["height"]
    k.fx, k.fy = intrinsics["fx"], intrinsics["fy"]
    k.ppx, k.ppy = intrinsics["ppx"], intrinsics["ppy"]
    k.coeffs, k.model = intrinsics["coeffs"], allowed[intrinsics["model"]]
    return np.array(rs.rs2_deproject_pixel_to_point(k, list(uv), z_mm / 1000)) * 1000


def sample_point(xyz_depth_m, color_uv_px, metadata, uv, radius_px=1., min_frames=10,
                 deproject=deproject_color):
    xyz = np.asarray(xyz_depth_m, dtype=float)
    pixels = np.asarray(color_uv_px, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or pixels.shape != (*xyz.shape[:2], 2):
        raise ValueError("paired arrays must be [frames, points, XYZ/UV]")
    if not np.isfinite(radius_px) or radius_px <= 0 or type(min_frames) is not int or min_frames < 1:
        raise ValueError("sampling radius and required frame count must be positive")
    center = vector(list(uv), 2, "image_uv_px")
    size = metadata["image_size"]
    if np.any(center < 0) or np.any(center >= size):
        raise ValueError("selected point is outside the raw full color image")
    if metadata.get("depth_xyz_units") != "m" or metadata.get("depth_to_color_units") != "mm":
        raise ValueError("capture must declare metre XYZ and millimetre extrinsic translation")
    good = (np.isfinite(xyz).all(axis=2) & (xyz[:, :, 2] > 0) &
            np.isfinite(pixels).all(axis=2) &
            (pixels >= 0).all(axis=2) & (pixels < size).all(axis=2) &
            (np.sum((pixels - center) ** 2, axis=2) <= radius_px ** 2))
    frame_count = int(good.any(axis=1).sum())
    if frame_count < min_frames:
        raise ValueError(f"insufficient observed depth at the mark: {frame_count}/{min_frames} frames")
    transform = rigid_matrix(metadata["depth_to_color_mm"], "depth_to_color_mm")
    color_samples = xyz[good] * 1000 @ transform[:3, :3].T + transform[:3, 3]
    color_z = float(np.median(color_samples[:, 2]))
    color_point = deproject(metadata["color"], center.tolist(), color_z)
    inverse = np.linalg.inv(transform)
    depth_point = inverse[:3, :3] @ color_point + inverse[:3, 3]
    return depth_point, {"valid_depth_samples": int(good.sum()), "frames_with_depth": frame_count,
        "sample_radius_px": float(radius_px),
        "sample_color_z_p05_p50_p95_mm": np.percentile(color_samples[:, 2], [5, 50, 95]).tolist()}


def compare_point(registration, expected_xyz, depth_xyz, serial, tolerance_mm=None):
    if registration.get("device_serial") != serial or not serial:
        raise ValueError("capture serial differs from the registration's physical camera")
    if registration.get("units") != "mm" or registration.get("output_frame") != "printer":
        raise ValueError("registration must target the printer frame in millimetres")
    expected = vector(list(expected_xyz), 3, "printer_xyz_mm")
    source = vector(list(depth_xyz), 3, "depth_xyz_mm")
    if any(np.linalg.norm(expected - vector(p["printer_xyz_mm"], 3, "fit point")) < .1
           for p in registration.get("measurements", [])):
        raise ValueError("the check point duplicates a fitted marker; measure an independent fifth point")
    matrix = rigid_matrix(registration["depth_camera_to_printer"], "depth_camera_to_printer")
    predicted = matrix[:3, :3] @ source + matrix[:3, 3]
    error = predicted - expected
    norm = float(np.linalg.norm(error))
    if tolerance_mm is not None and (isinstance(tolerance_mm, bool) or
            not np.isfinite(tolerance_mm) or tolerance_mm <= 0):
        raise ValueError("point tolerance must be a positive finite value")
    return {"status": "measured" if tolerance_mm is None else
            "within_point_tolerance" if norm <= tolerance_mm else "outside_point_tolerance",
            "expected_printer_xyz_mm": expected.tolist(), "predicted_printer_xyz_mm": predicted.tolist(),
            "error_xyz_mm": error.tolist(), "position_error_mm": norm,
            "absolute_z_error_mm": float(abs(error[2])), "point_tolerance_mm": tolerance_mm,
            "scope": "one withheld point; no transform refit or working-volume qualification",
            "motion_enabled": False, "printer_commands_sent": 0}


def check(capture_path, registration_path, point_path, output, tolerance_mm=None, uv=None):
    directory = Path(capture_path)
    metadata = read_json(directory / "metadata.json")
    registration, point = read_json(registration_path), read_json(point_path)
    if point.get("schema_version") != 1 or point.get("coordinate_frame") != "printer":
        raise ValueError("check point must declare schema 1 and the printer coordinate frame")
    expected = vector(point.get("printer_xyz_mm"), 3, "printer_xyz_mm")
    image_hash = hashlib.sha256((directory / "color.png").read_bytes()).hexdigest()
    if metadata.get("color_image_sha256") != image_hash:
        raise ValueError("capture color image changed after acquisition")
    if point.get("capture_image_sha256") not in (None, image_hash):
        raise ValueError("point pixel coordinates belong to a different capture")
    center = uv if uv is not None else point.get("image_uv_px")
    image = cv2.imdecode(np.fromfile(directory / "color.png", dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or [image.shape[1], image.shape[0]] != metadata["image_size"]:
        raise ValueError("raw color image does not match capture resolution")
    if center is None:
        chosen = pick_points(image, {"points": [{"id": point.get("id", "P5")}]})
        center = chosen["points"][0]["image_uv_px"]
    center = vector(list(center), 2, "image_uv_px")
    with np.load(directory / "paired_depth.npz", allow_pickle=False) as samples:
        xyz, stats = sample_point(samples["xyz_depth_m"], samples["color_uv_px"], metadata, center)
    result = compare_point(registration, expected, xyz, metadata["serial"], tolerance_mm)
    result.update(stats, image_uv_px=center.tolist(), depth_xyz_mm=xyz.tolist(),
        registration_file=str(Path(registration_path).resolve()),
        registration_sha256=hashlib.sha256(Path(registration_path).read_bytes()).hexdigest(),
        capture_directory=str(directory.resolve()), created_utc=datetime.now(timezone.utc).isoformat())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    point.update(image_uv_px=center.tolist(), capture_image_sha256=image_hash)
    write_json(output / "point_selected.json", point)
    write_json(output / "point_error.json", result)
    pixel = tuple(np.rint(center).astype(int))
    cv2.drawMarker(image, pixel, (0, 255, 255), cv2.MARKER_CROSS, 15, 1)
    ok, encoded = cv2.imencode(".png", image)
    if ok:
        encoded.tofile(output / "selected_point.png")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--xyz", type=float, nargs=3)
    cap = commands.add_parser("capture")
    cap.add_argument("--camera-config", required=True)
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--frames", type=int, default=15)
    run = commands.add_parser("check")
    run.add_argument("--capture", required=True)
    run.add_argument("--registration", required=True)
    run.add_argument("--point", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--uv", type=float, nargs=2)
    run.add_argument("--max-error-mm", type=float)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists; choose a new path")
        if args.command == "init":
            point = {"schema_version": 1, "id": "P5", "coordinate_frame": "printer",
                     "printer_xyz_mm": None if args.xyz is None else vector(args.xyz, 3, "XYZ").tolist(),
                     "image_uv_px": None, "capture_image_sha256": None,
                     "measurement_notes": "Enter actual independently measured mark coordinates; do not interpolate from P1-P4."}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, point)
            result = {"status": "template_written", "file": str(args.output)}
        elif args.command == "capture":
            result = capture(args.camera_config, args.output, args.frames)
        else:
            result = check(args.capture, args.registration, args.point, args.output, args.max_error_mm, args.uv)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("status") == "outside_point_tolerance" else 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"metric-check: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("metric-check: interrupted; no printer commands were sent", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

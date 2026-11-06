"""Label image points, estimate printer/camera poses, and check reprojection.

Run from the repository root: python -m CameraCalibration.transform --help.
All translations and object coordinates are millimetres. Images are RAW,
full-frame pixels, with the same stream and resolution as the intrinsics.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np


DEFAULT_INTRINSICS = Path(__file__).parent / "results" / "calibration.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_array(value, shape, name):
    if value is None:
        raise ValueError(f"{name} is missing")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite measured numbers")
    return array


def load_intrinsics(path):
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("intrinsics JSON must be an object")
    if any(name in str(raw.get("model", "")).lower() for name in ("fisheye", "kannala", "ftheta")):
        raise ValueError("this solver requires OpenCV pinhole/Brown-Conrady intrinsics")
    matrix = finite_array(raw["camera_matrix"], (3, 3), "camera_matrix")
    distortion = np.asarray(raw["distortion_coefficients"], dtype=np.float64).reshape(-1)
    if distortion.size not in {4, 5, 8, 12, 14} or not np.isfinite(distortion).all():
        raise ValueError("expected OpenCV pinhole distortion coefficients (4, 5, 8, 12 or 14)")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("invalid pinhole camera matrix")
    if abs(matrix[0, 1]) > 1e-9 or abs(matrix[1, 0]) > 1e-9:
        raise ValueError("OpenCV pinhole projection requires zero intrinsic skew")
    size = (raw["image_width"], raw["image_height"])
    if any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("intrinsics need positive integer image_width/image_height")
    return matrix, distortion, size


def point_template(size, camera_frame="color"):
    if camera_frame not in {"color", "depth"}:
        raise ValueError("camera_frame must be color or depth")
    return {"schema_version": 1, "camera_frame": camera_frame,
            "pixel_coordinates": "raw_full_image", "image_size": list(size),
            "synthetic_demo": False,
            "points": [{"id": f"P{i}", "printer_xyz_mm": None, "image_uv_px": None}
                       for i in range(1, 5)]}


def validate_points(raw, size, require_xyz=True, require_pixels=True):
    if not isinstance(raw, dict):
        raise ValueError("points JSON must be an object")
    if raw.get("schema_version") != 1 or raw.get("pixel_coordinates") != "raw_full_image":
        raise ValueError("use schema 1 with raw_full_image pixels (no resized/cropped/undistorted coordinates)")
    if raw.get("camera_frame") not in {"color", "depth"}:
        raise ValueError("declare camera_frame as color or depth")
    if raw.get("image_size") != list(size):
        raise ValueError(f"point image_size must match calibrated width/height {size}")
    points = raw.get("points", [])
    if not isinstance(points, list) or any(not isinstance(p, dict) for p in points):
        raise ValueError("points must be a list of labelled objects")
    ids = [p.get("id") for p in points]
    if len(points) < 4 or any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("at least four uniquely labelled points are required")
    xyz = uv = None
    if require_xyz:
        xyz = np.array([finite_array(p.get("printer_xyz_mm"), (3,), p["id"] + " XYZ") for p in points])
        if np.linalg.matrix_rank(xyz - xyz.mean(axis=0), tol=1e-7) < 2:
            raise ValueError("printer points must not be collinear or coincident")
        if len(np.unique(xyz, axis=0)) != len(xyz):
            raise ValueError("printer points must be distinct")
    if require_pixels:
        uv = np.array([finite_array(p.get("image_uv_px"), (2,), p["id"] + " uv") for p in points])
        if np.any(uv < 0) or np.any(uv >= size):
            raise ValueError("image points are outside the calibrated full image")
        if np.linalg.matrix_rank(uv - uv.mean(axis=0), tol=1e-7) < 2 or len(np.unique(uv, axis=0)) != len(uv):
            raise ValueError("image points must be distinct and non-collinear")
    return xyz, uv


def rigid_matrix(value, name="matrix"):
    matrix = finite_array(value, (4, 4), name)
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9) or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError(f"{name} must be a proper rigid transform")
    return matrix


def inverse_rigid(matrix):
    matrix = rigid_matrix(matrix)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def compose_depth_transform(camera_to_printer, camera_frame, path):
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("depth-to-color JSON must be an object")
    if camera_frame != "color" or raw.get("from_frame") != "depth" or raw.get("to_frame") != "color" or raw.get("units") != "mm":
        raise ValueError("depth-to-color needs from_frame=depth, to_frame=color, units=mm; solved pose must use color")
    return rigid_matrix(camera_to_printer) @ rigid_matrix(raw["matrix"], "depth_to_color")


def solve_pose(xyz, uv, matrix, distortion, max_error_px=1.0, ambiguity_px=0.05, pose_index=None):
    """Return all positive-depth PnP candidates; do not silently accept a tie."""
    if not np.isfinite([max_error_px, ambiguity_px]).all() or max_error_px <= 0 or ambiguity_px < 0:
        raise ValueError("max_error_px must be positive and ambiguity_px nonnegative")
    planar = np.linalg.matrix_rank(xyz - xyz.mean(axis=0), tol=1e-7) == 2
    flag = cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP
    count, rotations, translations, *_ = cv2.solvePnPGeneric(xyz, uv, matrix, distortion, flags=flag)
    if not count:
        raise ValueError("PnP produced no solution")
    candidates = []
    for rvec, tvec in zip(rotations, translations):
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            continue
        # Refine each basin independently; retain genuinely distinct poses only.
        rvec, tvec = cv2.solvePnPRefineLM(
            xyz, uv, matrix, distortion, rvec.copy(), tvec.copy(),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-12))
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = xyz @ rotation.T + tvec.reshape(3)
        if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
            continue
        forward = np.eye(4)
        forward[:3, :3], forward[:3, 3] = rotation, tvec.reshape(3)
        if any(np.allclose(forward[:3, :3], np.array(c["printer_to_camera"])[:3, :3], atol=1e-5, rtol=0)
               and np.allclose(forward[:3, 3], np.array(c["printer_to_camera"])[:3, 3], atol=1e-3, rtol=0)
               for c in candidates):
            continue
        projected, _ = cv2.projectPoints(xyz, rvec, tvec, matrix, distortion)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - uv, axis=1)
        candidates.append({"rvec": rvec.reshape(3).tolist(), "tvec_mm": tvec.reshape(3).tolist(),
                           "printer_to_camera": forward.tolist(), "camera_to_printer": inverse_rigid(forward).tolist(),
                           "projected_uv_px": projected.tolist(), "per_point_error_px": errors.tolist(),
                           "rms_error_px": float(np.sqrt(np.mean(errors**2))),
                           "max_error_px": float(errors.max()), "min_camera_depth_mm": float(camera_points[:, 2].min())})
    candidates.sort(key=lambda c: c["rms_error_px"])
    if not candidates:
        raise ValueError("no finite pose puts all reference points in front of the camera")
    for index, candidate in enumerate(candidates):
        candidate["pose_index"] = index
    if pose_index is not None and not 0 <= pose_index < len(candidates):
        raise ValueError("pose_index is outside the candidate list")
    index = 0 if pose_index is None else pose_index
    chosen = candidates[index]
    ambiguous = len(candidates) > 1 and candidates[1]["rms_error_px"] - candidates[0]["rms_error_px"] <= ambiguity_px
    if chosen["max_error_px"] > max_error_px:
        status = "reprojection_error_too_large"
    elif ambiguous and pose_index is None:
        status = "needs_pose_selection"
    else:
        status = "solved"
    return {"status": status, "solver": "IPPE + LM" if planar else "SQPNP + LM",
            "selected_pose_index": index if status == "solved" else None,
            "ambiguous": ambiguous, "candidates": candidates,
            "printer_to_camera": chosen["printer_to_camera"] if status == "solved" else None,
            "camera_to_printer": chosen["camera_to_printer"] if status == "solved" else None}


def load_image(path, size):
    # imdecode supports Windows Unicode paths.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or (image.shape[1], image.shape[0]) != size:
        raise ValueError(f"image must decode at calibrated full resolution {size}")
    return image


def pick_points(image, raw):
    """Native OpenCV window is opened only by the explicit `pick` command."""
    selected = []
    window = "Left click: add | Right click/U: undo | Enter: save | Esc: cancel"
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < len(raw["points"]):
            selected.append([float(x), float(y)])
        elif event == cv2.EVENT_RBUTTONDOWN and selected:
            selected.pop()
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            canvas = image.copy()
            for point, xy in zip(raw["points"], selected):
                pixel = tuple(np.rint(xy).astype(int))
                cv2.drawMarker(canvas, pixel, (0, 255, 0), cv2.MARKER_CROSS, 15, 1)
                cv2.putText(canvas, point["id"], (pixel[0]+8, pixel[1]-8), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 255
            if key in {27, ord("q")} or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("point selection cancelled; no output saved")
            if key in {ord("u"), 8} and selected:
                selected.pop()
            if key in {10, 13} and len(selected) == len(raw["points"]):
                break
    finally:
        cv2.destroyWindow(window)
    for point, xy in zip(raw["points"], selected):
        point["image_uv_px"] = xy
    return raw


def save_solution(intrinsics, points, output, image_path=None, depth_to_color=None, **options):
    matrix, distortion, size = load_intrinsics(intrinsics)
    raw = read_json(points)
    xyz, uv = validate_points(raw, size)
    image = load_image(image_path, size) if image_path else None
    result = solve_pose(xyz, uv, matrix, distortion, **options)
    result.update({"schema_version": 1, "units": "mm", "camera_frame": raw["camera_frame"],
                   "output_frame": "printer", "image_size": list(size), "point_ids": [p["id"] for p in raw["points"]],
                   "synthetic_demo": bool(raw.get("synthetic_demo", False)), "metric_validation": "not_performed",
                   "intrinsics_file": str(Path(intrinsics).resolve()), "points_file": str(Path(points).resolve()),
                   "created_utc": datetime.now(timezone.utc).isoformat(), "surface_mapper_update": None})
    if result["status"] == "solved":
        depth = None
        if depth_to_color:
            depth = compose_depth_transform(result["camera_to_printer"], raw["camera_frame"], depth_to_color)
        elif raw["camera_frame"] == "depth":
            depth = np.array(result["camera_to_printer"])
        if depth is not None:
            result["depth_camera_to_printer"] = depth.tolist()
            result["surface_mapper_update"] = {"frame_name": "printer", "camera_to_output": depth.tolist()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "transform.json", result)
    if image is not None:
        index = result["selected_pose_index"] or 0
        projected = result["candidates"][index]["projected_uv_px"]
        for point, observed, predicted in zip(raw["points"], uv, projected):
            a, b = tuple(np.rint(observed).astype(int)), tuple(np.rint(predicted).astype(int))
            cv2.circle(image, a, 6, (0, 255, 0), 1)
            cv2.drawMarker(image, b, (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
            cv2.line(image, a, b, (255, 255, 0), 1)
            cv2.putText(image, point["id"], (a[0]+8, a[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("failed to encode reprojection image")
        encoded.tofile(output / "reprojection.png")
    return result


def demo(intrinsics, output):
    """Known tilted pose + synthetic pixels: a runnable check without a photo."""
    matrix, distortion, size = load_intrinsics(intrinsics)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    xyz = np.array([[100, 100, 20], [140, 100, 20], [140, 140, 20], [100, 140, 20]], dtype=float)
    angle = np.deg2rad(20)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, -s], [0, -1, 0], [-s, 0, -c]])
    camera_origin = np.array([120, 120, 300])
    translation = -rotation @ camera_origin
    rvec, _ = cv2.Rodrigues(rotation)
    uv, _ = cv2.projectPoints(xyz, rvec, translation, matrix, distortion)
    raw = point_template(size)
    raw["synthetic_demo"] = True
    for point, world, pixel in zip(raw["points"], xyz, uv.reshape(-1, 2)):
        point["printer_xyz_mm"], point["image_uv_px"] = world.tolist(), pixel.tolist()
    image = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
    cv2.putText(image, "SYNTHETIC DEMO - NOT A CALIBRATION PHOTO", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 180), 1)
    for pixel in uv.reshape(-1, 2):
        cv2.circle(image, tuple(np.rint(pixel).astype(int)), 5, (40, 40, 40), -1)
    _, encoded = cv2.imencode(".png", image)
    encoded.tofile(output / "image.synthetic.png")
    write_json(output / "points.synthetic.json", raw)
    expected = np.eye(4)
    expected[:3, :3], expected[:3, 3] = rotation, translation
    write_json(output / "expected.synthetic.json", {"printer_to_camera": expected.tolist(), "camera_to_printer": inverse_rigid(expected).tolist()})
    return save_solution(intrinsics, output / "points.synthetic.json", output / "solution", output / "image.synthetic.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "pick", "solve", "demo"):
        sub = commands.add_parser(name)
        sub.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
        sub.add_argument("--output", type=Path, required=True)
        if name == "init":
            sub.add_argument("--camera-frame", choices=["color", "depth"], default="color")
        if name in {"pick", "solve"}:
            sub.add_argument("--points", type=Path, required=True)
            sub.add_argument("--image", type=Path, required=name == "pick")
        if name == "solve":
            sub.add_argument("--pose-index", type=int)
            sub.add_argument("--max-error-px", type=float, default=1.0)
            sub.add_argument("--ambiguity-px", type=float, default=0.05)
            sub.add_argument("--depth-to-color", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists; choose a new path")
        if args.command == "init":
            _, _, size = load_intrinsics(args.intrinsics)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, point_template(size, args.camera_frame))
            result = {"status": "template_written", "file": str(args.output)}
        elif args.command == "pick":
            _, _, size = load_intrinsics(args.intrinsics)
            raw = read_json(args.points)
            validate_points(raw, size, require_xyz=False, require_pixels=False)
            raw = pick_points(load_image(args.image, size), raw)
            raw["image_file"] = str(args.image.resolve())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, raw)
            result = {"status": "pixels_saved", "file": str(args.output)}
        elif args.command == "demo":
            result = demo(args.intrinsics, args.output)
        else:
            result = save_solution(args.intrinsics, args.points, args.output, args.image, args.depth_to_color,
                                   pose_index=args.pose_index, max_error_px=args.max_error_px, ambiguity_px=args.ambiguity_px)
        print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2, allow_nan=False))
        return 0 if result["status"] in {"solved", "template_written", "pixels_saved"} else 2
    except (ValueError, KeyError, TypeError, OSError, cv2.error) as exc:
        print(f"transform: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

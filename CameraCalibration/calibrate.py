"""Calibrate the supplied color images with OpenCV; preserve every source image."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


PATTERN = (9, 6)  # Inner corners, columns then rows; 10 x 7 squares.
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-10)
PARAMETERS = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
DOCUMENTATION = "https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html"


def fit(object_points, corners, size):
    return cv2.calibrateCameraExtended(
        [object_points.copy() for _ in corners], list(corners), size,
        None, None, flags=0, criteria=CRITERIA,
    )


def pose_error(object_points, corners, matrix, distortion):
    ok, rotation, translation = cv2.solvePnP(
        object_points, corners, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("Board pose estimation failed")
    projected, _ = cv2.projectPoints(
        object_points, rotation, translation, matrix, distortion
    )
    residual = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residual.astype(float) ** 2, axis=1))))
    return rms, rotation, translation, projected, residual


def write_image(path, pixels):
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f"Could not write {path}")


def label(pixels, text):
    banner = cv2.copyMakeBorder(pixels, 30, 0, 0, 0, cv2.BORDER_CONSTANT,
                               value=(248, 248, 248))
    cv2.putText(banner, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (25, 25, 25), 1, cv2.LINE_AA)
    return banner


def contact_sheet(tiles, columns=5):
    height, width = tiles[0].shape[:2]
    canvas = np.full((math.ceil(len(tiles) / columns) * height,
                      columns * width, 3), 248, np.uint8)
    for index, tile in enumerate(tiles):
        x, y = (index % columns) * width, (index // columns) * height
        canvas[y:y + height, x:x + width] = tile
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument("--keep-all", action="store_true",
                        help="Keep every detected view instead of one-pass robust filtering")
    args = parser.parse_args()
    if not math.isfinite(args.square_size_mm) or args.square_size_mm <= 0:
        parser.error("--square-size-mm must be positive and finite")
    source = args.input.resolve()
    output = (args.output or source / "results").resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("corners", "undistorted", "residuals"):
        (output / folder).mkdir(exist_ok=True)
    cv2.setNumThreads(4)
    files = sorted(source.glob("*_Color.png"), key=lambda p: int(p.stem.split("_")[0]))
    if not files:
        raise RuntimeError(f"No *_Color.png images in {source}")
    size = None
    views, failed = [], []
    detector_flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                      | cv2.CALIB_CB_ACCURACY)
    for path in files:
        pixels = cv2.imread(str(path))
        if pixels is None:
            raise RuntimeError(f"Cannot read {path}")
        this_size = (pixels.shape[1], pixels.shape[0])
        if size is not None and this_size != size:
            raise RuntimeError("All calibration images must have the same resolution")
        size = this_size
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=detector_flags)
        print(f"{path.name}: {'54 corners' if ok else 'not detected'}", flush=True)
        if ok:
            views.append({"path": path, "pixels": pixels, "corners": corners,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        else:
            failed.append(path.name)
    if len(views) < 10:
        raise RuntimeError("Fewer than 10 detected views; collect more varied board poses")
    objects = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objects[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
    objects *= args.square_size_mm
    corners = np.array([view["corners"] for view in views])
    baseline = fit(objects, corners, size)
    initial_errors = baseline[7].ravel()
    median = float(np.median(initial_errors))
    mad = float(np.median(np.abs(initial_errors - median)))
    # Single pass, with a half-pixel floor; never repeatedly prune to reduce RMS.
    cutoff = max(0.5, median + 3 * 1.4826 * mad)
    retained = np.ones(len(views), bool) if args.keep_all else initial_errors <= cutoff
    if int(retained.sum()) < 10:
        raise RuntimeError("Filtering would leave fewer than 10 views; inspect detections")
    result = baseline if retained.all() else fit(objects, corners[retained], size)
    rms, matrix, distortion, rotations, translations, std_intrinsics, std_extrinsics, _ = result
    if not (np.isfinite(matrix).all() and np.isfinite(distortion).all()
            and matrix[0, 0] > 0 and matrix[1, 1] > 0):
        raise RuntimeError("Invalid fitted camera parameters")

    # Hold out each retained image, fit intrinsics to the others, and estimate only
    # the held-out board pose. This checks view-to-view consistency, not metric accuracy.
    selected = np.flatnonzero(retained)
    heldout_errors = {}
    heldout_parameters = []
    for index in selected:
        train = selected[selected != index]
        trained = fit(objects, corners[train], size)
        heldout_errors[int(index)] = pose_error(objects, corners[index], trained[1], trained[2])[0]
        heldout_parameters.append([trained[1][0, 0], trained[1][1, 1],
                                   trained[1][0, 2], trained[1][1, 2]])
    heldout_rms = float(np.sqrt(np.mean(np.array(list(heldout_errors.values())) ** 2)))
    heldout_parameters = np.array(heldout_parameters)

    # Alpha=0 fills the output; alpha=1 keeps the source field of view with black borders.
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, size, 0, size)
    wide_matrix, wide_roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, size, 1, size)
    rows, corner_tiles, examples = [], [], []
    residual_sets, coverage_points = [], []
    view_results = []
    example_indices = {0, 8, len(views) - 1}
    for index, view in enumerate(views):
        error, rotation, translation, projected, residual = pose_error(
            objects, view["corners"], matrix, distortion
        )
        used = bool(retained[index])
        if used:
            residual_sets.append(residual)
            coverage_points.append(view["corners"].reshape(-1, 2))
        status = "used" if used else "excluded"
        rows.append({"image": view["path"].name, "used": used,
                     "baseline_rms_px": float(initial_errors[index]),
                     "final_model_pose_rms_px": error,
                     "held_out_rms_px": heldout_errors.get(index, ""),
                     "max_corner_error_px": float(np.linalg.norm(residual, axis=1).max())})
        view_results.append({**rows[-1], "source_sha256": view["sha256"],
                             "rvec_board_to_camera": rotation.ravel().tolist(),
                             "tvec_board_to_camera_mm": translation.ravel().tolist()})
        annotated = view["pixels"].copy()
        cv2.drawChessboardCorners(annotated, PATTERN, view["corners"], True)
        write_image(output / "corners" / view["path"].name, annotated)
        corner_tiles.append(label(cv2.resize(annotated, (339, 192)),
                                  f"{view['path'].stem}: {error:.3f} px ({status})"))
        undistorted = cv2.undistort(view["pixels"], matrix, distortion, None, new_matrix)
        write_image(output / "undistorted" / view["path"].name, undistorted)
        residual_image = view["pixels"].copy()
        for observed, fitted, delta in zip(view["corners"].reshape(-1, 2),
                                           projected.reshape(-1, 2), residual):
            center = tuple(np.rint(fitted).astype(int))
            cv2.circle(residual_image, center, 2, (0, 230, 0), -1, cv2.LINE_AA)
            cv2.arrowedLine(residual_image, center,
                            tuple(np.rint(fitted + 25 * delta).astype(int)),
                            (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.2)
        write_image(output / "residuals" / view["path"].name,
                    label(residual_image, f"{view['path'].name}: residual arrows x25; {error:.3f} px"))
        if index in example_indices:
            examples.append(np.hstack((label(view["pixels"], f"{view['path'].name} / original"),
                                       label(undistorted, f"Undistorted / alpha=0 / full {size[0]} x {size[1]}"))))
    write_image(output / "corner_detections.jpg", contact_sheet(corner_tiles))
    write_image(output / "undistortion_comparison.jpg", np.vstack(examples))
    coverage = np.full((size[1], size[0], 3), 248, np.uint8)
    coverage_points = np.vstack(coverage_points)
    for x, y in coverage_points:
        cv2.circle(coverage, (round(float(x)), round(float(y))), 2,
                   (170, 90, 20), -1, cv2.LINE_AA)
    write_image(output / "corner_coverage.png", label(coverage, "Detected corners in retained views"))
    residuals = np.vstack(residual_sets).astype(float)
    verified_rms = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    if not np.isclose(verified_rms, rms, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"Independent reprojection check failed: {verified_rms} vs {rms}")

    with (output / "per_image_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "opencv_version": cv2.__version__, "model": "OpenCV pinhole / Brown-Conrady, 5 coefficients",
        "image_width": size[0], "image_height": size[1],
        "pattern_inner_corners_columns_rows": list(PATTERN),
        "square_size_mm": args.square_size_mm,
        "object_point_convention": "Z=0; first detected inner corner is origin; x follows columns, y follows rows. Symmetric board orientation can flip between views.",
        "pose_convention": "X_camera_mm = Rodrigues(rvec) @ X_board_mm + tvec_mm; camera x right, y down, z forward",
        "camera_matrix": matrix.tolist(), "distortion_coefficients": distortion.ravel().tolist(),
        "distortion_coefficient_order": ["k1", "k2", "p1", "p2", "k3"],
        "rms_reprojection_error_px": float(rms), "verified_rms_px": verified_rms,
        "all_detected_views_baseline": {"rms_px": float(baseline[0]),
                                        "camera_matrix": baseline[1].tolist(),
                                        "distortion_coefficients": baseline[2].ravel().tolist(),
                                        "intrinsic_parameter_change_final_minus_baseline":
                                        [float(matrix[0, 0] - baseline[1][0, 0]),
                                         float(matrix[1, 1] - baseline[1][1, 1]),
                                         float(matrix[0, 2] - baseline[1][0, 2]),
                                         float(matrix[1, 2] - baseline[1][1, 2])]},
        "outlier_policy": {"enabled": not args.keep_all,
                           "rule": "Single pass on baseline per-view RMS: max(0.5 px, median + 3 * 1.4826 * MAD)",
                           "median_px": median, "mad_px": mad, "cutoff_px": cutoff},
        "images_found": len(files), "images_detected": len(views), "images_used": int(retained.sum()),
        "detection_failures": failed,
        "excluded_images": [view["path"].name for i, view in enumerate(views) if not retained[i]],
        "intrinsic_standard_deviations": dict(zip(PARAMETERS, std_intrinsics.ravel()[:9].tolist())),
        "leave_one_view_out": {"rms_px": heldout_rms,
                               "median_px": float(np.median(list(heldout_errors.values()))),
                               "max_px": max(heldout_errors.values()),
                               "intrinsic_order": ["fx", "fy", "cx", "cy"],
                               "intrinsic_min_px": heldout_parameters.min(axis=0).tolist(),
                               "intrinsic_max_px": heldout_parameters.max(axis=0).tolist(),
                               "note": "Intrinsics fitted without the held-out image; board pose fitted on its 54 corners. Checks consistency, not independent metric accuracy."},
        "corner_coordinate_bounds_px": [coverage_points.min(axis=0).tolist(), coverage_points.max(axis=0).tolist()],
        "new_camera_matrix_alpha0": new_matrix.tolist(), "valid_roi_alpha0_xywh": list(roi),
        "new_camera_matrix_alpha1": wide_matrix.tolist(), "valid_roi_alpha1_xywh": list(wide_roi),
        "views": view_results,
        "limitations": ["Color images only; does not calibrate the depth stream, color-depth registration, or camera-to-printer transform.",
                        "Reprojection RMS is a pixel fitting error, not a millimeter accuracy measurement.",
                        f"The model assumes a flat rigid target with accurately square {args.square_size_mm:g} mm cells; target deformation and motion can bias parameters.",
                        "Use with the same stream, resolution, crop, and focus; edge regions outside observed corners are less constrained.",
                        "Formal OpenCV standard deviations do not include target imperfections or other systematic errors."],
        "opencv_reference": DOCUMENTATION,
    }
    (output / "calibration.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "calibration.npz", camera_matrix=matrix,
                        distortion_coefficients=distortion, image_size=np.array(size),
                        square_size_mm=args.square_size_mm, pattern_size=np.array(PATTERN),
                        rms=rms, new_camera_matrix_alpha0=new_matrix, roi_alpha0=np.array(roi),
                        new_camera_matrix_alpha1=wide_matrix, roi_alpha1=np.array(wide_roi),
                        rvecs=np.array(rotations), tvecs_mm=np.array(translations),
                        std_intrinsics=std_intrinsics, std_extrinsics=std_extrinsics,
                        used_image_names=np.array([views[i]["path"].name for i in selected]))
    np.savez_compressed(output / "detected_points.npz", object_points_mm=objects,
                        image_points=corners, used_mask=retained,
                        image_names=np.array([v["path"].name for v in views]))
    np.savez_compressed(output / "all_views_calibration.npz", camera_matrix=baseline[1],
                        distortion_coefficients=baseline[2], rms=baseline[0],
                        image_size=np.array(size), square_size_mm=args.square_size_mm,
                        pattern_size=np.array(PATTERN), rvecs=np.array(baseline[3]),
                        tvecs_mm=np.array(baseline[4]), per_view_errors_px=baseline[7],
                        image_names=np.array([v["path"].name for v in views]))
    storage = cv2.FileStorage(str(output / "calibration.yaml"), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError("Could not open calibration.yaml")
    for key, value in {"image_width": size[0], "image_height": size[1],
                       "square_size_mm": args.square_size_mm, "pattern_columns": PATTERN[0],
                       "pattern_rows": PATTERN[1], "camera_matrix": matrix,
                       "distortion_coefficients": distortion,
                       "distortion_coefficient_order": "k1,k2,p1,p2,k3",
                       "rms_reprojection_error_px": float(rms),
                       "new_camera_matrix_alpha0": new_matrix,
                       "new_camera_matrix_alpha1": wide_matrix}.items():
        storage.write(key, value)
    storage.release()
    excluded = ", ".join(summary["excluded_images"]) or "none"
    table = "\n".join(f"| {row['image']} | {'yes' if row['used'] else 'no'} | "
                      f"{row['baseline_rms_px']:.4f} | {row['final_model_pose_rms_px']:.4f} | "
                      f"{row['held_out_rms_px']:.4f} |" if row['held_out_rms_px'] != "" else
                      f"| {row['image']} | no | {row['baseline_rms_px']:.4f} | "
                      f"{row['final_model_pose_rms_px']:.4f} | - |" for row in rows)
    matrix_text = "\n".join("  [" + ", ".join(f"{v:.9f}" for v in row) + "]" for row in matrix)
    distortion_text = ", ".join(f"{v:.10f}" for v in distortion.ravel())
    report = f"""# Color camera calibration

Calibrated {int(retained.sum())} of {len(files)} color images at **{size[0]} x {size[1]} pixels**
using OpenCV {cv2.__version__}, a **9 x 6 inner-corner board** (10 x 7 squares),
and the user-confirmed **{args.square_size_mm:g} mm** square side length.

## Result

- Final RMS reprojection error: **{rms:.6f} px** ({int(retained.sum()) * len(objects)} corners).
- Independently recomputed RMS with saved intrinsics and solvePnP poses: {verified_rms:.6f} px.
- Leave-one-view-out RMS: **{heldout_rms:.6f} px**; maximum: {max(heldout_errors.values()):.6f} px.
- All {len(views)} detected views baseline RMS: {baseline[0]:.6f} px.
- Excluded from final fit: **{excluded}**. All sources and their detections are preserved.

Camera matrix K (pixel units, for the original images):

```text
{matrix_text}
```

Distortion in OpenCV order **[k1, k2, p1, p2, k3]**:

```text
[{distortion_text}]
```

The standard five-coefficient pinhole model was fitted with flags=0 (fx, fy,
cx, cy and all five distortion coefficients free). Corners were found with
findChessboardCornersSB using NORMALIZE_IMAGE, EXHAUSTIVE and ACCURACY;
its returned subpixel corners were used directly. Calibration termination:
100 iterations or epsilon 1e-10.

## View selection and validation

One pass flags images above max(0.5 px, median + 3 * 1.4826 * MAD) on the
all-view baseline. Median={median:.6f}, MAD={mad:.6f}, cutoff={cutoff:.6f} px.
Filtering enabled: {not args.keep_all}. There is no repeated pruning.
The all-view matrix and distortion are also saved in calibration.json for comparison.
Removing the flagged view(s) changes [fx, fy, cx, cy] by
{summary['all_detected_views_baseline']['intrinsic_parameter_change_final_minus_baseline']} px;
this sensitivity exceeds the formal covariance estimate for some parameters.

Leave-one-view-out refits intrinsics on the other retained images, then fits only
the held-out board's pose. It checks consistency across views, with no independent
measurement of 3D accuracy. The JSON includes held-out errors, parameter ranges,
and OpenCV's formal intrinsic standard deviations.

| Image | Used | Baseline RMS px | Final model pose RMS px | Held-out RMS px |
|---|---|---:|---:|---:|
{table}

## Files and use

- calibration.yaml: OpenCV FileStorage matrices.
- calibration.json: full parameters, diagnostics, assumptions, source hashes, per-image board poses.
- calibration.npz: NumPy matrices and retained-view poses, translations in mm.
- all_views_calibration.npz: baseline fit using all detected views, with poses and errors.
- detected_points.npz: all image corners, 3D board points in mm, names, retained mask.
- per_image_errors.csv: all per-image errors.
- corners/: all images with detected inner corners.
- residuals/: projected points in green, observed-minus-projected arrows in red, enlarged 25 times.
- undistorted/: all color images corrected at full resolution using alpha=0.
- corner_detections.jpg, corner_coverage.png, undistortion_comparison.jpg: visual checks.

```python
import cv2
import numpy as np

cal = np.load("CameraCalibration/results/calibration.npz")
K = cal["camera_matrix"]
dist = cal["distortion_coefficients"]
image = cv2.imread("CameraCalibration/1_Color.png")
corrected = cv2.undistort(image, K, dist, None, cal["new_camera_matrix_alpha0"])
# Corrected images use new_camera_matrix_alpha0, rather than the original K.
```

From the project root, reproduce with:

```powershell
.\\.venv\\Scripts\\python.exe CameraCalibration\\calibrate.py --square-size-mm {args.square_size_mm:g}
```

Dependencies are numpy and opencv-python. To keep all detected views for a
comparison run, add `--keep-all --output CameraCalibration/results_all_views`.

## Interpretation

{chr(10).join('- ' + text for text in summary['limitations'])}
- Board poses describe each photographed board relative to this camera; they do not establish a fixed printer coordinate system.
- Corner coverage is shown separately; increasing board coverage near the frame edges would strengthen edge distortion estimates.
- The cause of image-level outliers was not established; inspect the saved corner and residual overlays.

Reference: [OpenCV calibration API]({DOCUMENTATION}).
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), "images_used": int(retained.sum()),
                      "excluded": summary["excluded_images"], "rms_px": float(rms),
                      "heldout_rms_px": heldout_rms, "camera_matrix": matrix.tolist(),
                      "distortion": distortion.ravel().tolist()}, indent=2))


if __name__ == "__main__":
    main()

# Color camera calibration

Calibrated 24 of 25 color images at **848 x 480 pixels**
using OpenCV 5.0.0, a **9 x 6 inner-corner board** (10 x 7 squares),
and the user-confirmed **25 mm** square side length.

## Result

- Final RMS reprojection error: **0.279979 px** (1296 corners).
- Independently recomputed RMS with saved intrinsics and solvePnP poses: 0.279979 px.
- Leave-one-view-out RMS: **0.292120 px**; maximum: 0.474511 px.
- All 25 detected views baseline RMS: 0.323066 px.
- Excluded from final fit: **25_Color.png**. All sources and their detections are preserved.

Camera matrix K (pixel units, for the original images):

```text
  [456.419887170, 0.000000000, 431.844190828]
  [0.000000000, 455.612709832, 225.579418837]
  [0.000000000, 0.000000000, 1.000000000]
```

Distortion in OpenCV order **[k1, k2, p1, p2, k3]**:

```text
[-0.0291779262, 0.0666146993, -0.0007653424, 0.0033625753, -0.0294367997]
```

The standard five-coefficient pinhole model was fitted with flags=0 (fx, fy,
cx, cy and all five distortion coefficients free). Corners were found with
findChessboardCornersSB using NORMALIZE_IMAGE, EXHAUSTIVE and ACCURACY;
its returned subpixel corners were used directly. Calibration termination:
100 iterations or epsilon 1e-10.

## View selection and validation

One pass flags images above max(0.5 px, median + 3 * 1.4826 * MAD) on the
all-view baseline. Median=0.242364, MAD=0.051121, cutoff=0.500000 px.
Filtering enabled: True. There is no repeated pruning.
The all-view matrix and distortion are also saved in calibration.json for comparison.
Removing the flagged view(s) changes [fx, fy, cx, cy] by
[2.852743680804224, 2.716294441146772, 0.877037387715518, -1.9874058168847455] px;
this sensitivity exceeds the formal covariance estimate for some parameters.

Leave-one-view-out refits intrinsics on the other retained images, then fits only
the held-out board's pose. It checks consistency across views, with no independent
measurement of 3D accuracy. The JSON includes held-out errors, parameter ranges,
and OpenCV's formal intrinsic standard deviations.

| Image | Used | Baseline RMS px | Final model pose RMS px | Held-out RMS px |
|---|---|---:|---:|---:|
| 1_Color.png | yes | 0.2231 | 0.2158 | 0.2191 |
| 2_Color.png | yes | 0.2187 | 0.2171 | 0.2283 |
| 3_Color.png | yes | 0.1823 | 0.1854 | 0.1897 |
| 4_Color.png | yes | 0.1915 | 0.1835 | 0.1913 |
| 5_Color.png | yes | 0.2351 | 0.2416 | 0.2493 |
| 6_Color.png | yes | 0.3585 | 0.3505 | 0.3675 |
| 7_Color.png | yes | 0.1912 | 0.1996 | 0.2066 |
| 8_Color.png | yes | 0.2200 | 0.2230 | 0.2241 |
| 9_Color.png | yes | 0.3764 | 0.3818 | 0.3966 |
| 10_Color.png | yes | 0.1738 | 0.1717 | 0.1782 |
| 11_Color.png | yes | 0.3402 | 0.3486 | 0.3608 |
| 12_Color.png | yes | 0.1971 | 0.1975 | 0.2047 |
| 13_Color.png | yes | 0.3342 | 0.3345 | 0.3490 |
| 14_Color.png | yes | 0.2714 | 0.2639 | 0.2745 |
| 15_Color.png | yes | 0.1676 | 0.1642 | 0.1665 |
| 16_Color.png | yes | 0.2611 | 0.2384 | 0.2498 |
| 17_Color.png | yes | 0.3695 | 0.3872 | 0.4059 |
| 18_Color.png | yes | 0.2503 | 0.2334 | 0.2385 |
| 19_Color.png | yes | 0.2424 | 0.2214 | 0.2334 |
| 20_Color.png | yes | 0.2202 | 0.2118 | 0.2160 |
| 21_Color.png | yes | 0.2382 | 0.2418 | 0.3055 |
| 22_Color.png | yes | 0.2958 | 0.2921 | 0.2998 |
| 23_Color.png | yes | 0.4485 | 0.4417 | 0.4535 |
| 24_Color.png | yes | 0.4652 | 0.4589 | 0.4745 |
| 25_Color.png | no | 0.8347 | 0.8757 | - |

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
.\.venv\Scripts\python.exe CameraCalibration\calibrate.py --square-size-mm 25
```

Dependencies are numpy and opencv-python. To keep all detected views for a
comparison run, add `--keep-all --output CameraCalibration/results_all_views`.

## Interpretation

- Color images only; does not calibrate the depth stream, color-depth registration, or camera-to-printer transform.
- Reprojection RMS is a pixel fitting error, not a millimeter accuracy measurement.
- The model assumes a flat rigid target with accurately square 25 mm cells; target deformation and motion can bias parameters.
- Use with the same stream, resolution, crop, and focus; edge regions outside observed corners are less constrained.
- Formal OpenCV standard deviations do not include target imperfections or other systematic errors.
- Board poses describe each photographed board relative to this camera; they do not establish a fixed printer coordinate system.
- Corner coverage is shown separately; increasing board coverage near the frame edges would strengthen edge distortion estimates.
- The cause of image-level outliers was not established; inspect the saved corner and residual overlays.

Reference: [OpenCV calibration API](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html).

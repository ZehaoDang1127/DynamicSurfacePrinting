# First physical step toward a camera-aligned stationary line

Measure one independent visible mark near printer X180/Y83. This point must not
be one of the four markers used for the fit. Its measured XYZ and its observed
depth will test the provisional registration without changing that registration.
The intended later line is X170–190 at Y83; its nozzle height and extrusion recipe
remain unassigned until surface height and clearance are established.

## Measure the fifth point

Use a small flat ink mark on the actual stationary printing surface, keeping the
camera and fixture fixed. Use the same verified printer coordinate frame as the
four measured markers. A coordinate reference lost since those measurements must
be re-established and checked before collecting another point.

With a cold, clean nozzle, align its XY with the mark. Establish the actual mark's
surface Z by an independent measurement that does not deflect the sheet. If using
a feeler gauge of measured thickness t, mark Z is the nozzle reading minus t;
merely hovering above the mark does not measure surface Z. Remove the gauge
before camera capture. Record the printer and G-code coordinate values so offsets
can be reconciled if they differ. A stationary readout can be obtained with:

```gcode
M400
GET_POSITION
```

Do not infer the fifth point's Z by interpolating the first four markers or by
reading the camera being checked. Marker-top height and bare-sheet height can
differ; the later printing gap is relative to the actual printing surface.

## Capture and compare on the Pi

Use the existing Pi D405 environment. The updated bundle adds OpenCV-based point
selection; install the project calibration dependencies in that environment:

```bash
python -m pip install -r requirements-calibration.txt
```

The calibration tool uses the already-installed `pyrealsense2`; it remains a
separate Pi runtime prerequisite. Stop other D405 streams, then run from the
extracted bundle directory:

```bash
python -m CameraCalibration.metric_check capture \
  --camera-config output/integrated_setup_20261025/camera.json \
  --output output/p5_capture_01
```

This acquires matching 848 x 480 color and depth frames and saves raw color,
depth-camera XYZ and SDK color-pixel correspondences for 15 stationary frames.
The camera stream is closed before any point-selection window opens. The SDK
keeps the factory inverse-distortion model in its own deprojection implementation;
it is not passed to an OpenCV forward-distortion model.
[SDK point-cloud mapping](https://realsenseai.github.io/librealsense/python_docs/_generated/pyrealsense2.pointcloud.html).

Edit `output/stationary_line_setup_20261025/p5.json`, filling `printer_xyz_mm`
with the actual measured XYZ. Leave `image_uv_px` null to select the mark in a
window during the next command. Then run:

```bash
python -m CameraCalibration.metric_check check \
  --capture output/p5_capture_01 \
  --registration output/integrated_setup_20261025/depth_points_registration/transform.json \
  --point output/stationary_line_setup_20261025/p5.json \
  --output output/p5_check_01
```

Left-click the fifth mark's center and press Enter. U/right-click undoes the
selection; Esc cancels. For a headless Pi, copy the capture folder to the laptop
and run selection/check there with the same registration and measurement file.
An explicitly known raw pixel can also be supplied using `--uv U V`.

The tool saves `point_error.json`, `point_selected.json` and `selected_point.png`.
`error_xyz_mm` means predicted printer XYZ minus independently measured XYZ;
`position_error_mm` is the 3D distance and `absolute_z_error_mm` is the vertical
error. Depth must actually be present in at least ten frames; missing depth is
not filled. Camera identity, image identity and transform rigidity are checked.

Without `--max-error-mm`, the tool reports the measurement without a pass/fail
claim. A chosen point-specific limit may be provided explicitly. Passing one
point does not qualify the whole working area and never enables hardware motion.
The check does not refit, overwrite, or silently correct the registration.

## After the measurement

If the independent error is unacceptable for the intended nozzle gap, improve
the measured correspondences and verify additional locations before proceeding.
When alignment is supported, measure the bare-sheet height along the entire
20 mm line, establish approach/travel clearance above the full fixture envelope,
and confirm the material recipe. The preview request records unknown heights,
gap and recipe as null rather than assuming values from the marker tops.

The next printer trial is a cold elevated positioning check, followed by the
short stationary deposition line. Camera access and point checks send no printer
commands. Existing dynamic `print` restrictions remain in effect.

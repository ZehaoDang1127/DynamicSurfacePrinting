# Dynamic Surface Printer — height-map MVP

This repository currently implements the first milestone from
[`PROJECT_OUTLINE.md`](PROJECT_OUTLINE.md): read timestamped depth frames from an
Intel RealSense D405 and turn each frame into a regular surface height map.
Prediction and printer motion are intentionally outside this MVP.

The output for every frame contains:

- surface height in millimetres on a fixed XY grid;
- a validity mask and the number of camera points contributing to each cell;
- the camera timestamp and a host monotonic timestamp;
- optional raw depth, useful for replay and later calibration work.

## 1. Install

Use Python 3.10 or newer. Intel's RealSense runtime/driver must be installed and
the D405 must be visible to the RealSense Viewer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`pyrealsense2` is only needed for a physical camera. The synthetic verification
path only needs NumPy.

## 2. Verify the geometry without hardware

```powershell
python -m surface_mapper --source synthetic --frames 5 --output output/synthetic
python -m unittest discover -s tests -v
```

The capture command writes one compressed `frame_*.npz` per frame, a
`manifest.json`, and `latest_height_mm.csv`.

## 3. Capture from the D405

```powershell
python -m surface_mapper `
  --config configs/d405_camera_frame.json `
  --frames 300 `
  --output output/d405_first_capture `
  --save-depth
```

Use `--frames 0` to run until `Ctrl+C`. The console reports valid grid coverage,
height range, timestamps, and processing rate while capturing.

## Coordinate calibration

The example configuration deliberately uses an identity transform and names the
resulting frame `camera`. RealSense camera coordinates are +X right, +Y down,
and +Z away from the camera. With the identity transform, the reported “height”
is therefore camera Z (depth), not printer Z.

Before using a height map for printer motion, replace `camera_to_output` with the
measured rigid camera-to-printer transform. The code first converts camera
points from metres to millimetres, then applies:

```text
p_output_mm = R * p_camera_mm + t_mm
```

The last row must be `[0, 0, 0, 1]`; the upper-left 3×3 block is the rotation,
and the last column is translation in millimetres. Set `frame_name` to `printer`
only after that calibration has been measured and checked at multiple known XYZ
locations.

## Configuration fields

- `camera`: stream size, frame rate, warm-up count, timeout, and optional serial.
- `grid`: XY bounds, cell size, accepted Z range, aggregation method, and minimum
  points per valid cell.
- `camera_to_output`: calibrated 4×4 transform described above.
- `frame_name`: label stored in the session manifest.

`mean`, `min`, and `max` height aggregation are supported. `mean` is the initial
choice for a single visible, gently sloped surface. Keep the Z limits tight
enough to reject the background and fixtures.

## Reading a recorded frame

```python
import numpy as np

with np.load("output/d405_first_capture/frame_000001.npz") as frame:
    height_mm = frame["height_mm"]       # float32; invalid cells are NaN
    valid = frame["valid"]               # bool
    point_count = frame["point_count"]   # uint32
    camera_timestamp_ms = float(frame["camera_timestamp_ms"])
```

This is a 2.5D height-field representation: it assumes one printable surface
height for each output XY cell. Overhangs, hidden surfaces, and camera
occlusions cannot be represented as a single height map and remain out of scope
for the first prototype.

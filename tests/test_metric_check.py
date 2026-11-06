"""Independent point-check metrology and provenance tests, without devices."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from CameraCalibration.metric_check import check, compare_point, deproject_color, main, sample_point


def inputs():
    metadata = {"schema_version": 1, "serial": "test_device", "image_size": [100, 100],
        "depth_xyz_units": "m", "depth_to_color_units": "mm",
        "depth_to_color_mm": np.eye(4).tolist(),
        "color": {"width": 100, "height": 100, "fx": 100., "fy": 100., "ppx": 50., "ppy": 50.,
                  "model": "distortion.none", "coeffs": [0., 0., 0., 0., 0.]}}
    xyz = np.zeros((15, 3, 3))
    xyz[:, :, 2] = 1.
    uv = np.full((15, 3, 2), 50.)
    transform = np.eye(4)
    transform[:3, 3] = [10., 20., 30.]
    registration = {"units": "mm", "output_frame": "printer", "device_serial": "test_device",
                    "depth_camera_to_printer": transform.tolist(),
                    "measurements": [{"printer_xyz_mm": [100., 100., 150.]}]}
    return xyz, uv, metadata, registration


def pinhole(k, uv, z):
    return np.array([(uv[0] - k["ppx"]) * z / k["fx"],
                     (uv[1] - k["ppy"]) * z / k["fy"], z])


class MetricCheckTests(unittest.TestCase):
    def test_units_and_transform_direction(self):
        xyz, uv, metadata, registration = inputs()
        metadata["depth_to_color_mm"][0][3] = 10.
        metadata["depth_to_color_mm"][1][3] = 20.
        metadata["depth_to_color_mm"][2][3] = 30.
        center = [50 + 100 * 10 / 1030, 50 + 100 * 20 / 1030]
        uv[:] = center
        point, stats = sample_point(xyz, uv, metadata, center, deproject=pinhole)
        np.testing.assert_allclose(point, [0., 0., 1000.], atol=1e-10)
        report = compare_point(registration, [10., 20., 1030.], point, "test_device", .1)
        self.assertEqual(report["status"], "within_point_tolerance")
        self.assertLess(report["position_error_mm"], 1e-9)
        self.assertEqual(stats["frames_with_depth"], 15)
        self.assertFalse(report["motion_enabled"])

    def test_signed_axis_errors_and_explicit_threshold(self):
        _, _, _, registration = inputs()
        report = compare_point(registration, [9.7, 20.4, 1029.], [0., 0., 1000.], "test_device", .5)
        np.testing.assert_allclose(report["error_xyz_mm"], [.3, -.4, 1.])
        self.assertAlmostEqual(report["position_error_mm"], np.sqrt(1.25))
        self.assertEqual(report["status"], "outside_point_tolerance")
        self.assertFalse(report["motion_enabled"])

    def test_missing_depth_is_not_filled(self):
        xyz, uv, metadata, _ = inputs()
        xyz[:6, :, 2] = 0.
        with self.assertRaisesRegex(ValueError, "insufficient observed depth"):
            sample_point(xyz, uv, metadata, [50., 50.], deproject=pinhole)

    def test_reusing_fit_point_is_rejected(self):
        _, _, _, registration = inputs()
        with self.assertRaisesRegex(ValueError, "duplicates a fitted marker"):
            compare_point(registration, [100., 100., 150.], [0., 0., 1000.], "test_device")

    def test_wrong_camera_and_nonrigid_transform_rejected(self):
        _, _, _, registration = inputs()
        with self.assertRaisesRegex(ValueError, "physical camera"):
            compare_point(registration, [10., 20., 1030.], [0., 0., 1000.], "different_device")
        registration["depth_camera_to_printer"][0][0] = 2.
        with self.assertRaisesRegex(ValueError, "proper rigid transform"):
            compare_point(registration, [10., 20., 1030.], [0., 0., 1000.], "test_device")

    def test_capture_provenance_and_saved_result(self):
        xyz, uv, metadata, registration = inputs()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cap = root / "capture"
            cap.mkdir()
            cv2.imwrite(str(cap / "color.png"), np.zeros((100, 100, 3), np.uint8))
            metadata["color_image_sha256"] = hashlib.sha256((cap / "color.png").read_bytes()).hexdigest()
            (cap / "metadata.json").write_text(json.dumps(metadata))
            np.savez_compressed(cap / "paired_depth.npz", xyz_depth_m=xyz, color_uv_px=uv)
            reg = root / "registration.json"
            reg.write_text(json.dumps(registration))
            point = root / "point.json"
            point.write_text(json.dumps({"schema_version": 1, "coordinate_frame": "printer",
                "printer_xyz_mm": [10., 20., 1030.], "image_uv_px": [50., 50.]}))
            with patch("CameraCalibration.metric_check.sample_point", return_value=(np.array([0., 0., 1000.]),
                         {"frames_with_depth": 15})):
                result = check(cap, reg, point, root / "result")
            self.assertEqual(result["status"], "measured")
            self.assertTrue((root / "result/point_error.json").exists())
            self.assertFalse(result["motion_enabled"])
            (cap / "color.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "image changed"):
                check(cap, reg, point, root / "altered")

    def test_template_does_not_invent_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "p5.json"
            with patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main(["init", "--output", str(file)]), 0)
            point = json.loads(file.read_text())
            self.assertIsNone(point["printer_xyz_mm"])
            self.assertIsNone(point["image_uv_px"])

    def test_sdk_deprojection_without_device_access(self):
        _, _, metadata, _ = inputs()
        try:
            result = deproject_color(metadata["color"], [50., 50.], 1000.)
        except ImportError:
            self.skipTest("pyrealsense2 not installed")
        np.testing.assert_allclose(result, [0., 0., 1000.], atol=1e-5)


if __name__ == "__main__":
    unittest.main()

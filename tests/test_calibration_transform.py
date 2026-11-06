"""Registration checks with an independently specified tilted camera pose."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

try:
    import cv2
except ImportError as exc:
    raise unittest.SkipTest("install requirements-calibration.txt to test PnP") from exc
import numpy as np

from CameraCalibration.transform import (compose_depth_transform, demo, inverse_rigid,
                                         load_intrinsics, main, point_template,
                                         rigid_matrix, solve_pose, validate_points, write_json)


ROOT = Path(__file__).resolve().parents[1]
INTRINSICS = ROOT / "CameraCalibration/results/calibration.json"


def fixture():
    matrix, distortion, size = load_intrinsics(INTRINSICS)
    xyz = np.array([[100, 100, 20], [140, 100, 20], [140, 140, 20], [100, 140, 20]], dtype=float)
    angle = np.deg2rad(20)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, -s], [0, -1, 0], [-s, 0, -c]])
    translation = -rotation @ np.array([120, 120, 300])
    uv, _ = cv2.projectPoints(xyz, cv2.Rodrigues(rotation)[0], translation, matrix, distortion)
    return matrix, distortion, size, xyz, uv.reshape(-1, 2), rotation, translation


class TransformTests(unittest.TestCase):
    def test_tilted_pose_inverse_and_metric_round_trip(self):
        matrix, dist, _, xyz, uv, rotation, translation = fixture()
        result = solve_pose(xyz, uv, matrix, dist)
        self.assertEqual(result["status"], "solved")
        forward = np.array(result["printer_to_camera"])
        backward = np.array(result["camera_to_printer"])
        np.testing.assert_allclose(forward[:3, :3], rotation, atol=1e-7)
        np.testing.assert_allclose(forward[:3, 3], translation, atol=1e-6)
        np.testing.assert_allclose(forward @ backward, np.eye(4), atol=1e-8)
        camera_xyz = xyz @ rotation.T + translation
        reconstructed = camera_xyz @ backward[:3, :3].T + backward[:3, 3]
        np.testing.assert_allclose(reconstructed, xyz, atol=1e-6)
        raised = xyz + [0, 0, 5]
        camera_raised = raised @ rotation.T + translation
        printer_raised = camera_raised @ backward[:3, :3].T + backward[:3, 3]
        np.testing.assert_allclose(printer_raised - reconstructed, np.tile([0, 0, 5], (4, 1)), atol=1e-6)

    def test_ambiguous_planar_pose_requires_selection(self):
        matrix, dist, _, xyz, uv, _, _ = fixture()
        # Near-frontal noisy observations have two genuinely distinct basins.
        rotation = cv2.Rodrigues(np.array([np.pi-.03, 0., 0.]))[0]
        translation = -rotation @ np.array([120., 120., 300.])
        uv = cv2.projectPoints(xyz, cv2.Rodrigues(rotation)[0], translation, matrix, dist)[0].reshape(-1, 2)
        uv += np.random.default_rng(15).normal(0, .3, (4, 2))
        result = solve_pose(xyz, uv, matrix, dist)
        self.assertGreaterEqual(len(result["candidates"]), 2)
        self.assertEqual(result["status"], "needs_pose_selection")
        self.assertIsNone(result["camera_to_printer"])
        selected = solve_pose(xyz, uv, matrix, dist, pose_index=0)
        self.assertEqual(selected["status"], "solved")

    def test_large_reprojection_error_is_not_accepted(self):
        matrix, dist, _, xyz, uv, _, _ = fixture()
        uv = uv.copy()
        uv[0] += [8, -5]
        result = solve_pose(xyz, uv, matrix, dist, max_error_px=.01, pose_index=0)
        self.assertEqual(result["status"], "reprojection_error_too_large")
        self.assertIsNone(result["printer_to_camera"])

    def test_template_missing_measurements_wrong_size_and_bad_geometry(self):
        matrix, dist, size, xyz, uv, _, _ = fixture()
        raw = point_template(size)
        with self.assertRaises(ValueError):
            validate_points(raw, size)
        for point, world, pixel in zip(raw["points"], xyz, uv):
            point["printer_xyz_mm"], point["image_uv_px"] = world.tolist(), pixel.tolist()
        validate_points(raw, size)
        with self.assertRaises(ValueError):
            validate_points(raw, (640, 480))
        raw["points"][0]["image_uv_px"] = [-1, 0]
        with self.assertRaises(ValueError):
            validate_points(raw, size)
        raw["points"][0]["image_uv_px"] = uv[0].tolist()
        for i, point in enumerate(raw["points"]):
            point["printer_xyz_mm"] = [i, i, 0]
        with self.assertRaisesRegex(ValueError, "collinear"):
            validate_points(raw, size)

    def test_depth_composition_checks_direction_and_units(self):
        _, _, _, _, _, rotation, translation = fixture()
        forward = np.eye(4)
        forward[:3, :3], forward[:3, 3] = rotation, translation
        backward = inverse_rigid(forward)
        depth_to_color = np.eye(4)
        depth_to_color[:3, 3] = [12, .5, 0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "depth_to_color.json"
            raw = {"from_frame": "depth", "to_frame": "color", "units": "mm", "matrix": depth_to_color.tolist()}
            write_json(path, raw)
            composed = compose_depth_transform(backward, "color", path)
            np.testing.assert_allclose(composed, backward @ depth_to_color)
            with self.assertRaises(ValueError):
                compose_depth_transform(backward, "depth", path)
            raw["units"] = "m"
            write_json(path, raw)
            with self.assertRaises(ValueError):
                compose_depth_transform(backward, "color", path)
        reflection = np.eye(4)
        reflection[0, 0] = -1
        with self.assertRaises(ValueError):
            rigid_matrix(reflection)

    def test_demo_marks_synthetic_and_never_exports_color_as_depth_config(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "demo"
            result = demo(INTRINSICS, output)
            self.assertEqual(result["status"], "solved")
            self.assertTrue(result["synthetic_demo"])
            self.assertIsNone(result["surface_mapper_update"])
            self.assertTrue((output / "solution/transform.json").is_file())
            with self.assertRaises(FileExistsError):
                demo(INTRINSICS, output)

    def test_cli_init_and_missing_data_fail_without_invented_pose(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            points = Path(folder) / "points.json"
            self.assertEqual(main(["init", "--output", str(points)]), 0)
            self.assertEqual(main(["solve", "--points", str(points), "--output", str(Path(folder)/"solve")]), 2)
            self.assertFalse((Path(folder)/"solve").exists())


if __name__ == "__main__":
    unittest.main()

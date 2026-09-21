import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from surface_mapper.config import AppConfig, CameraConfig, GridConfig, load_config
from surface_mapper.heightmap import HeightMapProjector, apply_rigid_transform


class TransformTests(unittest.TestCase):
    def test_applies_rotation_and_translation(self) -> None:
        matrix = np.array(
            [
                [0.0, -1.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = apply_rigid_transform(np.array([[1.0, 2.0, 3.0]]), matrix)
        np.testing.assert_allclose(result, [[8.0, 21.0, 33.0]])


class HeightMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = GridConfig(
            x_min_mm=0.0,
            x_max_mm=2.0,
            y_min_mm=0.0,
            y_max_mm=2.0,
            cell_size_mm=1.0,
            z_min_mm=5.0,
            z_max_mm=20.0,
            min_points_per_cell=2,
            aggregation="mean",
        )

    def test_projects_filters_and_averages(self) -> None:
        # Input is in metres. Two valid samples land in grid cell [y=0, x=0].
        points = np.array(
            [
                [0.0001, 0.0001, 0.010],
                [0.0009, 0.0009, 0.014],
                [0.0015, 0.0015, 0.009],  # only one point: below count threshold
                [0.0030, 0.0001, 0.010],  # outside XY grid
                [0.0001, 0.0001, 0.030],  # outside Z range
                [0.0, 0.0, 0.0],          # RealSense invalid point
            ],
            dtype=np.float32,
        )
        result = HeightMapProjector(self.grid, np.eye(4)).project(points)
        self.assertEqual(result.input_point_count, 6)
        self.assertEqual(result.accepted_point_count, 3)
        self.assertTrue(result.valid[0, 0])
        self.assertAlmostEqual(float(result.height_mm[0, 0]), 12.0, places=5)
        self.assertFalse(result.valid[1, 1])
        self.assertTrue(np.isnan(result.height_mm[1, 1]))
        self.assertEqual(int(result.point_count[1, 1]), 1)

    def test_min_and_max_aggregation(self) -> None:
        points = np.array(
            [[0.0001, 0.0001, 0.010], [0.0002, 0.0002, 0.014]],
            dtype=np.float32,
        )
        for aggregation, expected in (("min", 10.0), ("max", 14.0)):
            grid = GridConfig(**{**vars(self.grid), "aggregation": aggregation})
            result = HeightMapProjector(grid, np.eye(4)).project(points)
            self.assertAlmostEqual(float(result.height_mm[0, 0]), expected, places=5)


class ConfigTests(unittest.TestCase):
    def test_rejects_non_rigid_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a rotation"):
            AppConfig(
                "test",
                CameraConfig(),
                GridConfig(0, 1, 0, 1, 1, 0, 1),
                np.diag([2.0, 1.0, 1.0, 1.0]),
            )

    def test_loads_json_config(self) -> None:
        raw = {
            "frame_name": "test",
            "camera": {},
            "grid": {
                "x_min_mm": 0,
                "x_max_mm": 1,
                "y_min_mm": 0,
                "y_max_mm": 1,
                "cell_size_mm": 1,
                "z_min_mm": 0,
                "z_max_mm": 1,
            },
            "camera_to_output": np.eye(4).tolist(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.frame_name, "test")
        self.assertEqual(config.grid.width, 1)


if __name__ == "__main__":
    unittest.main()

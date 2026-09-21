import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from surface_mapper.camera import PointCloudFrame
from surface_mapper.config import AppConfig, CameraConfig, GridConfig
from surface_mapper.heightmap import HeightMap
from surface_mapper.recording import SessionWriter


class RecordingTests(unittest.TestCase):
    def test_writes_loadable_frame_and_complete_manifest(self) -> None:
        grid = GridConfig(0, 2, 0, 1, 1, 0, 20)
        config = AppConfig("test", CameraConfig(), grid, np.eye(4))
        source = PointCloudFrame(
            frame_number=7,
            camera_timestamp_ms=12.5,
            host_monotonic_ns=1234,
            timestamp_domain="test",
            points_camera_m=np.zeros((2, 3), dtype=np.float32),
            depth_m=np.array([[0.01, 0.02]], dtype=np.float32),
        )
        result = HeightMap(
            height_mm=np.array([[10.0, np.nan]], dtype=np.float32),
            valid=np.array([[True, False]]),
            point_count=np.array([[2, 0]], dtype=np.uint32),
            input_point_count=2,
            accepted_point_count=2,
        )

        with tempfile.TemporaryDirectory() as directory:
            writer = SessionWriter(directory, config, {"name": "test"}, save_depth=True)
            writer.start()
            frame_path = writer.write(source, result)
            writer.write_latest_csv(result)
            writer.finish()

            with np.load(frame_path) as recorded:
                np.testing.assert_array_equal(recorded["valid"], result.valid)
                np.testing.assert_allclose(recorded["depth_m"], source.depth_m)
                self.assertEqual(int(recorded["frame_number"]), 7)
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["frames"][0]["timestamp_domain"], "test")


if __name__ == "__main__":
    unittest.main()

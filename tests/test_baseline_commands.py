import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from prediction.baseline_commands import baseline_evaluate, baseline_live, baseline_predict
from prediction.data import load_session
from surface_mapper.camera import PointCloudFrame
from surface_mapper.config import AppConfig, CameraConfig, GridConfig
from surface_mapper.heightmap import HeightMap
from surface_mapper.recording import SessionWriter


def make_recording(root, name='recording', offsets_ms=None):
    config = AppConfig('camera', CameraConfig(width=3, height=2, fps=10, warmup_frames=0),
                       GridConfig(0, 3, 0, 2, 1, 0, 500), np.eye(4))
    writer = SessionWriter(root / name, config, {'name': 'quadratic test'})
    writer.start()
    for index in range(12):
        timestamp = index * 100.0 + (offsets_ms[index] if offsets_ms is not None else 0)
        seconds = timestamp / 1000.0
        height = np.full((2, 3), 200 + 5 * seconds + 10 * seconds**2, np.float32)
        valid = np.ones_like(height, dtype=bool)
        frame = PointCloudFrame(index, timestamp, int(timestamp * 1e6) + 1,
                                'synthetic', np.empty((0, 3), np.float32))
        writer.write(frame, HeightMap(height, valid, valid.astype(np.uint32), 6, 6))
    writer.finish()
    return root / name, config


class BaselineCommandTests(unittest.TestCase):
    def test_replay_and_evaluation_work_without_model_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, _ = make_recording(root)
            replay = baseline_predict(session, root / 'predictions')
            self.assertEqual(replay['forecast_count'], 8)
            self.assertTrue(replay['complete'])
            with np.load(root / 'predictions' / 'prediction_000000.npz') as forecast:
                self.assertEqual(float(forecast['source_timestamp_ms']), 400)
                self.assertEqual(float(forecast['target_timestamp_ms']), 500)
                np.testing.assert_allclose(forecast['prediction_mm'], 205, atol=2e-4)
                np.testing.assert_allclose(forecast['acceleration_mm_s2'], 20, atol=0.002)
                self.assertTrue(forecast['prediction_valid'].all())
            report = baseline_evaluate(session, root / 'evaluation')
            self.assertEqual(report['windows'], 7)
            self.assertLess(report['metrics']['all']['constant_acceleration']['mae_mm'], 2e-4)
            self.assertGreater(report['metrics']['all']['constant_velocity']['mae_mm'], 0.5)
            self.assertEqual(report['comparison_coverage'], 1)
            html = (root / 'evaluation' / 'report.html').read_text(encoding='utf-8')
            self.assertIn('every history observation valid', html)
            self.assertNotIn('__SUPPORT__', html)
            with self.assertRaises(FileExistsError):
                baseline_predict(session, root / 'predictions')

    def test_horizon_uses_requested_time_and_reports_observation_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, _ = make_recording(root)
            report = baseline_evaluate(session, root / 'evaluation', horizon_ms=90, target_tolerance_ms=11)
            self.assertEqual(report['observed_target_offset_from_nominal_ms']['mean'], 10)
            self.assertGreater(report['metrics']['all']['constant_acceleration']['mae_mm'], 0.1)
            with self.assertRaisesRegex(ValueError, 'no scorable forecasts'):
                baseline_evaluate(session, root / 'unmatched', horizon_ms=90, target_tolerance_ms=1)

    def test_live_records_input_and_forecasts_from_one_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = make_recording(root)
            camera_path = root / 'camera.json'
            camera_path.write_text(json.dumps(config.to_dict()), encoding='utf-8')
            manifest = baseline_live(camera_path, root / 'live', source_name='synthetic', frames=9,
                                     record_input=True)
            self.assertEqual(manifest['forecast_count'], 5)
            self.assertEqual(manifest['captured_frames'], 9)
            recorded = load_session(root / 'live' / 'observations')
            self.assertEqual(len(recorded.frames), 9)
            self.assertEqual(recorded.geometry, manifest['geometry'])
            report = baseline_evaluate(recorded.path, root / 'evaluation', target_tolerance_ms=1)
            self.assertEqual(report['windows'], 4)

    def test_save_depth_requires_input_recording_before_opening_camera(self):
        with patch('prediction.baseline_commands.RealSenseD405') as source:
            with self.assertRaisesRegex(ValueError, 'requires --record-input'):
                baseline_live('missing.json', 'unused', save_depth=True)
            source.assert_not_called()

    def test_recording_stops_before_restarted_counter_overwrites_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, config = make_recording(root)
            camera_path = root / 'camera.json'
            camera_path.write_text(json.dumps(config.to_dict()), encoding='utf-8')
            points = np.array([[0.0005, 0.0005, 0.2]], dtype=np.float32)
            first = PointCloudFrame(1, 100, 100_000_001, 'camera', points)
            restarted = PointCloudFrame(1, 200, 200_000_001, 'camera', points)
            with patch('prediction.baseline_commands.RealSenseD405') as factory:
                source = factory.return_value
                source.device_info = {'name': 'test camera'}
                source.capture.side_effect = [first, restarted]
                with self.assertRaisesRegex(RuntimeError, 'restarted'):
                    baseline_live(camera_path, root / 'live', frames=2, record_input=True)
                source.stop.assert_called_once()
            with np.load(root / 'live' / 'observations' / 'frame_000001.npz') as saved:
                self.assertEqual(float(saved['camera_timestamp_ms']), 100)
            manifest = json.loads((root / 'live' / 'observations' / 'manifest.json').read_text())
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['frame_count'], 1)

    def test_baseline_cli_runs_when_torch_import_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, _ = make_recording(root)
            script = '''import importlib.abc, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch deliberately blocked')
sys.meta_path.insert(0, BlockTorch())
from prediction.cli import main
raise SystemExit(main(['baseline-predict', '--session', sys.argv[1], '--output', sys.argv[2]]))
'''
            result = subprocess.run([sys.executable, '-c', script, str(session), str(root / 'prediction')],
                                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"forecasts": 8', result.stdout)


if __name__ == '__main__':
    unittest.main()

"""Checks for the commanded-motion delta evaluation (no torch needed)."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import evaluate_motion_delta as md

try:
    import torch
except ImportError:
    torch = None


def write_recording(maps_root, recordings_root, name, frames=240, shape=(8, 12), seed=0):
    """Recorder-format height maps plus frames.csv/trajectory.csv for a synthetic servo motion."""
    rng = np.random.default_rng(seed)
    trajectory = np.column_stack([np.arange(0, 10, 0.02), 150 + 60 * np.sin(np.arange(0, 10, 0.02) * 2.0 + seed)])
    rec = recordings_root / name
    rec.mkdir(parents=True)
    with (rec / 'trajectory.csv').open('w') as handle:
        handle.write('elapsed_s,commanded_angle_deg,pulse_us\n')
        for t, a in trajectory:
            handle.write(f'{t:.9f},{a:.9f},0\n')
    directory = maps_root / name
    directory.mkdir(parents=True)
    rows, entries = [], []
    for i in range(frames):
        elapsed = i / 30
        angle = np.interp(elapsed, trajectory[:, 0], trajectory[:, 1])
        height = (300 - 0.1 * angle + rng.normal(0, 0.2, shape)).astype(np.float32)
        valid = np.ones(shape, bool)
        meta = {'frame_number': i + 10, 'camera_timestamp_ms': 1000 + i * 1000 / 30,
                'host_monotonic_ns': (i + 1) * 33_333_333}
        filename = f'frame_{i + 10:06d}.npz'
        np.savez_compressed(directory / filename, height_mm=height, valid=valid,
                            point_count=valid.astype(np.uint32), **meta)
        entries.append({'file': filename, 'timestamp_domain': 'global', 'valid_fraction': 1.0, **meta})
        rows.append(f'{i},c,d,{elapsed:.9f},0,0,0,{i + 10}')
    (rec / 'frames.csv').write_text('index,color_file,depth_file,elapsed_s,host_monotonic_ns,host_unix_ns,'
                                    'color_frame_number,depth_frame_number\n' + '\n'.join(rows) + '\n')
    config = {'frame_name': 'camera', 'camera': {'width': 640, 'height': 480, 'fps': 30},
              'grid': {'x_min_mm': 0, 'x_max_mm': shape[1], 'y_min_mm': 0, 'y_max_mm': shape[0],
                       'cell_size_mm': 1, 'z_min_mm': 0, 'z_max_mm': 1000},
              'camera_to_output': np.eye(4).tolist()}
    manifest = {'schema_version': 1, 'complete': True, 'config': config,
                'grid': {'shape': list(shape), 'x_centers_mm': [c + .5 for c in range(shape[1])],
                         'y_centers_mm': [c + .5 for c in range(shape[0])]},
                'frame_count': frames, 'frames': entries}
    (directory / 'manifest.json').write_text(json.dumps(manifest))


class MotionDeltaTests(unittest.TestCase):
    def test_fit_map_recovers_linear_response(self):
        rng = np.random.default_rng(0)
        theta = rng.uniform(60, 240, 2000)
        slopes = np.linspace(-0.15, -0.05, 6)
        heights = 300 + theta[:, None] * slopes[None] + rng.normal(0, 0.5, (2000, 6))
        valid = np.ones_like(heights, bool)
        valid[:, 5] = False                      # never valid: unusable
        coef, usable, r2 = md.fit_map(theta, heights, valid)
        self.assertEqual(usable.tolist(), [True] * 5 + [False])
        surface = md.design(np.array([100., 110.])) @ coef.T
        np.testing.assert_allclose(surface[1, :5] - surface[0, :5], 10 * slopes[:5], atol=0.05)
        self.assertTrue(np.all(r2[:5] > 0.9))

    def test_compare_scores_model_against_persistence(self):
        ref = np.linspace(-3, 3, 50)
        perfect = md.DeltaStats.compare(ref, ref, active_mm=1.0)
        self.assertAlmostEqual(perfect['pearson_r'], 1.0)
        self.assertAlmostEqual(perfect['gain'], 1.0)
        self.assertAlmostEqual(perfect['mae_model_mm'], 0.0)
        self.assertGreater(perfect['mae_persistence_mm'], 1.0)
        self.assertEqual(perfect['active']['sign_agreement'], 1.0)
        still = md.DeltaStats.compare(np.zeros_like(ref) + 1e-9 * ref, ref, active_mm=1.0)
        self.assertAlmostEqual(still['mae_model_mm'], still['mae_persistence_mm'], places=6)

    def test_real_head_uses_saved_steps_and_designed_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pred_dir, prep_dir = root / 'pred', root / 'prep'
            pred_dir.mkdir(); prep_dir.mkdir()
            cells = 23 * 45
            # Depth falls 0.1 mm per degree everywhere; lag zero.
            coef = np.zeros((cells, md.DEGREE + 1))
            coef[:, 0] = 300 - 0.1 * md.CENTER_DEG
            coef[:, 1] = -0.1 * md.HALF_RANGE_DEG
            np.savez_compressed(root / 'cal.npz', coefficients=coef, usable=np.ones(cells, bool), lag_ms=0.0)
            name = '21_amplitude_beats'
            trajectory = md.designed_trajectory(name)
            t = np.arange(0, 28, 1 / 30)
            theta = np.interp(t, trajectory[:, 0], trajectory[:, 1])
            surface = (300 - 0.1 * theta)[:, None, None] * np.ones((1, 23, 45))
            hidden = np.zeros((23, 45), bool); hidden[:, :30] = True
            np.savez_compressed(prep_dir / f'{name}.npz', blocked_elapsed_s=t, conservative_hidden=hidden,
                                blocked_valid=np.ones(surface.shape, bool),
                                repeat_height_mm=surface.astype(np.float32),
                                repeat_valid=np.ones(surface.shape, bool), reference_index=np.arange(len(t)))
            origins = np.arange(24, len(t) - 30)
            steps = np.array([6, 30])
            future = np.stack([surface[origins + k] for k in steps], axis=1).astype(np.float32)
            np.savez_compressed(pred_dir / f'{name}.npz', origin_index=origins.astype(np.int32),
                                prediction_mm=surface[origins].astype(np.float32),
                                prediction_200ms_mm=surface[origins + 6].astype(np.float32),
                                prediction_steps=steps.astype(np.int32), prediction_future_mm=future)
            report = md.real_head(pred_dir, prep_dir, root / 'cal.npz', root / 'out')
            area = report['overall']['30']['hidden']['area_mean_vs_command']
            self.assertGreater(area['pearson_r'], 0.999)
            self.assertAlmostEqual(area['gain'], 1.0, places=3)
            self.assertLess(area['mae_model_mm'], 1e-3)
            self.assertTrue((root / 'out' / 'motion_delta_real_head.json').exists())
            json.loads((root / 'out' / 'motion_delta_real_head.json').read_text())


@unittest.skipIf(torch is None, 'optional prediction dependencies are not installed')
class MotionDeltaPipelineTests(unittest.TestCase):
    def test_calibrate_train_and_score_synthetic_test(self):
        from prediction import occlusion
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            maps, recordings = root / 'maps', root / 'recordings'
            for i, name in enumerate(('01_train', '02_validation', '03_test')):
                write_recording(maps, recordings, name, seed=i)
            (maps / 'splits.json').write_text(json.dumps(
                {'train': ['01_train'], 'validation': ['02_validation'], 'test': ['03_test']}))
            cal = md.calibrate(maps / 'splits.json', recordings, root / 'cal', [0, 100])
            self.assertEqual(cal['lag_ms'], 0.0)
            self.assertGreater(cal['r2_median'], 0.9)
            config = {'history': 4, 'horizon': 2, 'base_channels': 2, 'epochs': 1, 'batch_size': 4,
                      'patience': 1, 'expected_dt_ms': 1000 / 30, 'occlusion_shape': 'mixed',
                      'train_window_stride': 4}
            occlusion.train(maps / 'splits.json', config, root / 'run', 'cpu')
            report = md.synthetic(root / 'run' / 'best.pt', maps / 'splits.json', recordings,
                                  root / 'cal' / 'calibration.npz', root / 'out', [1, 3], 'cpu', None)
            self.assertGreater(report['windows'], 50)
            lead = report['leads']['3']['hidden']
            # The measured change follows the commanded change closely in this noise-light fixture.
            self.assertGreater(lead['measured_vs_command_area_mean']['pearson_r'], 0.9)
            for key in ('area_mean_vs_command', 'per_cell_vs_command', 'per_cell_vs_measured'):
                self.assertIn('mae_persistence_mm', lead[key])


if __name__ == '__main__':
    unittest.main()

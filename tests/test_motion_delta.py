"""Checks for the commanded-motion delta evaluation (no torch needed)."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools import evaluate_motion_delta as md


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


if __name__ == '__main__':
    unittest.main()

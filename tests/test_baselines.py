"""Physical trajectory, support and streaming checks for NumPy baselines."""

import unittest

import numpy as np

from prediction.baselines import (
    BaselineConfig,
    StreamingBaselinePredictor,
    fit_kinematic_baselines,
)


class BaselineFitTests(unittest.TestCase):
    def test_three_equally_spaced_frames_predict_exact_second_difference(self):
        heights = np.array([[[1., 10.]], [[4., 8.]], [[9., 5.]]])
        result = fit_kinematic_baselines(heights, np.ones_like(heights, bool), [0, 1000, 2000], 1000)
        estimate = result['constant_acceleration']
        expected = 3 * heights[-1] - 3 * heights[-2] + heights[-3]
        np.testing.assert_allclose(estimate.prediction_mm, expected)
        np.testing.assert_allclose(estimate.velocity_mm_s, [[6., -3.5]])
        np.testing.assert_allclose(estimate.acceleration_mm_s2, [[2., -1.]])
        self.assertEqual(estimate.prediction_mm.dtype, np.float32)
        self.assertEqual(estimate.prediction_valid.dtype, np.bool_)

    def test_irregular_history_arbitrary_horizon_and_physical_units(self):
        seconds = np.array([-.20, -.11, -.07, -.02, 0.])
        position = np.array([[150., 180.], [110., 200.]])
        velocity = np.array([[20., -15.], [0., 6.]])
        acceleration = np.array([[80., -30.], [25., 0.]])
        heights = (position + seconds[:, None, None] * velocity
                   + .5 * seconds[:, None, None] ** 2 * acceleration)
        estimate = fit_kinematic_baselines(
            heights, np.ones_like(heights, bool), 7500 + seconds * 1000, 80,
        )['constant_acceleration']
        np.testing.assert_allclose(estimate.prediction_mm, position + .08 * velocity + .5 * .08 ** 2 * acceleration, atol=1e-5)
        np.testing.assert_allclose(estimate.velocity_mm_s, velocity, atol=1e-5)
        np.testing.assert_allclose(estimate.acceleration_mm_s2, acceleration, atol=1e-5)

    def test_fit_uses_fitted_current_height_instead_of_noisy_last_observation(self):
        seconds = np.arange(-2., 3.)
        # This residual is orthogonal to all quadratic fit columns.
        residual = np.array([1., -4., 6., -4., 1.])
        heights = (100 + 2 * seconds + 3 * seconds ** 2 + residual)[:, None, None]
        estimate = fit_kinematic_baselines(heights, np.ones_like(heights, bool), seconds * 1000, 1000)['constant_acceleration']
        self.assertAlmostEqual(float(estimate.prediction_mm[0, 0]), 133.)
        self.assertAlmostEqual(float(estimate.velocity_mm_s[0, 0]), 14.)
        self.assertAlmostEqual(float(estimate.acceleration_mm_s2[0, 0]), 6.)

    def test_linear_trajectory_has_zero_acceleration(self):
        times = np.array([0., 20., 65., 100., 170.])
        heights = (120. + 40. * times / 1000.)[:, None, None]
        estimates = fit_kinematic_baselines(heights, np.ones_like(heights, bool), times, 50.)
        for name in ('constant_velocity', 'constant_acceleration'):
            np.testing.assert_allclose(estimates[name].prediction_mm, [[128.8]], atol=1e-5)
            np.testing.assert_allclose(estimates[name].velocity_mm_s, [[40.]], atol=1e-5)
        np.testing.assert_allclose(estimates['constant_acceleration'].acceleration_mm_s2, [[0.]], atol=1e-5)

    def test_stationary_trajectory(self):
        heights = np.full((5, 2, 3), 150.)
        estimates = fit_kinematic_baselines(heights, np.ones_like(heights, bool), [0, 30, 60, 90, 120], 33.)
        for estimate in estimates.values():
            np.testing.assert_allclose(estimate.prediction_mm, 150.)
            self.assertTrue(estimate.prediction_valid.all())
        np.testing.assert_allclose(estimates['constant_acceleration'].acceleration_mm_s2, 0., atol=1e-5)

    def test_large_timestamp_offset_does_not_change_prediction(self):
        times = np.array([0., 25., 75., 100., 200.])
        heights = (100. + 3. * times / 1000. + 8. * (times / 1000.) ** 2)[:, None, None]
        normal = fit_kinematic_baselines(heights, np.ones_like(heights, bool), times, 25.)
        shifted = fit_kinematic_baselines(heights, np.ones_like(heights, bool), times + 1_700_000_000_000., 25.)
        for name in normal:
            np.testing.assert_array_equal(normal[name].prediction_mm, shifted[name].prediction_mm)

    def test_two_frame_history_keeps_velocity_evaluation_available(self):
        heights = np.array([[[10.]], [[12.]]])
        estimates = fit_kinematic_baselines(heights, np.ones_like(heights, bool), [0., 100.], 50.)
        self.assertEqual(set(estimates), {'persistence', 'constant_velocity'})
        np.testing.assert_allclose(estimates['constant_velocity'].prediction_mm, [[13.]])

    def test_missing_data_has_no_implicit_fallback(self):
        heights = np.arange(24., dtype=float).reshape(3, 2, 4)
        masks = np.ones_like(heights, dtype=bool)
        masks[0, 0, 0] = False
        masks[-1, 0, 1] = False
        heights[1, 0, 2] = np.nan
        heights[1, 0, 3] = np.inf
        estimates = fit_kinematic_baselines(heights, masks, [0., 10., 20.], 10.)
        for name in ('constant_velocity', 'constant_acceleration'):
            estimate = estimates[name]
            self.assertFalse(estimate.prediction_valid[0].any())
            self.assertTrue(np.isnan(estimate.prediction_mm[0]).all())
            self.assertTrue(np.isnan(estimate.velocity_mm_s[0]).all())
            self.assertTrue(estimate.prediction_valid[1].all())
        latest = estimates['persistence']
        self.assertTrue(latest.prediction_valid[0, 0])
        self.assertFalse(latest.prediction_valid[0, 1])
        self.assertTrue(np.isnan(latest.prediction_mm[0, 1]))
        self.assertTrue(latest.prediction_valid[0, 2])

    def test_no_supported_cells_returns_nan_maps(self):
        heights = np.ones((5, 2, 3))
        estimates = fit_kinematic_baselines(heights, np.zeros_like(heights, bool), np.arange(5), 1.)
        for estimate in estimates.values():
            self.assertFalse(estimate.prediction_valid.any())
            self.assertTrue(np.isnan(estimate.prediction_mm).all())

    def test_invalid_fit_inputs(self):
        height = np.ones((3, 2, 2))
        mask = np.ones_like(height, bool)
        for timestamps in ([0, 0, 1], [0, 2, 1], [0, np.nan, 2], [0, np.inf, 2], [0, 1]):
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                fit_kinematic_baselines(height, mask, timestamps, 10.)
        for horizon in (0, -1, np.nan, np.inf):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                fit_kinematic_baselines(height, mask, [0, 1, 2], horizon)
        for heights, masks in ((height[0], mask[0]), (height, mask[:, :1]),
                               (height[:1], mask[:1]), (height[:, :0], mask[:, :0])):
            with self.subTest(shape=heights.shape), self.assertRaises(ValueError):
                fit_kinematic_baselines(heights, masks, [0, 1, 2], 10.)


class StreamingBaselineTests(unittest.TestCase):
    def make_predictor(self, **kwargs):
        values = dict(history=3, horizon_ms=10., max_gap_ms=30., min_valid_fraction=.1)
        values.update(kwargs)
        return StreamingBaselinePredictor(BaselineConfig(**values), {'frame': 'camera'}, (2, 2))

    def push(self, predictor, value, timestamp, domain='camera', frame_number=None, mask=None):
        height = np.full((2, 2), value, dtype=np.float32)
        if mask is None:
            mask = np.ones_like(height, bool)
        return predictor.push(height, mask, timestamp, domain, frame_number)

    def test_stream_matches_quadratic_and_target_timestamp(self):
        predictor = self.make_predictor()
        self.assertIsNone(self.push(predictor, 1., 100., frame_number=10))
        self.assertIsNone(self.push(predictor, 4., 110., frame_number=11))
        forecast = self.push(predictor, 9., 120., frame_number=12)
        np.testing.assert_allclose(forecast.prediction_mm, 16.)
        np.testing.assert_allclose(forecast.velocity_mm_s, 600.)
        np.testing.assert_allclose(forecast.acceleration_mm_s2, 20000.)
        self.assertEqual(forecast.source_timestamp_ms, 120.)
        self.assertEqual(forecast.target_timestamp_ms, 130.)
        self.assertEqual(forecast.timestamp_domain, 'camera')
        self.assertTrue(forecast.prediction_valid.all())
        self.assertGreaterEqual(forecast.inference_ms, 0.)

    def test_input_buffers_are_copied_and_prior_forecast_is_causal(self):
        predictor = self.make_predictor()
        height = np.ones((2, 2), np.float32)
        mask = np.ones((2, 2), bool)
        predictor.push(height, mask, 0., 'camera')
        height[:] = 999.
        mask[:] = False
        self.push(predictor, 4., 10.)
        forecast = self.push(predictor, 9., 20.)
        original_prediction = forecast.prediction_mm.copy()
        self.push(predictor, 8000., 30.)
        np.testing.assert_allclose(original_prediction, 16.)
        np.testing.assert_array_equal(forecast.prediction_mm, original_prediction)
        self.assertEqual(len(predictor.buffer), 3)

    def test_stream_fit_support_differs_from_observation_masks(self):
        predictor = self.make_predictor()
        mask = np.ones((2, 2), bool)
        mask[0, 0] = False
        self.push(predictor, 1., 0., mask=mask)
        self.push(predictor, 2., 10.)
        forecast = self.push(predictor, 3., 20.)
        self.assertTrue(forecast.observation_valid.all())
        self.assertTrue(forecast.history_observed.all())
        self.assertFalse(forecast.prediction_valid[0, 0])
        self.assertTrue(np.isnan(forecast.prediction_mm[0, 0]))
        self.assertTrue(np.isnan(forecast.acceleration_mm_s2[0, 0]))

    def test_domain_time_and_counter_discontinuities_start_new_history(self):
        cases = ((50., 'camera', 3, 'timestamp discontinuity'),
                 (10., 'camera', 3, 'timestamp discontinuity'),
                 (5., 'camera', 3, 'timestamp discontinuity'),
                 (20., 'host', 3, 'timestamp domain changed'),
                 (20., 'camera', 2, 'frame counter reset or duplicate'),
                 (20., 'camera', 1, 'frame counter reset or duplicate'))
        for timestamp, domain, counter, reason in cases:
            with self.subTest(timestamp=timestamp, domain=domain, counter=counter):
                predictor = self.make_predictor()
                self.push(predictor, 1., 0., frame_number=1)
                self.push(predictor, 2., 10., frame_number=2)
                self.assertIsNone(self.push(predictor, 3., timestamp, domain, counter))
                self.assertEqual(predictor.reset_count, 1)
                self.assertEqual(predictor.last_reset_reason, reason)
                self.assertEqual(len(predictor.buffer), 1)

    def test_skipped_frames_use_actual_time_within_gap_limit(self):
        predictor = self.make_predictor()
        self.push(predictor, 100., 0., frame_number=1)
        self.push(predictor, 101., 10., frame_number=2)
        forecast = self.push(predictor, 103., 30., frame_number=4)
        self.assertEqual(predictor.reset_count, 0)
        np.testing.assert_allclose(forecast.prediction_mm, 104.)

    def test_low_coverage_and_nonfinite_depth_clear_history(self):
        predictor = self.make_predictor(min_valid_fraction=.5)
        self.push(predictor, 1., 0.)
        self.assertIsNone(self.push(predictor, np.nan, 10.))
        self.assertEqual(len(predictor.buffer), 0)
        self.assertEqual(predictor.last_reset_reason, 'insufficient valid depth')
        self.push(predictor, 1., 20.)
        self.assertIsNone(self.push(predictor, 1., 30., mask=np.array([[True, False], [False, False]])))
        self.assertEqual(len(predictor.buffer), 0)
        self.assertEqual(predictor.reset_count, 2)

    def test_invalid_timestamp_clears_history_and_raises(self):
        predictor = self.make_predictor()
        self.push(predictor, 1., 0.)
        with self.assertRaises(ValueError):
            self.push(predictor, 2., np.nan)
        self.assertEqual(len(predictor.buffer), 0)
        self.assertEqual(predictor.last_reset_reason, 'invalid timestamp')

    def test_changed_grid_clears_history_and_raises(self):
        predictor = self.make_predictor()
        self.push(predictor, 1., 0.)
        with self.assertRaises(ValueError):
            predictor.push(np.ones((3, 2)), np.ones((3, 2), bool), 10., 'camera')
        self.assertEqual(len(predictor.buffer), 0)
        self.assertEqual(predictor.last_reset_reason, 'grid shape changed')

    def test_config_validation_and_serialization(self):
        config = BaselineConfig()
        self.assertEqual(config.to_dict()['history'], 5)
        for name, values in (('history', (2, 0, -1, 3.5, True)),
                             ('horizon_ms', (0, -1, np.nan, np.inf)),
                             ('max_gap_ms', (0, -1, np.nan, np.inf)),
                             ('min_valid_fraction', (0, -1, 1.1, np.nan, np.inf))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    BaselineConfig(**{name: value})


if __name__ == '__main__':
    unittest.main()

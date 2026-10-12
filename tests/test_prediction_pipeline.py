from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from prediction.config import PredictionConfig
from prediction.data import FrameRecord, HeightSequenceDataset, Normalization, Session


if torch is not None:
    from prediction.evaluation import ErrorStats, baselines, evaluate
    from prediction.inference import StreamingPredictor

    class LastFrameModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.seen = None

        def forward(self, inputs):
            self.seen = inputs.detach().clone()
            return inputs[:, -1, :1] + self.weight


@unittest.skipIf(torch is None, 'prediction tests require requirements-prediction.txt')
class PredictionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(PredictionConfig(), history=3, expected_dt_ms=10,
                              hidden_channels=(2,), min_valid_fraction=0.5)
        self.model = LastFrameModel()
        self.normalization = Normalization(100, 2)
        self.checkpoint = {'geometry': {}, 'shape': [2, 2], 'train_sessions': [],
                           'validation_sessions': []}

    def predictor(self):
        with patch('prediction.inference.load_checkpoint', return_value=(
                self.model, self.config, self.normalization, self.checkpoint)):
            return StreamingPredictor('unused.pt', 'cpu')

    def test_streaming_matches_window_normalization_and_nominal_target(self):
        predictor = self.predictor()
        values = np.full((2, 2), 102, dtype=np.float32)
        mask = np.ones((2, 2), dtype=bool)
        values[0, 0], mask[0, 0] = np.nan, False
        self.assertIsNone(predictor.push(values, mask, 0, 'camera', 0))
        self.assertIsNone(predictor.push(values, mask, 10, 'camera', 1))
        result = predictor.push(values, mask, 20, 'camera', 2)
        self.assertEqual(result.target_timestamp_ms, 30)
        self.assertEqual(result.source_timestamp_ms, 20)
        self.assertEqual(result.timestamp_domain, 'camera')
        self.assertTrue(torch.isfinite(self.model.seen).all())
        self.assertEqual(float(self.model.seen[0, 0, 0, 0, 0]), 0)
        np.testing.assert_array_equal(result.observation_valid, mask)
        np.testing.assert_allclose(result.prediction_mm[mask], 102)
        values[:] = 999
        np.testing.assert_allclose(predictor.buffer[-1][0][mask], 102)

    def test_gap_low_coverage_and_clock_change_clear_history(self):
        predictor = self.predictor()
        values = np.ones((2, 2), dtype=np.float32)
        mask = np.ones_like(values, dtype=bool)
        predictor.push(values, mask, 0, 'a', 0)
        predictor.push(values, mask, 10, 'a', 1)
        self.assertIsNone(predictor.push(values, mask, 40, 'a', 4))
        self.assertEqual(len(predictor.buffer), 1)
        predictor.push(values, mask, 50, 'b', 5)
        self.assertEqual(len(predictor.buffer), 1)
        predictor.push(values, np.zeros_like(mask), 60, 'b', 6)
        self.assertEqual(len(predictor.buffer), 0)
        self.assertEqual(predictor.reset_count, 3)

    def test_duplicate_counter_resets_even_with_regular_timestamps(self):
        predictor = self.predictor()
        values = np.ones((2, 2), dtype=np.float32)
        mask = np.ones_like(values, dtype=bool)
        predictor.push(values, mask, 0, 'a', 10)
        predictor.push(values, mask, 10, 'a', 10)
        self.assertEqual(predictor.reset_count, 1)
        self.assertEqual(len(predictor.buffer), 1)

    def test_checkpoint_shape_rejects_first_frame(self):
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.predictor().push(np.ones((3, 3)), np.ones((3, 3), bool), 0, 'a')

    def test_error_metrics_exact_sums_and_bounded_percentile_sample(self):
        metrics = ErrorStats(max_samples=3)
        metrics.update([-1, 2])
        metrics.update([3, -4])
        result = metrics.result()
        self.assertEqual(result['valid_pixels'], 4)
        self.assertEqual(result['mae_mm'], 2.5)
        self.assertAlmostEqual(result['rmse_mm'], np.sqrt(7.5))
        self.assertEqual(result['p95_sample_count'], 3)
        self.assertTrue(result['p95_is_sampled'])
        self.assertIsNone(ErrorStats().result()['mae_mm'])
        with self.assertRaises(ValueError):
            metrics.update([np.nan])

    def test_evaluation_compares_same_support_and_nominal_horizon(self):
        last = np.array([[1, 4], [9, 3]], dtype=np.float32)
        previous = np.array([[0, 2], [8, 1]], dtype=np.float32)
        target = np.array([[2, 6], [11, 100]], dtype=np.float32)
        sample = {'inputs': np.stack([np.stack((last, np.ones_like(last)))] * 3),
                  'target': target[None], 'target_valid': np.ones((1, 2, 2), bool),
                  'last_mm': last, 'previous_mm': previous,
                  'last_valid': np.array([[True, True], [True, False]]),
                  'previous_valid': np.array([[True, False], [True, True]]),
                  'dt_ms': 10, 'target_dt_ms': 15, 'last_timestamp_ms': 20,
                  'target_timestamp_ms': 35, 'session': 'held_out'}
        # Legacy callers without a full history keep the two-frame API.
        np.testing.assert_allclose(baselines(sample, 10)['constant_velocity'], [[2, 6], [10, 5]])
        self.assertNotIn('constant_acceleration', baselines(sample, 10))
        sample.update(
            history_mm=np.stack([2 * previous - last, previous, last]),
            history_valid=np.stack([np.ones_like(last, bool), sample['previous_valid'], sample['last_valid']]),
            history_timestamps_ms=np.array([0, 10, 20], dtype=np.float64),
        )
        with tempfile.TemporaryDirectory() as temp:
            session = SimpleNamespace(path=Path(temp) / 'held_out', geometry={}, shape=(2, 2))
            output = Path(temp) / 'evaluation'
            with patch('prediction.evaluation.load_checkpoint', return_value=(
                    self.model, self.config, Normalization(0, 1), self.checkpoint)), \
                 patch('prediction.evaluation.load_splits', return_value={'test': [session]}), \
                 patch('prediction.evaluation.HeightSequenceDataset', return_value=[sample]):
                report = evaluate('fake.pt', 'splits.json', output, device='cpu', motion_threshold_mm=1)
            self.assertEqual(report['comparison_valid_pixels'], 2)
            self.assertEqual(report['target_valid_pixels'], 4)
            self.assertEqual(report['metrics']['all']['persistence']['mae_mm'], 1.5)
            self.assertEqual(report['metrics']['all']['constant_velocity']['mae_mm'], 0.5)
            self.assertEqual(report['metrics']['all']['constant_acceleration']['mae_mm'], 0.5)
            self.assertEqual(report['metrics']['moving']['convlstm']['valid_pixels'], 1)
            self.assertEqual(report['metrics']['stationary']['convlstm']['valid_pixels'], 1)
            self.assertEqual(report['observed_target_offset_from_nominal_ms']['mean'], 5)
            self.assertTrue((output / 'report.html').is_file())
            json.loads((output / 'metrics.json').read_text())

    def test_acceleration_evaluation_uses_history_support_and_exact_quadratic(self):
        history = np.stack([np.array([[i * i, 7], [i * i, i * i]], dtype=np.float32)
                            for i in range(3)])
        history_valid = np.ones_like(history, bool)
        history_valid[0, 0, 0] = False  # Latest two frames alone cannot establish support.
        target = np.array([[9, 7], [9, 9]], dtype=np.float32)
        sample = {
            'inputs': np.stack((history, history_valid.astype(np.float32)), axis=1),
            'target': target[None], 'target_valid': np.ones((1, 2, 2), bool),
            'history_mm': history, 'history_valid': history_valid,
            'history_timestamps_ms': np.array([0, 10, 20], dtype=np.float64),
            'last_mm': history[-1], 'previous_mm': history[-2],
            'last_valid': history_valid[-1], 'previous_valid': history_valid[-2],
            'dt_ms': 10, 'target_dt_ms': 10, 'last_timestamp_ms': 20,
            'target_timestamp_ms': 30, 'session': 'held_out',
        }
        with tempfile.TemporaryDirectory() as temp:
            session = SimpleNamespace(path=Path(temp) / 'held_out', geometry={}, shape=(2, 2))
            output = Path(temp) / 'evaluation'
            with patch('prediction.evaluation.load_checkpoint', return_value=(
                    self.model, self.config, Normalization(0, 1), self.checkpoint)), \
                 patch('prediction.evaluation.load_splits', return_value={'test': [session]}), \
                 patch('prediction.evaluation.HeightSequenceDataset', return_value=[sample]):
                report = evaluate('fake.pt', 'splits.json', output, device='cpu', motion_threshold_mm=1)
            self.assertEqual(report['comparison_valid_pixels'], 3)
            self.assertEqual(report['comparison_coverage'], 0.75)
            for region, count in [('all', 3), ('moving', 2), ('stationary', 1)]:
                for name, metrics in report['metrics'][region].items():
                    self.assertEqual(metrics['valid_pixels'], count, (region, name))
                    self.assertEqual(report['per_session']['held_out'][region][name], metrics)
            self.assertAlmostEqual(report['metrics']['all']['constant_acceleration']['mae_mm'], 0)
            self.assertGreater(report['metrics']['all']['constant_velocity']['mae_mm'], 2)
            with np.load(output / 'example_0000.npz') as example:
                self.assertTrue(np.isnan(example['constant_acceleration_mm'][0, 0]))
                self.assertFalse(example['constant_acceleration_valid'][0, 0])
                self.assertTrue(np.isnan(example['constant_velocity_error_mm'][0, 0]))
                self.assertFalse(example['comparison_valid'][0, 0])
                np.testing.assert_allclose(example['constant_acceleration_mm'][example['comparison_valid']],
                                           target[example['comparison_valid']])

    def test_evaluation_retains_two_frame_checkpoint_compatibility(self):
        history = np.stack([np.ones((2, 2), np.float32), np.full((2, 2), 2, np.float32)])
        history_valid = np.ones_like(history, bool)
        sample = {
            'inputs': np.stack((history, history_valid.astype(np.float32)), axis=1),
            'target': np.full((1, 2, 2), 3, np.float32), 'target_valid': np.ones((1, 2, 2), bool),
            'history_mm': history, 'history_valid': history_valid,
            'history_timestamps_ms': np.array([0, 10], dtype=np.float64),
            'last_mm': history[-1], 'previous_mm': history[-2],
            'last_valid': history_valid[-1], 'previous_valid': history_valid[-2],
            'dt_ms': 10, 'target_dt_ms': 10, 'last_timestamp_ms': 10,
            'target_timestamp_ms': 20, 'session': 'held_out',
        }
        with tempfile.TemporaryDirectory() as temp:
            session = SimpleNamespace(path=Path(temp) / 'held_out', geometry={}, shape=(2, 2))
            output = Path(temp) / 'evaluation'
            with patch('prediction.evaluation.load_checkpoint', return_value=(
                    self.model, replace(self.config, history=2), Normalization(0, 1), self.checkpoint)), \
                 patch('prediction.evaluation.load_splits', return_value={'test': [session]}), \
                 patch('prediction.evaluation.HeightSequenceDataset', return_value=[sample]):
                report = evaluate('fake.pt', 'splits.json', output, device='cpu')
            self.assertEqual(set(report['metrics']['all']), {'convlstm', 'persistence', 'constant_velocity'})
            self.assertEqual(report['metrics']['all']['constant_velocity']['mae_mm'], 0)
            with np.load(output / 'example_0000.npz') as example:
                self.assertNotIn('constant_acceleration_mm', example.files)

    def test_evaluation_refuses_training_data_as_test(self):
        with tempfile.TemporaryDirectory() as temp:
            session = SimpleNamespace(path=Path(temp), geometry={}, shape=(2, 2))
            self.checkpoint['train_sessions'] = [str(Path(temp).resolve())]
            with patch('prediction.evaluation.load_checkpoint', return_value=(
                    self.model, self.config, self.normalization, self.checkpoint)), \
                 patch('prediction.evaluation.load_splits', return_value={'test': [session]}):
                with self.assertRaisesRegex(ValueError, 'overlaps'):
                    evaluate('fake.pt', 'splits.json', Path(temp) / 'eval')


class BaselineHistoryDataTests(unittest.TestCase):
    def test_dataset_preserves_physical_history_times_and_excludes_future(self):
        frames = tuple(FrameRecord(Path(str(i)), i, 10.0 * i, (i + 1) * 10000000,
                                   'camera', 1.0) for i in range(5))
        session = Session(Path('session'), {}, (2, 2), frames, {})
        history_mask = np.array([[False, True], [True, True]])

        def load_arrays(path, shape):
            index = int(path.name)
            return (np.full(shape, 100 + 2 * index, dtype=np.float32),
                    history_mask.copy() if index == 0 else np.ones(shape, dtype=bool))

        config = replace(PredictionConfig(), history=3, horizon=2, expected_dt_ms=10)
        dataset = HeightSequenceDataset([session], config, Normalization(100, 2))
        with patch('prediction.data._load_arrays', side_effect=load_arrays):
            sample = dataset[0]
        np.testing.assert_array_equal(sample['history_mm'][:, 1, 1], [100, 102, 104])
        np.testing.assert_array_equal(sample['history_timestamps_ms'], [0, 10, 20])
        np.testing.assert_array_equal(sample['history_valid'][0], history_mask)
        self.assertEqual(sample['history_mm'].dtype, np.float32)
        self.assertEqual(sample['history_valid'].dtype, np.bool_)
        self.assertEqual(sample['history_timestamps_ms'].dtype, np.float64)
        np.testing.assert_array_equal(sample['inputs'][:, 0, 1, 1], [0, 1, 2])
        self.assertEqual(sample['target'][0, 1, 1], 4)
        self.assertEqual(sample['target_timestamp_ms'], 40)
        self.assertLess(sample['history_timestamps_ms'][-1], sample['target_timestamp_ms'])


if __name__ == '__main__':
    unittest.main()

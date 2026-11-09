"""Causality and target-support checks for paired occlusion windows."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from prediction.data import FrameRecord, Normalization, Session
from prediction.occlusion_data import OcclusionDataset, encode_history, _gaussian_masks


class OcclusionDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = dict(history=3, horizon=2, expected_dt_ms=10,
                           timing_tolerance=0.2, min_valid_fraction=0.1,
                           age_scale_ms=20, occlusion_fraction=0.25)

    def session(self, timestamps=(0, 10, 20, 30, 40), masks=None):
        frames = []
        shape = (8, 8)
        for i, timestamp in enumerate(timestamps):
            valid = np.ones(shape, dtype=bool) if masks is None else masks[i]
            path = self.root / f'frame_{i}.npz'
            np.savez(path, height_mm=np.full(shape, i + 1, dtype=np.float32),
                     valid=valid, point_count=valid.astype(np.uint32))
            frames.append(FrameRecord(path, i, float(timestamp), i + 1, 'clock', float(valid.mean())))
        return Session(self.root, {}, shape, tuple(frames), {'grid': shape})

    def dataset(self, session, **changes):
        return OcclusionDataset([session], {**self.config, **changes}, Normalization(1, 2))

    def test_shapes_determinism_independent_validity_and_no_future_leakage(self):
        session = self.session()
        dataset = self.dataset(session)
        sample = dataset[0]
        self.assertEqual(sample['inputs'].shape, (3, 4, 8, 8))
        self.assertEqual(sample['targets'].shape, (3, 1, 8, 8))
        self.assertEqual(sample['target_valid'].shape, (3, 1, 8, 8))
        self.assertEqual(sample['hidden_origin'].shape, (1, 8, 8))
        self.assertTrue(sample['target_valid'].all())
        self.assertTrue(sample['hidden_origin'].any())
        self.assertTrue(np.all(sample['inputs'][:, 0][sample['inputs'][:, 1] == 0] == 0))
        np.testing.assert_array_equal(sample['targets'][:, 0, 0, 0], [1, 1.5, 2])
        np.testing.assert_array_equal(dataset[0]['inputs'], sample['inputs'])
        np.savez(session.frames[3].path, height_mm=np.full((8, 8), 999.),
                 valid=np.ones((8, 8), dtype=bool), point_count=np.ones((8, 8), dtype=np.uint32))
        changed = dataset[0]
        np.testing.assert_array_equal(changed['inputs'], sample['inputs'])
        self.assertNotEqual(changed['targets'][1, 0, 0, 0], sample['targets'][1, 0, 0, 0])

    def test_masks_are_contiguous_and_move_coherently(self):
        sample = self.dataset(self.session())[0]
        masks = sample['inputs'][:, 1] == 0
        for mask in masks:
            ys, xs = np.where(mask)
            self.assertEqual(mask.sum(), (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        for first, second in zip(masks, masks[1:]):
            y1, x1 = np.where(first)
            y2, x2 = np.where(second)
            self.assertLessEqual(abs(int(y1.min()) - int(y2.min())), 1)
            self.assertLessEqual(abs(int(x1.min()) - int(x2.min())), 1)

    def test_gaussian_masks_are_reproducible_and_coherent(self):
        first = _gaussian_masks((32, 40), 5, .2, np.random.default_rng(12))
        second = _gaussian_masks((32, 40), 5, .2, np.random.default_rng(12))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first.sum(axis=(1, 2)), np.full(5, 256))
        for a, b in zip(first, first[1:]):
            self.assertGreater((a & b).sum(), .8 * a.sum())
        self.assertFalse(_gaussian_masks((8, 8), 3, 0, np.random.default_rng(1)).any())
        self.assertTrue(_gaussian_masks((8, 8), 3, 1, np.random.default_rng(1)).all())

    def test_gaussian_augmentation_keeps_clean_targets(self):
        session = self.session()
        gaussian = self.dataset(session, occlusion_shape='gaussian')[0]
        clean = self.dataset(session, occlusion_fraction=0)[0]
        self.assertTrue(gaussian['hidden_origin'].any())
        np.testing.assert_array_equal(gaussian['targets'], clean['targets'])
        np.testing.assert_array_equal(gaussian['target_valid'], clean['target_valid'])
        np.testing.assert_array_equal(gaussian['hidden_origin'][0], gaussian['inputs'][-1, 1] == 0)
        self.assertEqual(self.dataset(session, occlusion_shape='gaussian').summary()['occlusion_shape'], 'gaussian')

    def test_gaussian_step_interval_preserves_speed_across_frame_rates(self):
        original = _gaussian_masks((32, 40), 3, .2, np.random.default_rng(12))
        explicit = _gaussian_masks((32, 40), 3, .2, np.random.default_rng(12), 1)
        at_30_fps = _gaussian_masks((32, 40), 13, .2, np.random.default_rng(12), 6)
        np.testing.assert_array_equal(original, explicit)
        np.testing.assert_array_equal(original, at_30_fps[::6])
        for i in range(1, 6):
            np.testing.assert_array_equal(at_30_fps[0], at_30_fps[i])
        self.assertFalse(np.array_equal(at_30_fps[0], at_30_fps[6]))

    def test_large_stationary_masks_cover_requested_range_and_do_not_move(self):
        session = self.session(timestamps=tuple(range(0, 100, 10)))
        dataset = self.dataset(session, history=6, occlusion_shape='large',
                               large_occlusion_fraction_range=[0.7, 0.9],
                               large_occlusion_stationary_probability=1.0)
        for index in range(len(dataset)):
            observed = dataset[index]['inputs'][:, 1] > .5
            hidden_fraction = 1 - observed.mean(axis=(1, 2))
            self.assertTrue(np.all((hidden_fraction >= 0.69) & (hidden_fraction <= 0.91)))
            self.assertTrue(np.all(observed == observed[0]))
        self.assertEqual(dataset.summary()['large_occlusion_fraction_range'], [0.7, 0.9])

    def test_mixed_masks_draw_both_regimes_reproducibly(self):
        session = self.session(timestamps=tuple(range(0, 400, 10)))
        dataset = self.dataset(session, history=3, horizon=1, occlusion_shape='mixed',
                               occlusion_fraction=0.2, large_occlusion_probability=0.5)
        fractions = [1 - (dataset[i]['inputs'][-1, 1] > .5).mean() for i in range(len(dataset))]
        self.assertTrue(any(f < 0.3 for f in fractions))
        self.assertTrue(any(f >= 0.39 for f in fractions))
        again = self.dataset(session, history=3, horizon=1, occlusion_shape='mixed',
                             occlusion_fraction=0.2, large_occlusion_probability=0.5)
        np.testing.assert_array_equal(dataset[5]['inputs'], again[5]['inputs'])

    def test_gaussian_masks_unchanged_by_large_mask_settings(self):
        session = self.session(timestamps=tuple(range(0, 100, 10)))
        plain = self.dataset(session, occlusion_shape='gaussian')
        configured = self.dataset(session, occlusion_shape='gaussian', large_occlusion_probability=1.0,
                                  large_occlusion_fraction_range=[0.8, 0.9])
        for index in range(len(plain)):
            np.testing.assert_array_equal(plain[index]['inputs'], configured[index]['inputs'])

    def test_invalid_large_mask_settings_are_rejected(self):
        session = self.session()
        for bad in ({'large_occlusion_fraction_range': [0.9, 0.4]}, {'large_occlusion_fraction_range': [0, 0.5]},
                    {'large_occlusion_fraction_range': [0.5]}, {'large_occlusion_probability': 1.5},
                    {'large_occlusion_stationary_probability': True}, {'occlusion_shape': 'ring'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.dataset(session, **bad)

    def test_frame_cache_and_window_stride_preserve_samples(self):
        from prediction.occlusion_data import _load_arrays
        session = self.session(timestamps=tuple(range(0, 100, 10)))
        norm = Normalization(1, 2)
        full = OcclusionDataset([session], self.config, norm)
        cached = OcclusionDataset([session], self.config, norm, cache_frames=True, window_stride=2)
        self.assertEqual(full.summary()['windows'], 6)
        self.assertEqual(cached.summary()['windows'], 3)
        self.assertEqual(cached.summary()['window_stride'], 2)
        with patch('prediction.occlusion_data._load_arrays', wraps=_load_arrays) as read:
            sample = cached[0]
            self.assertEqual(read.call_count, 5)
            cached[0]
            self.assertEqual(read.call_count, 5)
        for key, value in sample.items():
            np.testing.assert_array_equal(value, full[0][key])
        for key, value in cached[1].items():
            np.testing.assert_array_equal(value, full[2][key])
        for kwargs in ({'cache_frames': 1}, {'window_stride': 0}, {'window_stride': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OcclusionDataset([session], self.config, norm, **kwargs)

    def test_age_never_observed_and_clean_target_holes(self):
        valid = [np.ones((8, 8), dtype=bool) for _ in range(5)]
        valid[0][0, 0] = False
        valid[1][0, 0] = False
        valid[1][0, 1] = False
        valid[2][0, 1] = False
        valid[2][0, 2] = False
        sample = self.dataset(self.session(masks=valid), occlusion_fraction=0)[0]
        inputs = sample['inputs']
        np.testing.assert_array_equal(inputs[:, 3, 0, 0], [1, 1, 0])
        np.testing.assert_array_equal(inputs[:, 2, 0, 0], [1, 1, 0])
        np.testing.assert_array_equal(inputs[:, 2, 0, 1], [0, 0.5, 1])
        self.assertFalse(sample['target_valid'][0, 0, 0, 2])
        self.assertFalse(sample['hidden_origin'][0, 0, 2])

    def test_timing_and_all_future_coverage(self):
        bad_timing = self.dataset(self.session(timestamps=(0, 10, 20, 50, 60)))
        self.assertEqual(len(bad_timing), 0)
        self.assertEqual(bad_timing.summary()['rejected_timing'], 1)
        masks = [np.ones((8, 8), dtype=bool) for _ in range(5)]
        masks[3][:] = False
        missing_future = self.dataset(self.session(masks=masks))
        self.assertEqual(len(missing_future), 0)
        self.assertEqual(missing_future.summary()['rejected_coverage'], 1)

    def test_30fps_configuration_accepts_matching_cadence_only(self):
        config_path = Path(__file__).resolve().parents[1] / 'configs/occlusion_gaussian_30fps.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        self.assertEqual((config['history'], config['horizon'], config['gaussian_step_interval']), (25, 6, 6))
        norm = Normalization(1, 2)
        at_30fps = self.session(timestamps=tuple(i * 1000 / 30 for i in range(32)))
        dataset = OcclusionDataset([at_30fps], config, norm)
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        self.assertEqual(sample['inputs'].shape[0], 25)
        self.assertEqual(sample['targets'].shape[0], 7)
        at_5fps = self.session(timestamps=tuple(i * 200 for i in range(32)))
        mismatched = OcclusionDataset([at_5fps], config, norm)
        self.assertEqual(len(mismatched), 0)
        self.assertEqual(mismatched.summary()['rejected_timing'], 2)

    def test_shared_encoding_matches_dataset(self):
        valid = [np.ones((8, 8), dtype=bool) for _ in range(5)]
        valid[1][0, 0] = False
        session = self.session(masks=valid)
        sample = self.dataset(session, occlusion_fraction=0)[0]
        heights = np.stack([np.full((8, 8), i + 1, dtype=np.float32) for i in range(3)])
        encoded = encode_history(heights, valid[:3], [0, 10, 20], Normalization(1, 2), 20)
        np.testing.assert_array_equal(encoded, sample['inputs'])
        heights[2, 0, 1] = np.nan
        encoded = encode_history(heights, valid[:3], [0, 10, 20], Normalization(1, 2), 20)
        self.assertEqual(encoded[2, 1, 0, 1], 0)
        self.assertEqual(encoded[2, 2, 0, 1], 0.5)
        with self.assertRaises(ValueError):
            encode_history(heights, valid[:3], [0, 10, 10], Normalization(1, 2), 20)

    def test_validation_and_other_seed(self):
        session = self.session()
        first = self.dataset(session)[0]['inputs']
        second = OcclusionDataset([session], self.config, Normalization(1, 2), seed=43)[0]['inputs']
        self.assertFalse(np.array_equal(first, second))
        for change in ({'age_scale_ms': 0}, {'occlusion_fraction': 1.1}, {'horizon': 0},
                       {'occlusion_shape': 'unknown'}, {'gaussian_step_interval': 0},
                       {'gaussian_step_interval': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.dataset(session, **change)


if __name__ == '__main__':
    unittest.main()

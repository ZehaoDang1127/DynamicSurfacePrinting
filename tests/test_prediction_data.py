"""Dataset tests use tiny recorder-format files, without torch or a camera."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prediction.config import PredictionConfig
from prediction.data import (
    HeightSequenceDataset,
    Normalization,
    fit_normalization,
    geometry_signature,
    load_session,
    load_splits,
)


class PredictionDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = PredictionConfig(history=2, expected_dt_ms=10, timing_tolerance=0.1)

    def recording(self, name='session', values=(1, 2, 3, 4), timestamps=None, masks=None):
        directory = self.root / name
        directory.mkdir()
        timestamps = list(timestamps if timestamps is not None else np.arange(len(values)) * 10)
        config = {
            'frame_name': 'camera',
            'camera': {'width': 640, 'height': 480, 'fps': 100},
            'grid': {'x_min_mm': 0, 'x_max_mm': 2, 'y_min_mm': 0, 'y_max_mm': 2,
                     'cell_size_mm': 1, 'z_min_mm': -100, 'z_max_mm': 10000},
            'camera_to_output': np.eye(4).tolist(),
        }
        frames = []
        for index, value in enumerate(values):
            height = np.full((2, 2), value, dtype=np.float32) if np.ndim(value) == 0 else np.asarray(value, dtype=np.float32)
            valid = np.ones((2, 2), dtype=bool) if masks is None else np.asarray(masks[index], dtype=bool)
            filename = f'frame_{index + 1:06d}.npz'
            metadata = {'frame_number': index + 1, 'camera_timestamp_ms': float(timestamps[index]),
                        'host_monotonic_ns': (index + 1) * 10000000}
            np.savez_compressed(directory / filename, height_mm=height, valid=valid,
                                point_count=valid.astype(np.uint32), **metadata)
            frames.append({'file': filename, 'timestamp_domain': 'hardware_clock',
                           'valid_fraction': float(valid.mean()), **metadata})
        manifest = {'schema_version': 1, 'complete': True, 'config': config,
                    'grid': {'shape': [2, 2], 'x_centers_mm': [0.5, 1.5], 'y_centers_mm': [0.5, 1.5]},
                    'frame_count': len(frames), 'frames': frames}
        (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return directory

    def edit_manifest(self, directory, edit):
        path = directory / 'manifest.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        edit(data)
        path.write_text(json.dumps(data), encoding='utf-8')

    def edit_frame(self, directory, index, **changes):
        path = directory / f'frame_{index:06d}.npz'
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        data.update(changes)
        np.savez_compressed(path, **data)

    def split_file(self, train, validation, test):
        path = self.root / 'splits.json'
        path.write_text(json.dumps({'train': train, 'validation': validation, 'test': test}), encoding='utf-8')
        return path

    def test_windows_target_and_timing_metadata(self):
        session = load_session(self.recording())
        dataset = HeightSequenceDataset([session], self.config, Normalization(0, 1))
        self.assertEqual(len(dataset), 2)
        example = dataset[0]
        self.assertEqual(example['inputs'].shape, (2, 2, 2, 2))
        self.assertEqual(example['inputs'].dtype, np.float32)
        np.testing.assert_array_equal(example['inputs'][:, 0, 0, 0], [1, 2])
        np.testing.assert_array_equal(example['target'], np.full((1, 2, 2), 3))
        self.assertEqual(example['target_valid'].dtype, np.bool_)
        self.assertEqual(example['dt_ms'], 10)
        self.assertEqual(example['target_dt_ms'], 10)
        self.assertEqual(example['last_timestamp_ms'], 10)
        self.assertEqual(example['target_timestamp_ms'], 20)
        self.assertEqual(dataset.summary()['candidate_windows'], 2)

    def test_horizon_checks_every_intermediate_interval(self):
        session = load_session(self.recording(values=(0, 1, 2, 3, 4), timestamps=(0, 10, 20, 50, 60)))
        config = PredictionConfig(history=2, horizon=2, expected_dt_ms=10)
        dataset = HeightSequenceDataset([session], config, Normalization(0, 1))
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.summary()['rejected_timing'], 2)

    def test_horizon_target_is_beyond_last_input(self):
        session = load_session(self.recording(values=(0, 1, 2, 3, 4)))
        config = PredictionConfig(history=2, horizon=2, expected_dt_ms=10)
        dataset = HeightSequenceDataset([session], config, Normalization(0, 1))
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['target'][0, 0, 0], 3)
        self.assertEqual(dataset[0]['target_dt_ms'], 20)

    def test_gaps_remove_only_affected_windows(self):
        session = load_session(self.recording(values=(0, 1, 2, 3, 4, 5), timestamps=(0, 10, 20, 50, 60, 70)))
        dataset = HeightSequenceDataset([session], self.config, Normalization(0, 1))
        self.assertEqual(len(dataset), 2)
        self.assertEqual([dataset[i]['target_timestamp_ms'] for i in range(2)], [20, 70])

    def test_nonfinite_values_are_masked_and_arrays_are_finite(self):
        values = [np.array([[2, np.nan], [np.inf, 9000]])] * 3
        masks = [np.array([[True, True], [True, False]])] * 3
        session = load_session(self.recording(values=values, masks=masks))
        normalization = fit_normalization([session])
        self.assertEqual(normalization.mean_mm, 2)
        self.assertEqual(normalization.scale_mm, 1e-3)
        example = HeightSequenceDataset([session], self.config, normalization)[0]
        for key in ('inputs', 'target', 'last_mm', 'previous_mm'):
            self.assertTrue(np.isfinite(example[key]).all())
        self.assertEqual(example['target_valid'].sum(), 1)
        self.assertEqual(example['inputs'][:, 1].sum(), 2)
        self.assertEqual(example['last_mm'][1, 1], 0)

    def test_coverage_uses_finite_valid_intersection(self):
        session = load_session(self.recording(values=[np.array([[2, np.nan], [np.nan, np.nan]])] * 3))
        config = PredictionConfig(history=2, expected_dt_ms=10, min_valid_fraction=0.5)
        dataset = HeightSequenceDataset([session], config, Normalization(0, 1))
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.summary()['rejected_coverage'], 1)

    def test_stats_use_training_only_and_do_not_reweight_overlapping_windows(self):
        self.recording('train', values=(1, 2, 9))
        self.recording('validation', values=(100, 100, 100))
        self.recording('test', values=(1000, 1000, 1000))
        splits = load_splits(self.split_file(['train'], ['validation'], ['test']))
        result = fit_normalization(splits['train'] * 2)
        self.assertAlmostEqual(result.mean_mm, 4)
        self.assertAlmostEqual(result.scale_mm, float(np.std([1, 2, 9])))
        self.assertEqual(Normalization.from_dict(result.to_dict()), result)
        np.testing.assert_allclose(result.denormalize(result.normalize(np.array([1., 2.]), np.array([True, True]))), [1, 2], atol=1e-6)

    def test_empty_valid_training_raises(self):
        masks = [np.zeros((2, 2), dtype=bool)] * 3
        with self.assertRaisesRegex(ValueError, 'no finite valid'):
            fit_normalization([load_session(self.recording(values=(1, 2, 3), masks=masks))])

    def test_duplicate_session_split_alias_is_rejected(self):
        self.recording('train')
        self.recording('test')
        with self.assertRaisesRegex(ValueError, 'more than once'):
            load_splits(self.split_file(['train'], ['./train'], ['test']))

    def test_hardlinked_frames_across_splits_are_rejected(self):
        first = self.recording('train')
        second = self.recording('validation')
        self.recording('test')
        target = second / 'frame_000001.npz'
        target.unlink()
        try:
            os.link(first / 'frame_000001.npz', target)
        except OSError as exc:
            self.skipTest(f'hardlinks unavailable: {exc}')
        with self.assertRaisesRegex(ValueError, 'frame occurs more than once'):
            load_splits(self.split_file(['train'], ['validation'], ['test']))

    def test_incompatible_geometry_is_rejected(self):
        self.recording('train')
        second = self.recording('validation')
        self.recording('test')
        self.edit_manifest(second, lambda data: data['config'].update(frame_name='printer'))
        with self.assertRaisesRegex(ValueError, 'incompatible recording geometry'):
            load_splits(self.split_file(['train'], ['validation'], ['test']))

    def test_geometry_ignores_startup_settings_and_device_serial(self):
        directory = self.recording()
        original = load_session(directory).geometry
        self.edit_manifest(directory, lambda data: data['config']['camera'].update(serial='different-device', timeout_ms=999, warmup_frames=100))
        self.assertEqual(original, load_session(directory).geometry)
        changed = dict(original)
        changed['camera'] = {**changed['camera'], 'width': 320}
        self.assertNotEqual(original, geometry_signature(changed))

    def test_shape_disagreement_is_rejected(self):
        directory = self.recording()
        self.edit_frame(directory, 1, height_mm=np.ones((3, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'frame shape disagrees'):
            load_session(directory)

    def test_config_grid_disagreement_is_rejected(self):
        directory = self.recording()
        self.edit_manifest(directory, lambda data: data['config']['grid'].update(x_max_mm=3))
        with self.assertRaisesRegex(ValueError, 'grid shape disagrees'):
            load_session(directory)

    def test_manifest_npz_metadata_disagreement_is_rejected(self):
        for field, value in [('camera_timestamp_ms', 8.0), ('frame_number', 18), ('host_monotonic_ns', 1)]:
            with self.subTest(field=field):
                directory = self.recording(field)
                self.edit_frame(directory, 1, **{field: value})
                with self.assertRaisesRegex(ValueError, f'{field} disagrees'):
                    load_session(directory)

    def test_bad_manifest_entries_are_rejected_with_context(self):
        mutations = {
            'incomplete': lambda data: data.update(complete=False),
            'count': lambda data: data.update(frame_count=500),
            'traversal': lambda data: data['frames'][0].update(file='../outside.npz'),
            'duplicate': lambda data: data['frames'][1].update(file=data['frames'][0]['file']),
            'domain': lambda data: data['frames'][1].update(timestamp_domain='system_time'),
            'nan': lambda data: data['frames'][1].update(camera_timestamp_ms=float('nan')),
            'backwards': lambda data: data['frames'][1].update(camera_timestamp_ms=0),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                directory = self.recording(name)
                self.edit_manifest(directory, mutation)
                with self.assertRaisesRegex(ValueError, 'invalid session'):
                    load_session(directory)

    def test_malformed_split_does_not_silently_ignore_unknown_keys(self):
        path = self.root / 'bad.json'
        for value in ({'train': [], 'validation': [], 'test': []}, {'train': ['a'], 'val': ['b'], 'test': ['c']}, []):
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_splits(path)


if __name__ == '__main__':
    unittest.main()

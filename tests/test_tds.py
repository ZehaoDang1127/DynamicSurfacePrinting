import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prediction.config import PredictionConfig
from prediction.data import HeightSequenceDataset, fit_normalization, load_splits
from prediction.tds import TEST, VALIDATION, depth_points, prepare


class TDSPreparationTests(unittest.TestCase):
    def test_unprojection_rejects_missing_depth_and_preserves_metric_coordinates(self):
        depth = np.array([[0, 1, np.nan], [-1, 2, np.inf]], dtype=np.float32)
        intrinsics = np.array([[2, 0, 1], [0, 4, 0], [0, 0, 1]])
        np.testing.assert_allclose(depth_points(depth, intrinsics), [[0, 0, 1], [0, .5, 2]])

    def test_import_preserves_gaps_units_and_whole_sequence_holdouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'raw'
            root.mkdir()
            np.savetxt(root / 'camera_intrinsics.txt', [[242.5, 0, 112], [0, 242.5, 112], [0, 0, 1]])
            for name in (*VALIDATION, *TEST, 'train_trial'):
                destination = root / 'cloth/depth_maps' / name
                destination.mkdir(parents=True)
                numbers = [0, 1, 2, 3, 5, 6, 7, 8] if name == 'train_trial' else list(range(8))
                for number in numbers:
                    # A known frontoparallel plane, with source depth in metres.
                    depth = np.full((224, 224), 1 + number * .001, dtype=np.float32)
                    np.savez_compressed(destination / f'depth_{number:04d}.npz', depth=depth)
            with contextlib.redirect_stdout(io.StringIO()):
                split_path = prepare(root, Path(directory) / 'prepared')
            splits = load_splits(split_path)
            self.assertEqual([s.path.name for s in splits['train']], ['train_trial'])
            self.assertEqual(set(s.path.name for s in splits['test']), set(TEST))
            train = splits['train'][0]
            self.assertEqual([f.camera_timestamp_ms for f in train.frames], [0, 200, 400, 600, 1000, 1200, 1400, 1600])
            with np.load(train.frames[0].path) as frame:
                np.testing.assert_allclose(frame['height_mm'][frame['valid']], 1000)
            config = PredictionConfig(history=3, expected_dt_ms=200)
            normalization = fit_normalization(splits['train'])
            uncached = HeightSequenceDataset(splits['train'], config, normalization)
            cached = HeightSequenceDataset(splits['train'], config, normalization, cache_frames=True)
            self.assertEqual(len(uncached), 2)  # No window bridges the missing index 4.
            for key in ('inputs', 'target', 'target_valid', 'history_mm', 'history_valid'):
                np.testing.assert_array_equal(uncached[0][key], cached[0][key])
            audit = json.loads((split_path.parent / 'data_audit.json').read_text())
            self.assertEqual(audit['total_frames'], 56)
            self.assertEqual(len(list((root / 'cloth/depth_maps/train_trial').glob('*.npz'))), 8)


if __name__ == '__main__':
    unittest.main()

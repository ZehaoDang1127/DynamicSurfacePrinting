"""Printer-frame re-scoring of saved real print-head predictions."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import scipy  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest('install scipy to test printer-frame re-scoring') from exc

from tools.rescore_real_head_printer import rescore


class RescoreTests(unittest.TestCase):
    def test_constant_offsets_with_identity_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames, shape = 40, (23, 45)
            hidden = np.zeros(shape, bool)
            hidden[5:15, 10:30] = True
            depth = np.full((frames, *shape), 300.0, np.float32)
            valid = np.ones((frames, *shape), bool)
            blocked_valid = valid & ~hidden
            prepared = root / 'prepared'
            prepared.mkdir()
            np.savez(prepared / 'p.npz', blocked_height_mm=depth, blocked_valid=blocked_valid,
                     blocked_timestamp_ms=np.arange(frames) * 1000 / 30, repeat_height_mm=depth,
                     repeat_valid=valid, conservative_hidden=hidden, reference_index=np.arange(frames))
            predictions = root / 'model'
            predictions.mkdir()
            origins = np.array([24, 30])
            np.savez(predictions / 'p.npz', origin_index=origins,
                     prediction_mm=np.full((2, *shape), 302.0, np.float32),
                     prediction_200ms_mm=np.full((2, *shape), 299.0, np.float32))
            transform = root / 'transform.json'
            transform.write_text(json.dumps({'camera_frame': 'depth', 'output_frame': 'printer',
                                             'status': 'test', 'depth_camera_to_printer': np.eye(4).tolist()}))
            report = rescore(prepared, transform, {'m': predictions}, root / 'out')
            result = report['results']['m']
            for frame in ('camera_depth', 'printer_vertical'):
                self.assertAlmostEqual(result[frame]['hidden']['mae_mm'], 2.0, places=4)
                self.assertAlmostEqual(result[frame]['hidden']['bias_mm'], 2.0, places=4)
                self.assertAlmostEqual(result[frame]['future_hidden']['bias_mm'], -1.0, places=4)
                self.assertAlmostEqual(result[frame]['persistence_visible']['mae_mm'], 0.0, places=4)
            self.assertEqual(result['camera_depth']['hidden']['pixels'], 2 * int(hidden.sum()))
            self.assertTrue((root / 'out/printer_frame_real_head_rescore.json').exists())
            with self.assertRaises(FileExistsError):
                rescore(prepared, transform, {'m': predictions}, root / 'out')


if __name__ == '__main__':
    unittest.main()

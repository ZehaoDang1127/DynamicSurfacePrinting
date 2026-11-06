"""Exercise native depth conversion with minimal, synthetic recordings."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from prediction.data import load_splits
from tools.convert_servo_depth import convert


class ServoDepthConversionTests(unittest.TestCase):
    def test_projection_masks_units_timestamps_and_recording_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source', root / 'output'
            source.mkdir()
            camera = {'depth_scale_m_per_unit': 0.0001, 'color': {'fps': 5},
                      'depth': {'width': 640, 'height': 480, 'fps': 5, 'fx': 390., 'fy': 390.,
                                'ppx': 322., 'ppy': 231., 'distortion_model': 'distortion.brown_conrady',
                                'coefficients': [0., 0., 0., 0., 0.]}}
            for i in range(20):
                session = source / f'{i + 1:02d}_pattern'
                (session / 'depth').mkdir(parents=True)
                (session / 'session.json').write_text(json.dumps({'status': 'complete', 'settings': {'fps': 5}, 'camera': camera}), encoding='utf-8')
                raw = np.zeros((480, 640), dtype=np.uint16)
                raw[231, 322] = 3000  # exactly 300 mm at camera X=Y=0
                raw[244, 335] = 3000  # 300 * 13 / 390 = 10 mm on X and Y
                raw[231, 244] = 3000  # outside image crop
                raw[231, 323] = 5000  # outside depth gate
                Image.fromarray(raw).save(session / 'depth' / '000000.png')
                with (session / 'frames.csv').open('w', newline='', encoding='utf-8') as f:
                    fields = ['depth_file', 'depth_frame_number', 'depth_timestamp_ms', 'host_monotonic_ns', 'depth_timestamp_domain']
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(dict(depth_file='depth/000000.png', depth_frame_number=7,
                                         depth_timestamp_ms=1234.5, host_monotonic_ns=987654321,
                                         depth_timestamp_domain='hardware_clock'))
            with contextlib.redirect_stdout(io.StringIO()):
                convert(source, output)
            splits = load_splits(output / 'splits.json')
            self.assertEqual({k: len(v) for k, v in splits.items()}, {'train': 16, 'validation': 2, 'test': 2})
            session = splits['train'][0]
            self.assertEqual(session.manifest['config']['camera']['fps'], 5)
            self.assertEqual(session.manifest['config']['frame_name'], 'camera_depth_frame_mm_not_printer_height')
            self.assertEqual(session.frames[0].camera_timestamp_ms, 1234.5)
            self.assertEqual(session.frames[0].host_monotonic_ns, 987654321)
            self.assertEqual(session.frames[0].timestamp_domain, 'hardware_clock')
            with np.load(session.frames[0].path, allow_pickle=False) as frame:
                valid = frame['valid']
                self.assertEqual(valid.shape, (23, 45))
                self.assertEqual(int(valid.sum()), 2)
                # x=0 and y=0 map to column 20, row 7 with 4 mm cells.
                self.assertTrue(valid[7, 20])
                self.assertAlmostEqual(float(frame['height_mm'][7, 20]), 300.)
                self.assertEqual(int(frame['point_count'][7, 20]), 1)
                self.assertTrue(valid[9, 22])
                self.assertAlmostEqual(float(frame['height_mm'][9, 22]), 300.)
                self.assertTrue(np.isnan(frame['height_mm'][~valid]).all())

            # Preserve a real 30-fps stream rate while retaining its source timestamps.
            files = sorted(source.glob('*/session.json'))
            for path in files:
                meta = json.loads(path.read_text(encoding='utf-8'))
                meta['settings']['fps'] = meta['camera']['depth']['fps'] = meta['camera']['color']['fps'] = 30
                path.write_text(json.dumps(meta), encoding='utf-8')
            output30 = root / 'output30'
            with contextlib.redirect_stdout(io.StringIO()):
                convert(source, output30)
            converted30 = load_splits(output30 / 'splits.json')['train'][0]
            self.assertEqual(converted30.manifest['config']['camera']['fps'], 30)
            self.assertEqual(converted30.frames[0].camera_timestamp_ms, 1234.5)

            # A stream disagreement or an independently consistent mixed-fps
            # session must not be represented as a uniform recording set.
            path = files[-1]
            meta = json.loads(path.read_text(encoding='utf-8'))
            meta['settings']['fps'] = 5
            path.write_text(json.dumps(meta), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'frame rates disagree'):
                convert(source, root / 'mismatch')
            meta['camera']['depth']['fps'] = meta['camera']['color']['fps'] = 5
            path.write_text(json.dumps(meta), encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'incompatible camera calibration'):
                convert(source, root / 'mixed')


if __name__ == '__main__':
    unittest.main()

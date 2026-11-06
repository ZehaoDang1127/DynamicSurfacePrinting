"""Convert verified FFV1 D405 recordings to camera-coordinate height maps.

This uses a scene-specific crop/depth gate, not verified surface segmentation;
camera-axis Z is not printer height. The source recordings are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from surface_mapper.config import AppConfig, CameraConfig, GridConfig


def convert(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    with (source / 'sequences.csv').open(newline='', encoding='utf-8-sig') as f:
        selected = list(csv.DictReader(f))
    if len(selected) != 20 or len({x['directory'] for x in selected}) != 20:
        raise ValueError('expected 20 unique selected sessions')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'output must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    grid = GridConfig(-80, 100, -28, 64, 4, 230, 450)
    config = AppConfig('camera_depth_frame_mm_not_printer_height',
                       CameraConfig(width=640, height=480, fps=30), grid, np.eye(4))
    split = {'train': [], 'validation': [], 'test': []}
    reference = None
    for index, selected_row in enumerate(selected, 1):
        name = selected_row['directory']
        if Path(name).name != name or selected_row['status'] != 'complete':
            raise ValueError(f'invalid selected session: {name}')
        session = source / name
        meta = json.loads((session / 'session.json').read_text(encoding='utf-8'))
        if meta.get('status') != 'complete':
            raise ValueError(f'incomplete session: {session}')
        depth = meta['camera']['depth']
        scale = meta['camera']['depth_scale_m_per_unit']
        signature = (depth['width'], depth['height'], depth['fps'], depth['fx'], depth['fy'],
                     depth['ppx'], depth['ppy'], depth['distortion_model'], tuple(depth['coefficients']), scale)
        if reference is None:
            reference = signature
        if signature != reference or signature[:3] != (640, 480, 30):
            raise ValueError(f'incompatible camera calibration: {session}')
        if depth['distortion_model'] not in ('distortion.brown_conrady', 'distortion.none') or any(abs(x) > 1e-12 for x in depth['coefficients']):
            raise ValueError('expected zero-distortion depth intrinsics')
        with (session / 'frames.csv').open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != int(selected_row['saved_pairs']) or len(rows) != meta['saved_pairs']:
            raise ValueError(f'frame count mismatch: {session}')
        xray = (np.arange(245, 420, dtype=np.float32) - depth['ppx']) / depth['fx']
        yray = (np.arange(205, 315, dtype=np.float32) - depth['ppy']) / depth['fy']
        target = output / name
        target.mkdir()
        cap = cv2.VideoCapture(str(session / 'depth.mkv'))
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_CONVERT_RGB, 0):
            raise ValueError(f'cannot open native-depth FFV1 decoder: {session}')
        records = []
        try:
            for i, row in enumerate(rows):
                if int(row['index']) != i:
                    raise ValueError(f'nonconsecutive video indices in {session}')
                ok, image = cap.read()
                if not ok or image.shape != (480, 640) or image.dtype != np.uint16:
                    raise ValueError(f'invalid decoded native depth frame {i}: {session}')
                # The source hashes decoded native little-endian pixel bytes.
                if hashlib.sha256(image.astype('<u2', copy=False).tobytes()).hexdigest() != row['depth_sha256']:
                    raise ValueError(f'depth checksum mismatch at frame {i}: {session}')
                z = image[205:315, 245:420].astype(np.float32) * (scale * 1000)
                x, y = z * xray[None, :], z * yray[:, None]
                keep = ((z >= grid.z_min_mm) & (z <= grid.z_max_mm) & (x >= grid.x_min_mm) &
                        (x < grid.x_max_mm) & (y >= grid.y_min_mm) & (y < grid.y_max_mm))
                ix = np.floor((x[keep] - grid.x_min_mm) / grid.cell_size_mm).astype(np.int64)
                iy = np.floor((y[keep] - grid.y_min_mm) / grid.cell_size_mm).astype(np.int64)
                ids = iy * grid.width + ix
                count = np.bincount(ids, minlength=grid.height * grid.width).astype(np.uint32)
                sums = np.bincount(ids, weights=z[keep], minlength=count.size)
                valid = count > 0
                height = np.full(count.size, np.nan, dtype=np.float32)
                height[valid] = sums[valid] / count[valid]
                shape = (grid.height, grid.width)
                height, valid, count = height.reshape(shape), valid.reshape(shape), count.reshape(shape)
                number, stamp, host = int(row['depth_frame_number']), float(row['depth_timestamp_ms']), int(row['host_monotonic_ns'])
                filename = f'frame_{number:06d}.npz'
                np.savez_compressed(target / filename, height_mm=height, valid=valid, point_count=count,
                                    frame_number=np.int64(number), camera_timestamp_ms=np.float64(stamp), host_monotonic_ns=np.int64(host))
                records.append({'file': filename, 'frame_number': number, 'camera_timestamp_ms': stamp,
                                'host_monotonic_ns': host, 'timestamp_domain': row['depth_timestamp_domain'],
                                'valid_fraction': float(valid.mean())})
            if cap.read()[0]:
                raise ValueError(f'video contains more frames than frames.csv: {session}')
        finally:
            cap.release()
        manifest = {'schema_version': 1, 'complete': True, 'config': config.to_dict(),
                    'grid': {'shape': [grid.height, grid.width], 'x_centers_mm': grid.x_centers_mm().tolist(),
                             'y_centers_mm': grid.y_centers_mm().tolist()},
                    'frame_count': len(records), 'frames': records,
                    'provenance': {'source_session': str(session), 'depth_camera': depth, 'depth_scale_m_per_unit': scale,
                                   'selection': 'depth pixels x=[245,420), y=[205,315), camera Z=[230,450] mm',
                                   'depth_integrity': 'each decoded native uint16 frame checked against source SHA-256',
                                   'warning': 'Scene-specific crop/depth gate approximates surface; no measured printer transform or segmentation. Z is camera-axis depth.'}}
        (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        subset = 'validation' if index in (15, 19) else 'test' if index in (18, 20) else 'train'
        split[subset].append(name)
        print(f'{index}/20 {name}: {len(records)} frames -> {subset}', flush=True)
    (output / 'splits.json').write_text(json.dumps(split, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.output)

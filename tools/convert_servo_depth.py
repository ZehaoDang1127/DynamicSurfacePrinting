"""Convert verified D405 depth recordings to camera-coordinate surface maps.

The conservative image-space crop and depth gate are scene-specific heuristics,
not a calibrated cloth segmentation. Z is camera depth, not printer height.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from surface_mapper.config import AppConfig, CameraConfig, GridConfig


def convert(source: Path, output: Path) -> None:
    sessions = sorted(p for p in source.iterdir() if p.is_dir() and (p / 'session.json').exists())
    if len(sessions) != 20:
        raise ValueError('expected the verified set of exactly 20 recordings')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    grid = GridConfig(-80, 100, -28, 64, 4, 230, 450)
    ref = None
    split = {'train': [], 'validation': [], 'test': []}
    for index, session in enumerate(sessions, 1):
        meta = json.loads((session / 'session.json').read_text(encoding='utf-8'))
        if meta.get('status') != 'complete':
            raise ValueError(f'incomplete recording: {session}')
        camera = meta['camera']
        depth = camera['depth']
        fps = depth['fps']
        if type(fps) is not int or fps <= 0:
            raise ValueError(f'invalid depth stream frame rate: {session}')
        if meta.get('settings', {}).get('fps') != fps or camera.get('color', {}).get('fps') != fps:
            raise ValueError(f'requested, color, and depth frame rates disagree: {session}')
        intrinsics = (depth['width'], depth['height'], depth['fx'], depth['fy'], depth['ppx'], depth['ppy'],
                      depth['distortion_model'], tuple(depth['coefficients']), camera['depth_scale_m_per_unit'], fps)
        if ref is None:
            ref = intrinsics
        if intrinsics != ref or intrinsics[:2] != (640, 480):
            raise ValueError(f'incompatible camera calibration or depth frame rate: {session}')
        if depth['distortion_model'] not in ('distortion.brown_conrady', 'distortion.none') or any(abs(x) > 1e-12 for x in depth['coefficients']):
            raise ValueError('converter only supports zero-distortion Brown-Conrady or undistorted depth')
        config = AppConfig('camera_depth_frame_mm_not_printer_height', CameraConfig(width=640, height=480, fps=fps), grid, np.eye(4))
        fx, fy, ppx, ppy = (depth[k] for k in ('fx', 'fy', 'ppx', 'ppy'))
        xray = (np.arange(245, 420, dtype=np.float32) - ppx) / fx
        yray = (np.arange(205, 315, dtype=np.float32) - ppy) / fy
        target = output / session.name
        target.mkdir()
        records = []
        with (session / 'frames.csv').open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            image = np.asarray(Image.open(session / row['depth_file']))
            if image.shape != (480, 640) or image.dtype.kind not in 'ui':
                raise ValueError(f'invalid depth image: {row["depth_file"]}')
            z = image[205:315, 245:420].astype(np.float32) * (camera['depth_scale_m_per_unit'] * 1000)
            x = z * xray[None, :]
            y = z * yray[:, None]
            keep = (z >= grid.z_min_mm) & (z <= grid.z_max_mm) & (x >= grid.x_min_mm) & (x < grid.x_max_mm) & (y >= grid.y_min_mm) & (y < grid.y_max_mm)
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
            number = int(row['depth_frame_number'])
            stamp = float(row['depth_timestamp_ms'])
            host = int(row['host_monotonic_ns'])
            domain = row['depth_timestamp_domain']
            filename = f'frame_{number:06d}.npz'
            np.savez_compressed(target / filename, height_mm=height, valid=valid, point_count=count,
                                frame_number=np.int64(number), camera_timestamp_ms=np.float64(stamp), host_monotonic_ns=np.int64(host))
            records.append({'file': filename, 'frame_number': number, 'camera_timestamp_ms': stamp,
                            'host_monotonic_ns': host, 'timestamp_domain': domain, 'valid_fraction': float(valid.mean())})
        manifest = {'schema_version': 1, 'complete': True, 'config': config.to_dict(),
                    'grid': {'shape': [grid.height, grid.width], 'x_centers_mm': grid.x_centers_mm().tolist(),
                             'y_centers_mm': grid.y_centers_mm().tolist()},
                    'frame_count': len(records), 'frames': records,
                    'provenance': {'source_session': str(session.resolve()), 'depth_camera': depth,
                        'depth_scale_m_per_unit': camera['depth_scale_m_per_unit'],
                        'selection': 'depth pixels x=[245,420), y=[205,315), camera Z=[230,450] mm',
                        'warning': 'Scene-specific crop/depth gate approximates cloth support; no measured printer transform or ground-truth segmentation. Z is camera-axis depth.'}}
        (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        subset = 'validation' if index in (15, 19) else 'test' if index in (18, 20) else 'train'
        split[subset].append(session.name)
        print(f'{index}/20 {session.name}: {len(records)} frames -> {subset}', flush=True)
    (output / 'splits.json').write_text(json.dumps(split, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.output)

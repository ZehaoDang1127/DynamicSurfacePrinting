"""Prepare conservative real-head observations and approximate repeat references.

The recordings are separate runs, not synchronized ground truth. Rectangular
head coverage, pose stability and motion repeatability are assumptions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SHAPE = (23, 45)
# Hand-inspected RGB head bounds padded substantially in both axes. This is
# applied to native depth coordinates as a conservative approximation, not a
# calibrated per-frame segmentation. Clipped to the training crop below.
HEAD_RECT = (275, 80, 380, 260)  # hand-inspected RGB head bounding box
ROI = (245, 205, 420, 315)


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def training_envelope(split_path, depth_margin):
    """Training-only observed camera-depth extrema, sampled uniformly by session."""
    split = json.loads(split_path.read_text(encoding='utf-8'))
    frames = []
    provenance = None
    for name in split['train']:
        session = split_path.parent / name
        manifest = json.loads((session / 'manifest.json').read_text(encoding='utf-8'))
        current = manifest['provenance']
        metadata = {'depth_camera':current['depth_camera'], 'depth_scale_m_per_unit':current['depth_scale_m_per_unit']}
        if provenance is None:
            provenance = metadata
        elif provenance != metadata:
            raise ValueError('training recordings have incompatible depth calibration')
        fs = manifest['frames']
        for i in np.unique(np.linspace(0, len(fs)-1, min(120, len(fs)), dtype=int)):
            with np.load(session / fs[int(i)]['file'], allow_pickle=False) as data:
                frames.append(np.where(data['valid'], data['height_mm'], np.nan))
    stack = np.stack(frames)
    have = np.isfinite(stack).any(axis=0)
    lo = np.min(np.where(np.isfinite(stack), stack, np.inf), axis=0)
    hi = np.max(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    return np.clip(lo-depth_margin,230,450), np.clip(hi+depth_margin,230,450), have, len(frames), provenance


def make_cone(camera, envelope, margin):
    """Project training-only depth envelope for each cell into the color view."""
    lo, hi, have, *_ = envelope
    color = camera['color']
    extrinsic = camera['depth_to_color_extrinsics']
    rotation = np.array(extrinsic['rotation_column_major']).reshape(3, 3, order='F')
    translation = np.array(extrinsic['translation_m']) * 1000
    yy, xx = np.indices(SHAPE)
    x = -80 + (xx + .5) * 4
    y = -28 + (yy + .5) * 4
    mask = ~have.copy()
    # Cell corners and depth envelope extremes/interior cover the projected cell.
    for dx in (-2., 0., 2.):
        for dy in (-2., 0., 2.):
            for fraction in np.linspace(0, 1, 9):
                z = np.where(have, lo + fraction*(hi-lo), 300)
                points = np.stack((x+dx,y+dy,z), axis=-1) @ rotation.T + translation
                norm = points[..., :2] / points[..., 2:3]
                # Inverse-Brown model maps distorted to undistorted coordinates;
                # solve its inverse via OpenCV's iterative undistortion routine.
                distorted = cv2.undistortPoints(norm.reshape(-1,1,2), np.eye(3), np.array(color['coefficients'])).reshape(*SHAPE,2)
                u = distorted[...,0]*color['fx'] + color['ppx']
                v = distorted[...,1]*color['fy'] + color['ppy']
                mask |= ((u >= HEAD_RECT[0]-margin) & (u < HEAD_RECT[2]+margin) &
                         (v >= HEAD_RECT[1]-margin) & (v < HEAD_RECT[3]+margin))
    return mask


def project(raw, camera):
    d = camera['depth']
    z = raw[205:315, 245:420].astype(np.float32) * (camera['depth_scale_m_per_unit'] * 1000)
    x = z * ((np.arange(245, 420, dtype=np.float32) - d['ppx']) / d['fx'])[None, :]
    y = z * ((np.arange(205, 315, dtype=np.float32) - d['ppy']) / d['fy'])[:, None]
    good = (z >= 230) & (z <= 450) & (x >= -80) & (x < 100) & (y >= -28) & (y < 64)
    col = np.floor((x[good] + 80) / 4).astype(np.int64)
    row = np.floor((y[good] + 28) / 4).astype(np.int64)
    ids = row * SHAPE[1] + col
    counts = np.bincount(ids, minlength=np.prod(SHAPE))
    sums = np.bincount(ids, weights=z[good], minlength=counts.size)
    valid = counts > 0
    value = np.zeros(counts.size, np.float32)
    value[valid] = sums[valid] / counts[valid]
    return value.reshape(SHAPE), valid.reshape(SHAPE)


def read_video(directory, rows, camera, cone, blocked):
    cap = cv2.VideoCapture(str(directory / 'depth.mkv'))
    if not cap.isOpened() or not cap.set(cv2.CAP_PROP_CONVERT_RGB, 0):
        raise ValueError(f'cannot decode native depth: {directory}')
    heights = np.zeros((len(rows), *SHAPE), np.float32)
    valid = np.zeros((len(rows), *SHAPE), bool)
    try:
        for i, entry in enumerate(rows):
            ok, raw = cap.read()
            if not ok or raw.shape != (480, 640) or raw.dtype != np.uint16:
                raise ValueError(f'invalid native uint16 depth at {directory} frame {i}')
            if hashlib.sha256(raw.astype('<u2', copy=False).tobytes()).hexdigest() != entry['depth_sha256']:
                raise ValueError(f'depth checksum mismatch at {directory} frame {i}')
            heights[i], valid[i] = project(raw, camera)
            if blocked:
                valid[i] &= ~cone
                heights[i, ~valid[i]] = 0
    finally:
        cap.release()
    return heights, valid


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {'count': int(a.size), 'mae_mm': float(np.mean(np.abs(a))) if a.size else None,
            'rmse_mm': float(np.sqrt(np.mean(a*a))) if a.size else None,
            'median_absolute_mm': float(np.median(np.abs(a))) if a.size else None,
            'p95_absolute_mm': float(np.percentile(np.abs(a), 95)) if a.size else None,
            'bias_blocked_minus_repeat_mm': float(np.mean(a)) if a.size else None}


def prepare(blocked_root, reference_root, output, training_splits, pixel_margin=20, depth_margin=20):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'output must be new or empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    patterns = sorted(p for p in blocked_root.iterdir() if (p / 'session.json').is_file())
    if not patterns:
        raise ValueError('no test sessions')
    envelope = training_envelope(training_splits, depth_margin)
    report = {'warning': 'Separate repeat runs; phase pairing and cell alignment are approximate. Repeat-reference values are not simultaneous hidden ground truth. Fixed features show shifted viewpoint between runs.',
              'roi_depth_pixels_xyxy_exclusive': ROI, 'head_rect_color_pixels_xyxy_exclusive': HEAD_RECT,
              'mask_note': 'Hand-inspected RGB box, pixel margin, and training-only depth envelope projected using depth-to-color factory geometry; preliminary approximation, not calibrated segmentation.',
              'training_split_path': str(training_splits.resolve()), 'training_frames_sampled':envelope[3],
              'training_depth_calibration': envelope[4],
              'training_depth_envelope_margin_mm':depth_margin,'head_box_pixel_margin':pixel_margin,
              'grid': 'camera XY: x=[-80,100), y=[-28,64), 4 mm cells; Z=230..450 mm; camera depth, not printer Z',
              'matching': 'nearest host receipt elapsed_s from each recording start, <=20 ms; software commanded motion only',
              'patterns': {}}
    aggregate = []
    for folder in patterns:
        reference = reference_root / folder.name
        bm = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
        rm = json.loads((reference / 'session.json').read_text(encoding='utf-8'))
        if bm['status'] != 'complete' or rm['status'] != 'complete' or bm['camera'] != rm['camera']:
            raise ValueError(f'incomplete or incompatible camera metadata: {folder.name}')
        camera = bm['camera']; depth = camera['depth']
        if {'depth_camera':depth, 'depth_scale_m_per_unit':camera['depth_scale_m_per_unit']} != envelope[4]:
            raise ValueError(f'test and training depth calibration differ: {folder.name}')
        if 'test_camera' not in report:
            report['test_camera'] = camera
        elif camera != report['test_camera']:
            raise ValueError(f'test camera calibration changes: {folder.name}')
        if (depth['width'],depth['height'],depth['fps']) != (640,480,30) or any(depth['coefficients']):
            raise ValueError(f'unsupported geometry for {folder.name}')
        if bm['trajectory'] != rm['trajectory']:
            raise ValueError(f'commanded trajectories differ: {folder.name}')
        br, rr = read_rows(folder / 'frames.csv'), read_rows(reference / 'frames.csv')
        cone = make_cone(camera, envelope, pixel_margin)
        if 'mask_sensitivity_grid_cells' not in report:
            report['mask_sensitivity_grid_cells'] = {str(m):int(make_cone(camera,envelope,m).sum()) for m in (0,10,20,30)}
        bh, bv = read_video(folder, br, camera, cone, True)
        rh, rv = read_video(reference, rr, camera, cone, False)
        bt = np.array([float(x['elapsed_s']) for x in br])
        rt = np.array([float(x['elapsed_s']) for x in rr])
        pos = np.searchsorted(rt, bt)
        left, right = np.clip(pos - 1, 0, len(rt)-1), np.clip(pos, 0, len(rt)-1)
        match = np.where(np.abs(rt[left]-bt) <= np.abs(rt[right]-bt), left, right)
        gap = rt[match] - bt
        match[np.abs(gap) > .020] = -1
        errors = []
        deltas = []
        for i, j in enumerate(match):
            if j < 0:
                continue
            support = bv[i] & rv[j] & ~cone
            errors.append((bh[i][support] - rh[j][support]).astype(np.float32))
            deltas.append(float(br[i]['commanded_angle_at_receive_deg'])-float(rr[j]['commanded_angle_at_receive_deg']))
        errors = np.concatenate(errors) if errors else np.array([])
        aggregate.append(errors)
        np.savez_compressed(output / (folder.name + '.npz'), blocked_height_mm=bh, blocked_valid=bv,
                            repeat_height_mm=rh, repeat_valid=rv, conservative_hidden=cone,
                            blocked_timestamp_ms=np.array([float(r['depth_timestamp_ms']) for r in br]),
                            blocked_elapsed_s=bt, repeat_elapsed_s=rt, reference_index=match,
                            pair_elapsed_difference_s=gap,
                            blocked_frame_number=np.array([int(r['depth_frame_number']) for r in br]))
        report['patterns'][folder.name] = {'blocked_frames':len(br), 'repeat_frames':len(rr),
            'matched_frames':int(np.sum(match >= 0)), 'conservative_hidden_grid_cells':int(cone.sum()),
            'mean_blocked_observed_fraction':float(bv.mean()),
            'phase_offset_abs_p95_ms':float(np.percentile(np.abs(gap[match>=0]),95)*1000) if np.any(match>=0) else None,
            'commanded_angle_delta_abs_p95_deg':float(np.percentile(np.abs(deltas),95)) if deltas else None,
            'visible_repeat_disagreement':stats(errors)}
        print(folder.name, report['patterns'][folder.name], flush=True)
    report['overall_visible_repeat_disagreement'] = stats(np.concatenate(aggregate))
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blocked', type=Path, required=True)
    p.add_argument('--repeat-reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-splits', type=Path, required=True)
    p.add_argument('--pixel-margin', type=int, default=20)
    p.add_argument('--depth-margin-mm', type=float, default=20)
    a = p.parse_args()
    if a.pixel_margin < 0 or a.depth_margin_mm < 0:
        p.error('margins must be nonnegative')
    prepare(a.blocked, a.repeat_reference, a.output, a.training_splits, a.pixel_margin, a.depth_margin_mm)

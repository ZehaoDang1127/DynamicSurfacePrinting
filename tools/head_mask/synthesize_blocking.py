"""Create masked image-space sequence crops from a CLEAN RGB recording.

ROI masks must describe the source frames and share their color coordinates.
Use --aligned-depth-dir only for 16-bit depth already registered to color.
Original color/depth frames are never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Visible portion of the measured head, shifted to a local origin, including
# the concave bottom edge. This is a mask shape, not a photorealistic rendering.
SHAPE = np.array([(0, 0), (83, 0), (82, 24), (77, 35), (71, 38),
                  (56, 38), (52, 32), (32, 32), (28, 38), (13, 38),
                  (5, 32), (2, 21)], dtype=np.float64)
SHAPE -= (41.5, 19)


def read(path, flags):
    image = cv2.imdecode(np.fromfile(path, np.uint8), flags)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def write(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError(f'Cannot encode image: {path}')
    encoded.tofile(path)


def silhouette(roi, center, scale):
    points = np.rint(SHAPE * scale + center).astype(np.int32)
    shape = np.zeros(roi.shape, np.uint8)
    cv2.fillPoly(shape, [points], 255)
    return cv2.bitwise_and(shape, roi)


def fit_mask(roi, center, fraction):
    """Fit scale to the target area at a fixed center; retain the closest raster result."""
    target = fraction * cv2.countNonZero(roi)
    low, high = 0.0, 1.0
    while cv2.countNonZero(silhouette(roi, center, high)) < target and high < 32:
        high *= 2
    best, best_count, best_scale = None, 0, 0.0
    for _ in range(20):
        scale = (low + high) / 2
        mask = silhouette(roi, center, scale)
        count = cv2.countNonZero(mask)
        if best is None or abs(count - target) < abs(best_count - target):
            best, best_count, best_scale = mask, count, scale
        if count < target:
            low = scale
        else:
            high = scale
    return best, best_count, best_scale


def load_roi(directory, name, full_shape, crop):
    path = directory / name
    mask = read(path, cv2.IMREAD_UNCHANGED)
    if mask.ndim != 2:
        raise ValueError(f'ROI mask must be single-channel: {path}')
    x0, y0, x1, y1 = crop
    crop_shape = (y1-y0, x1-x0)
    if mask.shape == full_shape:
        mask = mask[y0:y1, x0:x1]
    elif mask.shape != crop_shape:
        raise ValueError(f'ROI mask dimensions must equal full image or crop: {path}, {mask.shape}')
    mask = np.where(mask != 0, 255, 0).astype(np.uint8)
    if cv2.countNonZero(mask) == 0:
        raise ValueError(f'Empty ROI mask: {path}')
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path, help='clean recording with frames.csv and color paths')
    parser.add_argument('output', type=Path, help='new output directory')
    roi_source = parser.add_mutually_exclusive_group(required=True)
    roi_source.add_argument('--roi-masks', type=Path,
                            help='per-frame masks named by zero-padded frame index, in full-color or crop coordinates')
    roi_source.add_argument('--same-camera-view', action='store_true',
                            help='estimate ROI in each clean image using the original recording geometry; inspect results')
    parser.add_argument('--crop', nargs=4, type=int, required=True, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='color pixel bounds, x1/y1 exclusive')
    parser.add_argument('--aligned-depth-dir', type=Path,
                        help='OPTIONAL: 16-bit depth PNGs ALREADY registered to full color coordinates')
    parser.add_argument('--fraction', type=float, default=.1225)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tolerance', type=float, default=.005,
                        help='maximum absolute per-frame fraction error (default 0.005)')
    args = parser.parse_args()
    if not 0 < args.fraction < 1 or not 0 < args.tolerance < 1:
        parser.error('fraction and tolerance must each be strictly between 0 and 1')
    if args.output.exists():
        raise FileExistsError(f'Output must not already exist: {args.output}')
    x0, y0, x1, y1 = args.crop
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        parser.error('crop must be a non-empty, non-negative half-open rectangle')
    with (args.session / 'frames.csv').open(newline='', encoding='utf-8-sig') as f:
        source = list(csv.DictReader(f))
    if not source or not {'index', 'color_file'} <= source[0].keys():
        raise ValueError('frames.csv needs index and color_file columns and at least one row')
    indices = [int(row['index']) for row in source]
    if len(indices) != len(set(indices)):
        raise ValueError('Frame indices must be unique')
    first = read(args.session / source[0]['color_file'], cv2.IMREAD_COLOR)
    full_shape = first.shape[:2]
    if x1 > full_shape[1] or y1 > full_shape[0]:
        raise ValueError('crop exceeds color image bounds')
    if args.same_camera_view:
        import analyze_blocking
        if full_shape != (480, 640) or tuple(args.crop) != analyze_blocking.CROP:
            raise ValueError('same-camera-view requires 640x480 color and crop 95 230 339 376')

    def roi_for(image, index):
        if args.same_camera_view:
            edge, _ = analyze_blocking.front_edge(image)
            roi = analyze_blocking.surface_mask(edge)[y0:y1, x0:x1]
            if cv2.countNonZero(roi) == 0:
                raise ValueError(f'Empty inferred ROI at frame {index}')
            return roi
        return load_roi(args.roi_masks, f'{index:06d}.png', full_shape, args.crop)

    first_roi = roi_for(first, indices[0])
    # Sample once per sequence. Center is fixed, avoiding frame-to-frame jumps;
    # only scale adjusts for changing ROI area. Restrict center to the ROI.
    rng = np.random.default_rng(args.seed)
    locations = np.argwhere(first_roi != 0)
    center = None
    for choice in rng.permutation(len(locations))[:min(len(locations), 1000)]:
        y, x = locations[choice]
        candidate = np.array([x, y], dtype=float)
        _, count, _ = fit_mask(first_roi, candidate, args.fraction)
        if abs(count / cv2.countNonZero(first_roi) - args.fraction) <= args.tolerance:
            center = candidate
            break
    if center is None:
        raise ValueError('Could not place a mask with requested fraction and tolerance in first ROI')
    args.output.mkdir(parents=True)
    folders = ['clean_color', 'blocked_color', 'surface_mask', 'blocking_mask']
    if args.aligned_depth_dir is not None:
        folders += ['clean_aligned_depth', 'blocked_aligned_depth']
    for folder in folders:
        (args.output / folder).mkdir()
    result_rows = []
    for row, index in zip(source, indices):
        name = f'{index:06d}.png'
        color = read(args.session / row['color_file'], cv2.IMREAD_COLOR)
        if color.shape[:2] != full_shape:
            raise ValueError(f'Color image size changed at frame {index}')
        roi = roi_for(color, index)
        mask, count, scale = fit_mask(roi, center, args.fraction)
        roi_count = cv2.countNonZero(roi)
        actual = count / roi_count
        if abs(actual - args.fraction) > args.tolerance:
            raise ValueError(f'Frame {index} fraction {actual:.5f} exceeds tolerance; output is incomplete')
        clean = color[y0:y1, x0:x1].copy()
        clean[roi == 0] = 0
        blocked = clean.copy()
        blocked[mask != 0] = 0
        for folder, image in [('clean_color', clean), ('blocked_color', blocked),
                              ('surface_mask', roi), ('blocking_mask', mask)]:
            write(args.output / folder / name, image)
        if args.aligned_depth_dir is not None:
            depth = read(args.aligned_depth_dir / name, cv2.IMREAD_UNCHANGED)
            if depth.shape != full_shape or depth.dtype != np.uint16:
                raise ValueError(f'Frame {index} depth must be full-color-sized uint16 single-channel PNG')
            clean_depth = depth[y0:y1, x0:x1].copy()
            clean_depth[roi == 0] = 0
            blocked_depth = clean_depth.copy()
            blocked_depth[mask != 0] = 0
            write(args.output / 'clean_aligned_depth' / name, clean_depth)
            write(args.output / 'blocked_aligned_depth' / name, blocked_depth)
        output_row = dict(row)
        output_row['source_color_file'] = row['color_file']
        output_row['source_depth_file'] = row.get('depth_file', '')
        output_row['color_file'] = f'blocked_color/{name}'
        output_row['clean_color_file'] = f'clean_color/{name}'
        output_row['surface_mask_file'] = f'surface_mask/{name}'
        output_row['blocking_mask_file'] = f'blocking_mask/{name}'
        output_row['depth_file'] = f'blocked_aligned_depth/{name}' if args.aligned_depth_dir else ''
        output_row['clean_aligned_depth_file'] = f'clean_aligned_depth/{name}' if args.aligned_depth_dir else ''
        output_row.update({'surface_pixels': roi_count, 'blocked_pixels': count,
                           'blocked_fraction': actual, 'shape_scale': scale})
        result_rows.append(output_row)
    with (args.output / 'frames.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    values = [r['blocked_fraction'] for r in result_rows]
    summary = {'status': 'complete', 'source_session': str(args.session.resolve()),
               'roi_masks': str(args.roi_masks.resolve()) if args.roi_masks else None,
               'roi_method': 'same-camera-view estimate' if args.same_camera_view else 'per-frame supplied masks',
               'aligned_depth_dir': str(args.aligned_depth_dir.resolve()) if args.aligned_depth_dir else None,
               'frame_count': len(result_rows), 'crop_xyxy_exclusive': args.crop,
               'seed': args.seed, 'target_fraction': args.fraction, 'tolerance': args.tolerance,
               'center_xy_in_crop': center.tolist(), 'coverage_mean': float(np.mean(values)),
               'coverage_min': float(min(values)), 'coverage_max': float(max(values)),
               'notes': ['Black regions represent missing observations, not a rendered nozzle.',
                         'Center fixed for temporal consistency; scale adjusts to changing ROI area.',
                         'Same-camera-view ROI estimator assumes the reference camera setup and needs visual review.',
                         'No native-depth registration is performed; aligned depth must be supplied explicitly.',
                         'Source must be clean; the script cannot verify hidden surface ground truth.']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

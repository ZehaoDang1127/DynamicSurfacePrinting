"""Preliminary 3D fixed-scene registration of a repeated D405 pair.

The output maps ground-truth-run *depth-camera* coordinates to test-run
depth-camera coordinates in millimetres. It is a diagnostic, not a validated
camera calibration or proof that repeated surface shapes are identical.

Uses first RGB/depth frames, SIFT matches in outer image strips, an approximate
color-to-depth association based on the recorded intrinsics/extrinsics, and
RANSAC point-to-point rigid fitting. Run from the project root, e.g.::

    .venv\Scripts\python.exe -m tools.estimate_pair_registration \
      --reference Dataset/Test_groundTruth/21_amplitude_beats \
      --test Dataset/Test/21_amplitude_beats \
      --output output/servo30_registration/registration.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_first(path: Path, stream: str) -> np.ndarray:
    capture = cv2.VideoCapture(str(path / f'{stream}.mkv'))
    if stream == 'depth':
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise ValueError(f'Cannot decode first {stream} frame in {path}')
    if stream == 'depth' and (image.shape != (480, 640) or image.dtype != np.uint16):
        raise ValueError('Depth decoder must return native 640x480 uint16 frames')
    if stream == 'color' and image.shape != (480, 640, 3):
        raise ValueError('Color decoder must return 640x480 RGB/BGR frames')
    return image


def color_feature_point(uv: np.ndarray, depth: np.ndarray, camera: dict):
    """Associate one color feature to nearby depth, with approximations.

    Inverse Brown-Conrady color deprojection is evaluated on the feature ray.
    A nominal 350 mm point is inverse-transformed to locate a depth pixel;
    the small recorded depth-to-color translation makes this a useful first
    approximation, but it is not an SDK-verified color/depth registration.
    """
    color, dep = camera['color'], camera['depth']
    if any(abs(v) > 1e-12 for v in dep['coefficients']):
        raise ValueError('This diagnostic requires zero depth distortion')
    if color['distortion_model'] != 'distortion.inverse_brown_conrady':
        raise ValueError('Unexpected color distortion model')
    x = (float(uv[0]) - color['ppx']) / color['fx']
    y = (float(uv[1]) - color['ppy']) / color['fy']
    k1, k2, p1, p2, k3 = color['coefficients']
    r2 = x*x + y*y
    f = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    ray = np.array([x*f + 2*p1*x*y + p2*(r2 + 2*x*x),
                    y*f + 2*p2*x*y + p1*(r2 + 2*y*y), 1.])
    extr = camera['depth_to_color_extrinsics']
    rotation = np.asarray(extr['rotation_column_major']).reshape(3, 3, order='F')
    translation = np.asarray(extr['translation_m'])
    depth_ray = rotation.T @ (ray*.35 - translation)
    u = int(round(dep['fx']*depth_ray[0]/depth_ray[2] + dep['ppx']))
    v = int(round(dep['fy']*depth_ray[1]/depth_ray[2] + dep['ppy']))
    if u < 2 or u >= 638 or v < 2 or v >= 478:
        return None
    scale = camera['depth_scale_m_per_unit']*1000
    region = depth[v-1:v+2, u-1:u+2].astype(float)*scale
    region = region[(region > 150) & (region < 800)]
    if len(region) < 5 or np.ptp(region) > 15:
        return None
    z = float(np.median(region))
    return np.array([(u-dep['ppx'])/dep['fx']*z,
                     (v-dep['ppy'])/dep['fy']*z, z])


def rigid_fit(source: np.ndarray, target: np.ndarray):
    src_mean, dst_mean = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source-src_mean).T @ (target-dst_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, dst_mean - rotation @ src_mean


def errors(source, target, rotation, translation):
    return np.linalg.norm(source @ rotation.T + translation - target, axis=1)


def estimate(source, target):
    if len(source) < 5:
        raise ValueError('Need at least five 3D correspondences')
    rng = np.random.default_rng(4)
    best = None
    for _ in range(4000):
        idx = rng.choice(len(source), 3, replace=False)
        rotation, translation = rigid_fit(source[idx], target[idx])
        error = errors(source, target, rotation, translation)
        inliers = error < 4
        score = (int(inliers.sum()), -float(np.median(error[inliers])) if inliers.any() else -999.)
        if best is None or score > best[0]:
            best = (score, inliers)
    inliers = best[1]
    if inliers.sum() < 5:
        raise ValueError('Fewer than five provisional 4 mm inliers')
    rotation, translation = rigid_fit(source[inliers], target[inliers])
    return rotation, translation, inliers


def stats(values):
    return {'count': len(values), 'median_mm': float(np.median(values)),
            'rms_mm': float(np.sqrt(np.mean(values**2))),
            'p95_mm': float(np.percentile(values, 95))}


def run(reference: Path, test: Path, output: Path):
    reference, test, output = reference.resolve(), test.resolve(), output.resolve()
    cameras = [json.loads((path/'session.json').read_text(encoding='utf-8'))['camera']
               for path in (reference, test)]
    rgb = [read_first(path, 'color') for path in (reference, test)]
    depth = [read_first(path, 'depth') for path in (reference, test)]
    mask = np.zeros((480, 640), np.uint8)
    mask[20:460, :130] = 255
    mask[20:460, 530:] = 255
    sift = cv2.SIFT_create(nfeatures=3000)
    features = [sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask) for frame in rgb]
    (key_ref, desc_ref), (key_test, desc_test) = features
    matches = [m for m, n in cv2.BFMatcher().knnMatch(desc_ref, desc_test, k=2)
               if m.distance < .7*n.distance]
    a = np.float32([key_ref[m.queryIdx].pt for m in matches])
    b = np.float32([key_test[m.trainIdx].pt for m in matches])
    if len(matches) < 5:
        raise ValueError('Insufficient SIFT matches')
    _, status = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2)
    if status is None:
        raise ValueError('Cannot find provisional 2D feature consensus')
    rows = []
    for uv_ref, uv_test in zip(a[status[:, 0] > 0], b[status[:, 0] > 0]):
        src = color_feature_point(uv_ref, depth[0], cameras[0])
        dst = color_feature_point(uv_test, depth[1], cameras[1])
        if src is not None and dst is not None:
            rows.append((uv_ref, uv_test, src, dst))
    source = np.asarray([row[2] for row in rows])
    target = np.asarray([row[3] for row in rows])
    rotation, translation, inliers = estimate(source, target)
    residual = errors(source, target, rotation, translation)
    left = np.asarray([row[0][0] < 320 for row in rows])
    heldout = {}
    for label, train, test_mask in (('left_to_right', left, ~left), ('right_to_left', ~left, left)):
        if train.sum() < 5 or test_mask.sum() < 2:
            heldout[label] = {'available': False}
            continue
        r, t, fit_inliers = estimate(source[train], target[train])
        heldout[label] = {'available': True, 'fit_count': int(train.sum()),
                          'fit_inliers': int(fit_inliers.sum()),
                          'held_out': stats(errors(source[test_mask], target[test_mask], r, t))}
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    result = {'status': 'provisional_diagnostic_not_calibrated',
              'direction': 'reference_depth_camera_mm_to_test_depth_camera_mm',
              'reference': str(reference), 'test': str(test), 'frame_index': 0,
              'transform_4x4_mm': transform.tolist(),
              'method': 'outer-strip SIFT; 2D affine consensus; approximate inverse-Brown color-to-depth association; 3D rigid 4mm RANSAC',
              'correspondences': len(rows), 'left_count': int(left.sum()), 'right_count': int((~left).sum()),
              'ransac_inliers_before_refit': int(inliers.sum()),
              'all_correspondence_residual': stats(residual),
              'fitted_inlier_residual': stats(residual[inliers]), 'spatial_holdout': heldout,
              'limitations': ['Color-depth association is approximate and not verified against RealSense SDK.',
                              'Fixed-scene features and their depth associations are not manually verified.',
                              'SIFT can report multiple orientations at one location; match counts are not independent physical points.',
                              '2D screening and depth-neighborhood selection affect the sample.',
                              'Rigid repeat-run registration cannot establish identical surface shape.',
                              'Do not use as validated nozzle-height ground truth or printer calibration.'],
              'points': [{'reference_rgb_xy': row[0].tolist(), 'test_rgb_xy': row[1].tolist(),
                          'reference_xyz_mm': row[2].tolist(), 'test_xyz_mm': row[3].tolist(),
                          'fit_inlier': bool(inlier), 'residual_mm': float(error)}
                         for row, inlier, error in zip(rows, inliers, residual)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Choose a new output file: {output}')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Wrote {output}; {len(rows)} matches, {int(inliers.sum())} fit inliers, '
          f'RMS {np.sqrt(np.mean(residual**2)):.2f} mm (all correspondences)')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.test, args.output)


if __name__ == '__main__':
    main()

"""Re-score saved real print-head predictions as vertical errors in the printer frame.

tools.evaluate_real_occlusion_repeats scores camera-depth differences along the D405's
line of sight. This tool reads the predictions that evaluation saved
(<pattern>.npz per pattern), maps predicted cells and the paired repeat reference to
printer XYZ with a depth-camera-to-printer registration (CameraCalibration.transform
``solve`` output, key ``depth_camera_to_printer``), and scores the predicted point's
printer Z against the reference surface's Z at the same printer XY (linear
interpolation over the reference frame's valid cells; points outside the reference
footprint take the nearest reference cell). Window selection, head mask, frame pairing
and the 200 ms rule follow tools/evaluate_real_occlusion_repeats.py, and camera-depth
errors are recomputed alongside as a check against each evaluation's metrics.json.

The registration must describe the camera pose at the time of the recordings. Pure
NumPy/SciPy; no model or GPU is needed.

Example:
    python -m tools.rescore_real_head_printer --transform output/registration_20261108/transform.json \
        --model mixed_finetune=output/eval_round2_20261108/mixed_finetune_real_head --output output/printer_rescore
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

# Prepared real-head maps use the training grid: 23 x 45 cells of 4 mm in camera XY.
X_CENTERS = -78 + 4 * np.arange(45, dtype=float)
Y_CENTERS = -26 + 4 * np.arange(23, dtype=float)
DT_MS, TOLERANCE, HISTORY, FUTURE_STEPS = 1000 / 30, 0.2, 25, 6
NAMES = ('hidden', 'visible', 'persistence_visible', 'future_hidden', 'future_visible', 'future_persistence_visible')


def stats(chunks):
    a = np.concatenate(chunks) if chunks else np.empty(0)
    a = a[np.isfinite(a)]
    if not a.size:
        return {'pixels': 0, 'mae_mm': None, 'rmse_mm': None, 'bias_mm': None, 'p95_abs_mm': None}
    return {'pixels': int(a.size), 'mae_mm': float(np.mean(np.abs(a))), 'rmse_mm': float(np.sqrt(np.mean(a * a))),
            'bias_mm': float(np.mean(a)), 'p95_abs_mm': float(np.percentile(np.abs(a), 95))}


class PrinterFrame:
    def __init__(self, depth_camera_to_printer):
        matrix = np.asarray(depth_camera_to_printer, dtype=float)
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError('depth_camera_to_printer must be a 4x4 rigid transform')
        self.rotation, self.translation = matrix[:3, :3], matrix[:3, 3]
        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-6) or np.linalg.det(self.rotation) <= 0:
            raise ValueError('depth_camera_to_printer rotation is not a proper rotation')
        self.x, self.y = np.meshgrid(X_CENTERS, Y_CENTERS)
        self.outside = self.points = 0

    def points_of(self, depth_mm):
        return np.stack([self.x, self.y, depth_mm.astype(float)], -1) @ self.rotation.T + self.translation

    def reference(self, depth_mm, valid):
        p = self.points_of(depth_mm)[valid]
        return LinearNDInterpolator(Delaunay(p[:, :2]), p[:, 2]), NearestNDInterpolator(p[:, :2], p[:, 2])

    def vertical_error(self, depth_mm, mask, reference):
        linear, nearest = reference
        p = self.points_of(depth_mm)[mask]
        z = linear(p[:, 0], p[:, 1])
        outside = ~np.isfinite(z)
        if outside.any():
            z[outside] = nearest(p[outside, 0], p[outside, 1])
        self.points += int(len(p))
        self.outside += int(outside.sum())
        return (p[:, 2] - z).astype(np.float32)


def timing_ok(timestamps):
    return not np.any(np.abs(np.diff(timestamps) - DT_MS) > DT_MS * TOLERANCE + 1e-6)


def rescore(prepared: Path, transform: Path, models: dict[str, Path], output: Path) -> dict:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'output must be new or empty: {output}')
    registration = json.loads(transform.read_text(encoding='utf-8-sig'))
    if registration.get('camera_frame') != 'depth' or registration.get('output_frame') != 'printer':
        raise ValueError('transform must map the depth camera frame to the printer frame')
    frame = PrinterFrame(registration['depth_camera_to_printer'])
    patterns = sorted(prepared.glob('*.npz'))
    if not patterns:
        raise FileNotFoundError(f'no prepared patterns in {prepared}')
    data, references, geometry = {}, {}, {}
    for path in patterns:
        with np.load(path, allow_pickle=False) as f:
            data[path.stem] = {k: f[k] for k in f.files}
        d = data[path.stem]
        references[path.stem] = {}
        # Sanity check of the registration on these recordings: a plane through every
        # 60th repeat frame, in printer coordinates.
        pts = np.concatenate([frame.points_of(d['repeat_height_mm'][j])[d['repeat_valid'][j]]
                              for j in range(0, len(d['repeat_height_mm']), 60)])
        coef = np.linalg.lstsq(np.c_[pts[:, :2], np.ones(len(pts))], pts[:, 2], rcond=None)[0]
        normal = np.array([-coef[0], -coef[1], 1.0]) / np.linalg.norm([coef[0], coef[1], 1.0])
        geometry[path.stem] = {'mean_printer_z_mm': float(pts[:, 2].mean()),
                               'plane_tilt_from_level_deg': float(np.degrees(np.arccos(normal[2])))}

    results = {}
    for model, folder in models.items():
        camera = {n: [] for n in NAMES}
        vertical = {n: [] for n in NAMES}
        frame.points = frame.outside = 0
        for path in patterns:
            d, cache = data[path.stem], references[path.stem]
            saved = folder / path.name
            if not saved.exists():
                raise FileNotFoundError(f'{saved} (run tools.evaluate_real_occlusion_repeats first)')
            with np.load(saved, allow_pickle=False) as f:
                origins, current, future = f['origin_index'], f['prediction_mm'], f['prediction_200ms_mm']
            h, v, t = d['blocked_height_mm'], d['blocked_valid'], d['blocked_timestamp_ms']
            refh, refv = d['repeat_height_mm'], d['repeat_valid']
            hidden, match = d['conservative_hidden'], d['reference_index']
            for k, i in enumerate(origins):
                past_valid, past_h = v[i - HISTORY + 1:i + 1], h[i - HISTORY + 1:i + 1]
                ever = past_valid.any(axis=0)
                persistence = np.take_along_axis(past_h[::-1], np.argmax(past_valid[::-1], axis=0)[None], axis=0)[0]
                leads = [('', current[k], match[i])]
                j_future = i + FUTURE_STEPS
                if j_future < len(t) and match[j_future] >= 0 and timing_ok(t[i:j_future + 1]):
                    leads.append(('future_', future[k], match[j_future]))
                for prefix, predicted, j in leads:
                    if j not in cache:
                        cache[j] = frame.reference(refh[j], refv[j])
                    for region, mask in (('hidden', hidden & refv[j]), ('visible', ~hidden & refv[j] & v[i])):
                        camera[prefix + region].append(predicted[mask] - refh[j][mask])
                        vertical[prefix + region].append(frame.vertical_error(predicted, mask, cache[j]))
                    supported = ~hidden & refv[j] & v[i] & ever
                    camera[prefix + 'persistence_visible'].append(persistence[supported] - refh[j][supported])
                    vertical[prefix + 'persistence_visible'].append(frame.vertical_error(persistence, supported, cache[j]))
        results[model] = {
            'predictions': str(folder),
            'camera_depth': {n: stats(camera[n]) for n in NAMES},
            'printer_vertical': {n: stats(vertical[n]) for n in NAMES},
            'vertical_component_of_camera_error': {n: stats([frame.rotation[2, 2] * np.asarray(x, float) for x in camera[n]])
                                                   for n in NAMES},
            'points_outside_reference_footprint': {'points': frame.points, 'outside': frame.outside},
        }
    report = {
        'schema_version': 1,
        'description': 'Vertical (printer Z) errors of saved real print-head predictions against the separate repeat recording.',
        'transform': str(transform.resolve()),
        'transform_status': registration.get('status'),
        'prepared': str(prepared.resolve()),
        'sign': 'prediction minus reference; printer Z up, so negative means the predicted surface is too low',
        'assumption': 'The registration describes the camera pose during the recordings.',
        'notes': ['The repeat recording is not simultaneous ground truth; persistence_visible measures how much the two recordings disagree.',
                  'camera_depth reproduces tools.evaluate_real_occlusion_repeats and should match its metrics.json.'],
        'recording_geometry_under_transform': geometry,
        'results': results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'printer_frame_real_head_rescore.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--transform', type=Path, required=True, help='transform.json with depth_camera_to_printer')
    parser.add_argument('--prepared', type=Path, default=Path('output/real_occlusion_prior_repeats_20261105'))
    parser.add_argument('--model', action='append', required=True, metavar='NAME=FOLDER',
                        help='real-head evaluation folder with saved <pattern>.npz predictions (repeatable)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    models = {}
    for item in args.model:
        name, sep, folder = item.partition('=')
        if not sep or not name or not folder:
            parser.error(f'--model expects NAME=FOLDER, got {item!r}')
        models[name] = Path(folder)
    report = rescore(args.prepared, args.transform, models, args.output)
    for name, result in report['results'].items():
        c, p = result['camera_depth'], result['printer_vertical']
        print(f"{name}: camera hidden {c['hidden']['mae_mm']:.2f}/{c['future_hidden']['mae_mm']:.2f} mm; "
              f"printer vertical hidden {p['hidden']['mae_mm']:.2f}/{p['future_hidden']['mae_mm']:.2f} mm "
              f"(bias {p['hidden']['bias_mm']:+.2f}); recordings on visible cells {p['persistence_visible']['mae_mm']:.2f} mm")


if __name__ == '__main__':
    main()

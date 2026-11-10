"""Score current surface estimates against an approximate separate repeat.

The reference is not simultaneous ground truth. Camera motion and the
preliminary head mask confound hidden and visible errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prediction.occlusion import load_checkpoint
from prediction.occlusion_data import encode_history


def stats(values):
    a = np.concatenate(values) if values else np.empty(0)
    return {'pixels': int(a.size), 'mae_mm': float(np.mean(np.abs(a))) if a.size else None,
            'rmse_mm': float(np.sqrt(np.mean(a*a))) if a.size else None,
            'p95_absolute_mm': float(np.percentile(np.abs(a),95)) if a.size else None,
            'bias_prediction_minus_repeat_mm': float(np.mean(a)) if a.size else None}


def evaluate(prepared, checkpoint, output, device, stride, batch_size, steps=(6,)):
    steps=tuple(sorted(set(int(k) for k in steps)))
    if not steps or steps[0] < 1:
        raise ValueError('forecast steps must be positive')
    rollout=max(steps)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'output must be new or empty: {output}')
    model, config, normalization, ckpt = load_checkpoint(checkpoint, device)
    if tuple(ckpt['shape']) != (23,45) or abs(config['expected_dt_ms']-1000/30)>1:
        raise ValueError('checkpoint must match the 23x45, 30-fps camera grid')
    signature=ckpt['geometry']
    geometry = signature['grid']
    for key, expected in {'x_min_mm':-80,'x_max_mm':100,'y_min_mm':-28,'y_max_mm':64,'cell_size_mm':4,
                          'z_min_mm':230,'z_max_mm':450,'min_points_per_cell':1,'aggregation':'mean'}.items():
        if geometry[key] != expected:
            raise ValueError('checkpoint geometry does not match prepared test maps')
    if (signature['frame_name'] != 'camera_depth_frame_mm_not_printer_height' or
        signature['camera'] != {'width':640,'height':480,'fps':30} or
        not np.allclose(signature['camera_to_output'],np.eye(4),rtol=0,atol=1e-12)):
        raise ValueError('checkpoint frame, camera stream, or transform differs from prepared maps')
    prepared_report=json.loads((prepared/'report.json').read_text(encoding='utf-8'))
    if not ckpt.get('train_sessions'):
        raise ValueError('checkpoint lacks training provenance needed to verify calibration')
    training_path=Path(ckpt['train_sessions'][0])/'manifest.json'
    training_provenance=json.loads(training_path.read_text(encoding='utf-8'))['provenance']
    expected_calibration={'depth_camera':training_provenance['depth_camera'],
                          'depth_scale_m_per_unit':training_provenance['depth_scale_m_per_unit']}
    if expected_calibration != prepared_report['training_depth_calibration']:
        raise ValueError('prepared maps and checkpoint training calibration differ')
    test_camera=prepared_report['test_camera']
    if {'depth_camera':test_camera['depth'], 'depth_scale_m_per_unit':test_camera['depth_scale_m_per_unit']} != expected_calibration:
        raise ValueError('test camera calibration differs from checkpoint training data')
    output.mkdir(parents=True, exist_ok=True)
    report = {'warning': 'Separate repeats, not simultaneous ground truth. Fixed scene features shift between runs; no 3D registration was validated. Mask and repeatability are preliminary; hidden errors are approximate.',
              'prepared':str(prepared.resolve()), 'checkpoint':str(checkpoint.resolve()),
              'stride':stride, 'rollout_steps':rollout, 'saved_forecast_steps':list(steps),
              'trained_horizon_steps':config['horizon'],
              'future_note':'200 ms target pairs repeat with blocked frame origin+6; future blocked measurements are not model inputs; hidden/visible support is defined at origin.',
              'patterns':{}}
    overall = {'hidden':[], 'visible':[], 'persistence_hidden':[], 'persistence_visible':[],
               'model_on_persistence_hidden':[], 'model_on_persistence_visible':[],
               'future_hidden':[], 'future_visible':[], 'future_persistence_hidden':[], 'future_persistence_visible':[],
               'future_model_on_persistence_hidden':[], 'future_model_on_persistence_visible':[]}
    history = config['history']
    device = next(model.parameters()).device
    for path in sorted(prepared.glob('*.npz')):
        with np.load(path, allow_pickle=False) as f:
            data = {k:f[k] for k in f.files}
        h, v, t = data['blocked_height_mm'], data['blocked_valid'], data['blocked_timestamp_ms']
        refh, refv = data['repeat_height_mm'], data['repeat_valid']
        hidden = data['conservative_hidden']
        match = data['reference_index']
        selected=[]
        for origin in range(history-1, len(h), stride):
            ts=t[origin-history+1:origin+1]
            if match[origin] < 0 or np.any(np.abs(np.diff(ts)-config['expected_dt_ms']) > config['expected_dt_ms']*config['timing_tolerance']+1e-6):
                continue
            selected.append(origin)
        errors={name:[] for name in overall}
        predictions=[]
        future_predictions=[]
        step_predictions=[]
        future_count=0
        for start in range(0,len(selected),batch_size):
            ids=selected[start:start+batch_size]
            batch=np.stack([encode_history(h[i-history+1:i+1],v[i-history+1:i+1],t[i-history+1:i+1],normalization,config['age_scale_ms']) for i in ids])
            with torch.inference_mode():
                x=torch.from_numpy(batch).to(device)
                predicted=normalization.denormalize(model(x, rollout_steps=max(rollout,6))[:,:,0].cpu().numpy())
            predictions.append(predicted[:,0].astype(np.float32))
            future_predictions.append(predicted[:,6].astype(np.float32))
            step_predictions.append(predicted[:,list(steps)].astype(np.float32))
            for i,pred in zip(ids,predicted):
                j=match[i]
                past_valid=v[i-history+1:i+1]
                past_h=h[i-history+1:i+1]
                ever=past_valid.any(axis=0)
                reverse_index=np.argmax(past_valid[::-1],axis=0)
                persistence=np.take_along_axis(past_h[::-1],reverse_index[None],axis=0)[0]
                for name, mask in (('hidden',hidden & refv[j]), ('visible',~hidden & refv[j] & v[i])):
                    errors[name].append((pred[0][mask]-refh[j][mask]).astype(np.float32))
                    supported=mask & ever
                    errors['persistence_'+name].append((persistence[supported]-refh[j][supported]).astype(np.float32))
                    errors['model_on_persistence_'+name].append((pred[0][supported]-refh[j][supported]).astype(np.float32))
                future=i+6
                if (future >= len(t) or match[future] < 0 or
                    np.any(np.abs(np.diff(t[i:future+1])-config['expected_dt_ms']) > config['expected_dt_ms']*config['timing_tolerance']+1e-6)):
                    continue
                future_count+=1
                j=match[future]
                for name,mask in (('hidden',hidden & refv[j]),('visible',~hidden & refv[j] & v[i])):
                    errors['future_'+name].append((pred[6][mask]-refh[j][mask]).astype(np.float32))
                    supported=mask & ever
                    errors['future_persistence_'+name].append((persistence[supported]-refh[j][supported]).astype(np.float32))
                    errors['future_model_on_persistence_'+name].append((pred[6][supported]-refh[j][supported]).astype(np.float32))
        for key in errors:
            overall[key].extend(errors[key])
        np.savez_compressed(output/path.name, origin_index=np.asarray(selected,dtype=np.int32),
                            prediction_mm=np.concatenate(predictions) if predictions else np.empty((0,23,45),np.float32),
                            prediction_200ms_mm=np.concatenate(future_predictions) if future_predictions else np.empty((0,23,45),np.float32),
                            prediction_steps=np.asarray(steps,dtype=np.int32),
                            prediction_future_mm=np.concatenate(step_predictions) if step_predictions else np.empty((0,len(steps),23,45),np.float32))
        report['patterns'][path.stem]={'scored_windows':len(selected),
            'mean_current_observed_fraction':float(v[selected].mean()) if selected else None,
            'hidden_repeat_error':stats(errors['hidden']), 'visible_repeat_error':stats(errors['visible']),
            'persistence_hidden_repeat_error':stats(errors['persistence_hidden']),
            'persistence_visible_repeat_error':stats(errors['persistence_visible']),
            'model_on_persistence_hidden_support':stats(errors['model_on_persistence_hidden']),
            'model_on_persistence_visible_support':stats(errors['model_on_persistence_visible']),
            'future_scored_windows':future_count,
            'future_hidden_repeat_error':stats(errors['future_hidden']),
            'future_visible_repeat_error':stats(errors['future_visible']),
            'future_persistence_hidden_repeat_error':stats(errors['future_persistence_hidden']),
            'future_persistence_visible_repeat_error':stats(errors['future_persistence_visible']),
            'future_model_on_persistence_hidden_support':stats(errors['future_model_on_persistence_hidden']),
            'future_model_on_persistence_visible_support':stats(errors['future_model_on_persistence_visible'])}
        print(path.stem, report['patterns'][path.stem],flush=True)
    report['overall']={name:stats(values) for name,values in overall.items()}
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--steps',type=int,nargs='+',default=[6],
                   help='forecast steps (frames) whose predictions are saved; 200 ms scoring always uses step 6')
    a=p.parse_args()
    if a.stride < 1 or a.batch_size < 1:
        p.error('stride and batch size must be positive')
    evaluate(a.prepared,a.checkpoint,a.output,a.device,a.stride,a.batch_size,a.steps)

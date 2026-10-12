"""Plots and a self-contained interactive report for the TDS benchmark."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from .tds import METHODS, write_summary


COLORS = {'convlstm': '#2368b5', 'persistence': '#343c43',
          'constant_velocity': '#198b7c', 'constant_acceleration': '#db8125'}
LABELS = {'convlstm': 'ConvLSTM', 'persistence': 'Latest measurement',
          'constant_velocity': 'Constant velocity', 'constant_acceleration': 'Constant acceleration'}


def plot_report(benchmark_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(benchmark_path).resolve()
    report = json.loads((output / 'benchmark.json').read_text(encoding='utf-8'))
    results = report['results']
    write_summary(output, Path(report['data_audit']).parent, results)
    horizons = [r['horizon_ms'] for r in results]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    for axis, region, title in zip(axes, ('all', 'moving', 'central_roi'),
                                  ('Whole surface', 'Moving cells (>5 mm change)', 'Central 200 x 200 mm region')):
        for method in METHODS:
            values = [r['paired']['regions'][region]['metrics'][method] for r in results]
            axis.plot(horizons, values, 'o-', color=COLORS[method], label=LABELS[method], linewidth=2)
        axis.set(title=title, xlabel='Forecast horizon (ms)', ylabel='Height MAE (mm)', xticks=horizons)
        axis.grid(axis='y', alpha=.2)
        axis.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(.5, .04), frameon=False)
    fig.suptitle('Cloth forecasting on three held-out recordings', fontsize=15, fontweight='bold')
    fig.text(.5, .005, 'Identical valid cells for all methods | 5 observations | fixed 20 mm XY grid | nominal 5 FPS',
             ha='center', fontsize=9, color='#506171')
    fig.tight_layout(rect=(0, .13, 1, .94))
    fig.savefig(output / 'prediction_vs_baselines.png', dpi=180)
    fig.savefig(output / 'prediction_vs_baselines.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(13, 4), squeeze=False)
    for axis, result in zip(axes[0], results):
        history = json.loads(Path(result['training']['history']).read_text(encoding='utf-8'))
        axis.plot([v['epoch'] for v in history], [v['train_mae_mm'] for v in history], label='Train', color='#2368b5')
        axis.plot([v['epoch'] for v in history], [v['validation_mae_mm'] for v in history], label='Validation', color='#db8125')
        axis.axvline(result['training']['best_epoch'], linestyle=':', color='#343c43', label='Selected checkpoint')
        axis.set(title=f"{result['horizon_ms']} ms model", xlabel='Epoch', ylabel='All-target MAE (mm)')
        axis.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Training and checkpoint selection use no test errors', fontsize=14)
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=180)
    plt.close(fig)

    examples = []
    table = []
    for result in results:
        evaluation_dir = output / f"horizon_{result['horizon_ms']}ms/evaluation"
        with (evaluation_dir / 'per_window_errors.csv').open(newline='', encoding='utf-8') as stream:
            window_rows = list(csv.DictReader(stream))
        for file in sorted(evaluation_dir.glob('example_*.npz')):
            with np.load(file, allow_pickle=False) as archive:
                common = archive['comparison_valid']
                def values(key):
                    array = archive[key]
                    return [float(value) if valid and np.isfinite(value) else None
                            for value, valid in zip(array.ravel(), common.ravel())]
                examples.append({'name': file.stem, 'horizon': result['horizon_ms'],
                    'sequence': Path(str(archive['source_session'].item())).name if 'source_session' in archive
                                else window_rows[int(file.stem.split('_')[-1])]['sequence'],
                    'width': common.shape[1], 'height': common.shape[0], 'target': values('target_mm'),
                    'methods': {m: values('prediction_mm' if m == 'convlstm' else m + '_mm') for m in METHODS},
                    'targetTime': float(archive['observed_target_timestamp_ms'])})
        for region in ('all', 'moving', 'central_roi'):
            data = result['paired']['regions'][region]
            for method in METHODS:
                compare = data['paired_vs_persistence'].get(method)
                table.append({'horizon': result['horizon_ms'], 'region': region,
                              'method': method, 'mae': data['metrics'][method],
                              'improvement': compare['mae_reduction_percent'] if compare else None,
                              'ci': compare['conditional_95pct_ci_mm'] if compare else None})
        stationary = result['evaluation']['metrics']['stationary']
        reference = stationary['persistence']['mae_mm']
        for method in METHODS:
            value = stationary[method]['mae_mm']
            table.append({'horizon': result['horizon_ms'], 'region': 'stationary',
                          'method': method, 'mae': value,
                          'improvement': 100 * (1 - value / reference) if method != 'persistence' and reference else None,
                          'ci': None})
    write_html(output, results, examples, table)


def write_html(output, results, examples, table):
    payload = json.dumps({'examples': examples, 'table': table, 'labels': LABELS}, allow_nan=False).replace('<', '\\u003c')
    options = ''.join(f'<option value="{r["horizon_ms"]}">{r["horizon_ms"]} ms</option>' for r in results)
    html_text = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cloth forecasting benchmark</title>
<style>body{font:15px system-ui,sans-serif;background:#f5f7fa;color:#24364b;max-width:1250px;margin:36px auto;padding:0 24px}
h1{font-size:30px;margin-bottom:8px}p{line-height:1.6;color:#506171}select{font:inherit;padding:8px;border:1px solid #b9c8d6;border-radius:6px;margin:0 18px 12px 6px}
.panel{background:white;border-radius:10px;padding:20px;margin:18px 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #e0e7ef;padding:10px}
.maps{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}canvas{width:100%;image-rendering:pixelated;background:#d3d9e0}img{width:100%}
.small{font-size:12px}.badge{display:inline-block;background:#e6eef6;padding:6px 10px;border-radius:4px;margin-right:8px}a{color:#2368b5}
@media(max-width:760px){.maps{grid-template-columns:1fr}body{padding:0 12px}table{font-size:12px}}</style></head><body>
<h1>Does forecasting improve future surface height?</h1>
<p>ConvLSTM and constant acceleration compared with the latest measured surface and constant velocity.
12 complete cloth sequences train the models, 3 select checkpoints, and 3 are held out for testing.
Consecutive segments of the same apparent capture stay together in validation.</p>
<span class="badge">15,799 source frames</span><span class="badge">Nominal 5 FPS</span><span class="badge">Camera-Z error in mm</span>
<div class="panel"><img src="prediction_vs_baselines.png" alt="Held-out forecast MAE by horizon and surface region"></div>
<div class="panel"><label>Horizon<select id="horizon">__OPTIONS__</select></label>
<label>Region<select id="region"><option value="all">Whole surface</option><option value="moving">Moving cells</option><option value="stationary">Stationary cells (at most 5 mm change)</option><option value="central_roi">Central region</option></select></label>
<table><thead><tr><th>Method</th><th>MAE (mm)</th><th>Reduction vs latest measurement</th><th>95% interval for reduction (mm)</th></tr></thead><tbody id="scores"></tbody></table>
<p class="small">Positive reduction means lower error. Intervals resample 5-second temporal blocks within each held-out recording.
They describe these three recordings and do not establish generalization to new trials.</p></div>
<div class="panel"><label>Forecast<select id="method"><option value="convlstm">ConvLSTM</option><option value="constant_acceleration">Constant acceleration</option><option value="constant_velocity">Constant velocity</option><option value="persistence">Latest measurement</option></select></label>
<label>Example <input id="example" type="range" min="0" max="4" value="0"></label><p id="frame"></p>
<div class="maps"><div><h3>Measured future height</h3><canvas id="target"></canvas></div><div><h3 id="predlabel">Predicted height</h3><canvas id="prediction"></canvas></div><div><h3>Forecast error</h3><canvas id="error"></canvas></div></div>
<p id="scale" class="small"></p><p class="small">Grey cells are excluded from comparison. The display uses the common offline target/history mask.
The selected region changes the scores; maps always show the whole common-support surface. Five fixed, evenly spaced windows are shown per horizon.</p></div>
<div class="panel"><h2>What the experiment supports</h2><p>Lower held-out error than the latest observation supports useful offline forecasting at that horizon.
These Kinect sequences include lateral motion and occlusion. Source depth is interpreted as metres and converted to millimetres;
timestamps are inferred from frame indices at the publisher's 5 FPS. Windows crossing missing indices are rejected.
The fixed 32 × 32 camera XY grid uses 20 mm cells and averages fine detail.</p>
<p>Printer use requires measured camera-to-printer calibration, D405 recordings, latency matched forecasts, fine spatial resolution,
toolpath gap measurements, and printing trials. This dataset cannot validate forecasts shorter than 200 ms.</p>
<a href="SUMMARY.md">Detailed experiment summary</a> · <a href="benchmark.json">Full metrics</a> ·
<a href="summary.csv">CSV comparison</a> · <a href="__DATASET__">Dataset publisher</a></div>
<div class="panel"><img src="training_curves.png" alt="Training and validation learning curves"></div>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
function draw(id,a,w,h,lo,hi,diverging){const c=$(id);c.width=w;c.height=h;const ctx=c.getContext('2d'),im=ctx.createImageData(w,h);
a.forEach((v,i)=>{let rgb=[211,217,224];if(v!==null){const t=Math.max(0,Math.min(1,(v-lo)/(hi-lo||1)));
rgb=diverging?(t<.5?[255*t*2,255*t*2,255]:[255,255*(2-2*t),255*(2-2*t)]):[30+220*t,65+155*t,145-100*t];}
im.data.set([...rgb.map(Math.round),255],i*4)});ctx.putImageData(im,0,0);}
function render(){const horizon=+$('horizon').value,region=$('region').value,method=$('method').value;
$('scores').innerHTML=data.table.filter(v=>v.horizon===horizon&&v.region===region).map(v=>`<tr><td>${data.labels[v.method]}</td><td>${v.mae.toFixed(3)}</td><td>${v.improvement===null?'Reference':v.improvement.toFixed(1)+'%'}</td><td>${v.ci?v.ci.map(x=>x.toFixed(3)).join(' to '):'—'}</td></tr>`).join('');
const es=data.examples.filter(v=>v.horizon===horizon);$('example').max=es.length-1;
const e=es[Math.min(+$('example').value,es.length-1)],p=e.methods[method],err=p.map((v,i)=>v===null||e.target[i]===null?null:v-e.target[i]);
let lo=Infinity,hi=-Infinity,cap=.001;for(const v of e.target.concat(...Object.values(e.methods))){if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v)}}
for(const m of Object.values(e.methods)){m.forEach((v,i)=>{if(v!==null&&e.target[i]!==null)cap=Math.max(cap,Math.abs(v-e.target[i]))})}
draw('target',e.target,e.width,e.height,lo,hi,false);draw('prediction',p,e.width,e.height,lo,hi,false);draw('error',err,e.width,e.height,-cap,cap,true);
$('predlabel').textContent=data.labels[method];$('frame').textContent=`${e.sequence} · ${e.name} · target ${(e.targetTime/1000).toFixed(1)} s · ${horizon} ms forecast`;
$('scale').textContent=`Shared scales across methods: height ${lo.toFixed(1)}–${hi.toFixed(1)} mm; blue / white / red error ${(-cap).toFixed(1)} / 0 / ${cap.toFixed(1)} mm.`;}
['horizon','region','method','example'].forEach(id=>$(id).addEventListener('input',render));render();</script></body></html>'''
    from .tds import DATASET_URL
    (output / 'report.html').write_text(html_text.replace('__OPTIONS__', options).replace('__DATA__', payload)
                                      .replace('__DATASET__', html.escape(DATASET_URL)), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True, type=Path)
    plot_report(parser.parse_args().benchmark)

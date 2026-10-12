"""A standalone HTML viewer for held-out surface forecast examples."""

import html
import json
from pathlib import Path

import numpy as np


def write_visual_report(output: Path, report: dict) -> None:
    examples = []
    for path in sorted(output.glob('example_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            valid = data['target_valid']

            def values(key):
                array = np.asarray(data[key], dtype=float)
                return [float(v) if ok and np.isfinite(v) else None
                        for v, ok in zip(array.ravel(), valid.ravel())]

            examples.append({'name': path.stem, 'height': valid.shape[0], 'width': valid.shape[1],
                             'target': values('target_mm'), 'prediction': values('prediction_mm'),
                             'error': values('error_mm'),
                             'timestamp': float(data['observed_target_timestamp_ms'])})
    rows = []
    for name, metrics in report['metrics']['all'].items():
        cells = [html.escape(name)] + [f'{metrics[key]:.4f}' if metrics[key] is not None else 'no support'
                                      for key in ('mae_mm', 'rmse_mm', 'p95_absolute_error_mm')]
        rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    document = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Surface forecast evaluation</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;background:#f4f6f8;color:#172b3a}
h1{font-size:28px}p{line-height:1.5;color:#43596a}table{border-collapse:collapse;background:white;width:100%;max-width:760px}
td,th{text-align:left;padding:12px;border-bottom:1px solid #d7e0e7}th{background:#e6eef3}
.maps{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:24px 0}.panel{background:white;padding:16px;border-radius:8px}
canvas{width:100%;image-rendering:pixelated;background:#d0d4d8}.label{font-size:14px;color:#43596a}input{width:60%}
@media(max-width:720px){.maps{grid-template-columns:1fr}}
</style><h1>Surface forecast evaluation</h1>
<p>Errors are measured in millimetres against recorded sensor measurements. This report does not establish physical ground truth accuracy.</p>
<table><thead><tr><th>Predictor</th><th>MAE (mm)</th><th>RMSE (mm)</th><th>95th percentile (mm)</th></tr></thead>
<tbody>__ROWS__</tbody></table>
<p>All methods use the same comparison support: __SUPPORT__.
Large error sets use a bounded uniform sample for the 95th percentile; exact MAE and RMSE use all scored cells. See metrics.json for coverage and per-session results.</p>
<label>Example <input id="slider" type="range" min="0" value="0" step="1"></label> <span id="frame"></span>
<div class="maps"><div class="panel"><h3>Measured future surface</h3><canvas id="target"></canvas></div>
<div class="panel"><h3>Predicted future surface</h3><canvas id="prediction"></canvas></div>
<div class="panel"><h3>Prediction minus measurement</h3><canvas id="error"></canvas></div></div>
<p id="scale" class="label"></p><p class="label">Grey cells have missing targets, unsupported predictions, or fall outside the region. This display mask uses offline target measurements; it is not available during live inference.</p>
<script type="application/json" id="data">__DATA__</script>
<script>
const examples=JSON.parse(document.getElementById('data').textContent);
const slider=document.getElementById('slider');slider.max=Math.max(0,examples.length-1);
function draw(id,values,w,h,lo,hi,diverging){
const canvas=document.getElementById(id);canvas.width=w;canvas.height=h;
const ctx=canvas.getContext('2d'),image=ctx.createImageData(w,h);
values.forEach((v,i)=>{let color=[208,212,216];if(v!==null){const t=Math.max(0,Math.min(1,(v-lo)/(hi-lo||1)));
if(diverging){color=t<.5?[Math.round(255*t*2),Math.round(255*t*2),255]:[255,Math.round(255*(2-2*t)),Math.round(255*(2-2*t))];}
else{color=[Math.round(25+225*t),Math.round(65+160*t),Math.round(150-110*t)];}}
image.data.set([...color,255],i*4);});ctx.putImageData(image,0,0);}
function render(){if(!examples.length)return;const e=examples[Number(slider.value)];
let lo=Infinity,hi=-Infinity,abs=0.001;
for(const v of e.target.concat(e.prediction)){if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v);}}
for(const v of e.error){if(v!==null)abs=Math.max(abs,Math.abs(v));}
if(!Number.isFinite(lo)){lo=0;hi=1;}
draw('target',e.target,e.width,e.height,lo,hi,false);draw('prediction',e.prediction,e.width,e.height,lo,hi,false);draw('error',e.error,e.width,e.height,-abs,abs,true);
document.getElementById('frame').textContent=e.name+' / target '+e.timestamp.toFixed(2)+' ms';
document.getElementById('scale').textContent='Shared depth scale: dark blue '+lo.toFixed(3)+' mm to yellow '+hi.toFixed(3)+' mm. Error scale: blue '+(-abs).toFixed(3)+' mm / white 0 / red '+abs.toFixed(3)+' mm.';}
slider.addEventListener('input',render);render();
</script></html>'''
    payload = json.dumps(examples, allow_nan=False).replace('<', '\\u003c')
    (output / 'report.html').write_text(document.replace('__ROWS__', ''.join(rows)).replace('__DATA__', payload).replace('__SUPPORT__', html.escape(report.get('comparison_support', 'see metrics.json'))), encoding='utf-8')

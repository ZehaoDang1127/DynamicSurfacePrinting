"""Plot a provisional real-head prediction against a separate repeat run."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(prepared, predictions, output):
    with np.load(prepared, allow_pickle=False) as f:
        data={k:f[k] for k in f.files}
    with np.load(predictions, allow_pickle=False) as f:
        indices=f['origin_index']; predicted=f['prediction_mm']
    if not len(indices):
        raise ValueError('no predictions to plot')
    k=len(indices)//2; i=int(indices[k]); j=int(data['reference_index'][i])
    target=np.where(data['repeat_valid'][j],data['repeat_height_mm'][j],np.nan)
    observed=np.where(data['blocked_valid'][i],data['blocked_height_mm'][i],np.nan)
    estimate=np.where(data['repeat_valid'][j],predicted[k],np.nan)
    hidden=data['conservative_hidden'] & data['repeat_valid'][j]
    error=np.where(hidden,predicted[k]-target,np.nan)
    if not np.any(np.isfinite(target)):
        raise ValueError('reference has no supported cells')
    lower,upper=np.nanpercentile(target,[1,99]); limit=max(1,float(np.nanpercentile(np.abs(error),95))) if np.any(np.isfinite(error)) else 1
    fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
    for ax,field,title in zip(axes[:3],(target,observed,estimate),('Separate repeat reference','Blocked input: measured cells','Model: current estimate')):
        im=ax.imshow(field,origin='lower',extent=(-80,100,-28,64),cmap='viridis',vmin=lower,vmax=upper,aspect='equal')
        ax.set_title(title); ax.set_xlabel('Camera X (mm)'); ax.set_ylabel('Camera Y (mm)')
    fig.colorbar(im,ax=axes[:3],label='Camera depth (mm)',shrink=.72)
    diff=axes[3].imshow(error,origin='lower',extent=(-80,100,-28,64),cmap='coolwarm',vmin=-limit,vmax=limit,aspect='equal')
    axes[3].set_title('Model − repeat\n(preliminary hidden region)'); axes[3].set_xlabel('Camera X (mm)'); axes[3].set_ylabel('Camera Y (mm)')
    fig.colorbar(diff,ax=axes[3],label='Depth difference (mm)',shrink=.72)
    fig.suptitle(f'{prepared.stem}: separate repeats, no validated registration; frame {i} vs {j}\nColor-camera head mask and timing alignment are provisional; differences are not ground-truth errors.',fontsize=11)
    fig.savefig(output,dpi=160); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--prepared',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    plot(a.prepared,a.predictions,a.output)

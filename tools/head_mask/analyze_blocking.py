"""Estimate print-head coverage of a manually defined surface in this recording.

Coordinates and thresholds were selected for recordings/20261102T184423_929388Z
only; inspect overlays and re-label geometry before using another recording.
Requires numpy and OpenCV. Native depth is intentionally not cropped/aligned.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np

# Full-resolution COLOR image coordinates. Half-open crop, x then y.
CROP = (95, 230, 339, 376)
BACK_LEFT, BACK_RIGHT = (110, 235), (302, 235)
HEAD = np.array([(183,220),(266,220),(265,259),(260,270),(254,273),
                 (239,273),(235,267),(215,267),(211,273),(196,273),(188,267),(185,256)], np.int32)
XS = np.arange(95, 339)
YS = np.arange(305, 376)


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (480, 640):
        raise ValueError(f'Expected a 640x480 color PNG: {path}')
    return image


def write_image(path, image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError(f'Could not encode {path}')
    data.tofile(path)


def front_edge(image):
    """Smooth path through a constrained bright-to-dark vertical gradient."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5,5), 0).astype(float)
    gradient = gray[YS[:,None]-2, XS] - gray[YS[:,None]+2, XS]
    prior = 352 - (XS - 95) * .06
    score = gradient - .12 * np.abs(YS[:,None] - prior[None,:])
    dp = score[:,0].copy()
    penalty = 1.7 * np.abs(YS[:,None] - YS[None,:])
    back = []
    for j in range(1, len(XS)):
        transitions = dp[:,None] - penalty
        best = transitions.argmax(axis=0)
        dp = transitions[best, np.arange(len(YS))] + score[:,j]
        back.append(best)
    path = [int(dp.argmax())]
    for best in reversed(back):
        path.append(int(best[path[-1]]))
    return YS[np.asarray(path[::-1])], gradient


def surface_mask(edge, y_offset=0):
    poly = np.array([(BACK_LEFT[0],BACK_LEFT[1]+y_offset),
                     (BACK_RIGHT[0],BACK_RIGHT[1]+y_offset)] +
                    list(zip(XS[::-1], edge[::-1])), np.int32)
    mask = np.zeros((480,640), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('output', type=Path, help='new output directory')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Output must not already exist: {args.output}')
    with (args.session/'frames.csv').open(newline='', encoding='utf-8-sig') as f:
        frames=list(csv.DictReader(f))
    if not frames:
        raise ValueError('No frames in frames.csv')
    args.output.mkdir(parents=True)
    for name in ('color_crop', 'surface_mask', 'head_mask', 'overlays'):
        (args.output/name).mkdir()
    head=np.zeros((480,640),np.uint8)
    cv2.fillPoly(head,[HEAD],255)
    x0,y0,x1,y1=CROP
    rows=[]
    for frame in frames:
        index=int(frame['index'])
        image=read_image(args.session/frame['color_file'])
        edge, gradient=front_edge(image)
        roi=surface_mask(edge)
        blocked=cv2.bitwise_and(head,roi)
        count=int(cv2.countNonZero(roi)); nblock=int(cv2.countNonZero(blocked))
        sensitivity=[]
        for offset in (-5,5):
            alt=surface_mask(edge,offset)
            sensitivity.append(cv2.countNonZero(cv2.bitwise_and(head,alt))/cv2.countNonZero(alt))
        cropped=image[y0:y1,x0:x1].copy()
        cropped[roi[y0:y1,x0:x1] == 0]=0
        overlay=image.copy()
        overlay[blocked>0] = (0.5*overlay[blocked>0] + np.array([0,0,127.5])).astype(np.uint8)
        contours,_=cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(0,255,0),1)
        contours,_=cv2.findContours(blocked,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(0,0,255),1)
        name=f'{index:06d}.png'
        for directory, data in [('color_crop',cropped),('surface_mask',roi[y0:y1,x0:x1]),
                                ('head_mask',blocked[y0:y1,x0:x1]),('overlays',overlay)]:
            write_image(args.output/directory/name,data)
        rows.append({'index':index, 'elapsed_s':frame['elapsed_s'],
                     'color_timestamp_ms':frame['color_timestamp_ms'],
                     'commanded_angle_at_receive_deg':frame['commanded_angle_at_receive_deg'],
                     'surface_pixels':count, 'blocked_pixels':nblock,
                     'blocked_fraction':nblock/count,
                     'fraction_back_minus_5px':sensitivity[0],
                     'fraction_back_plus_5px':sensitivity[1],
                     'front_edge_min_y':int(edge.min()),'front_edge_max_y':int(edge.max()),
                     'front_edge_search_limit_pixels':int(np.count_nonzero((edge==YS[0]) | (edge==YS[-1]))),
                     'median_front_gradient':float(np.median(gradient[edge-YS[0],np.arange(len(XS))]))})
    with (args.output/'frames.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    frac=np.array([r['blocked_fraction'] for r in rows]);lo=np.array([r['fraction_back_minus_5px'] for r in rows]);hi=np.array([r['fraction_back_plus_5px'] for r in rows]);
    summary={'source_session':str(args.session.resolve()), 'frame_count':len(rows),
        'coordinate_system':'raw full-resolution color pixels, x right, y down',
        'crop_xyxy_exclusive':list(CROP), 'fixed_back_boundary':[list(BACK_LEFT),list(BACK_RIGHT)],
        'fixed_printhead_polygon':HEAD.tolist(),
        'front_edge_method':'5x5 blurred luminance, constrained y=305..375; maximize vertical bright-to-dark gradient with smoothness and weak position prior',
        'blocked_fraction_definition':'count(surface mask AND printhead silhouette) / count(surface mask); projected image pixels',
        'fraction':{'mean':float(frac.mean()),'median':float(np.median(frac)),'min':float(frac.min()),'max':float(frac.max()), 'std':float(frac.std())},
        'back_boundary_shift_minus_5px_mean_fraction':float(lo.mean()),
        'back_boundary_shift_plus_5px_mean_fraction':float(hi.mean()),
        'total_front_edge_search_limit_pixels':sum(r['front_edge_search_limit_pixels'] for r in rows),
        'assumptions_and_limits':['ROI back and side boundaries are manually approximated; hidden surface geometry is not observed.',
          'Fixed polygon approximates stationary printhead assembly silhouette, including housing, not the nozzle tip alone.',
          'Front edge is color-based, constrained and may fail under reflections, occlusion, or changed camera pose; inspect overlays.',
          'Metric is projected RGB image fraction, not physical surface area or depth/grid coverage.',
          'The unaligned depth images are untouched; map through calibrated RGB-D geometry before depth or height-map usage.',
          'The +/-5 pixel back-boundary sensitivity is a local assumption check, not a statistical confidence interval.',
          'This recording lacks ground truth for the hidden surface.']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    # Contact sheet samples the start, end and ten evenly spaced frames.
    tiles=[]
    for i in np.linspace(0, len(rows)-1, min(12,len(rows)), dtype=int):
        row=rows[i]
        a=read_image(args.output/'overlays'/f"{row['index']:06d}.png")[220:385,80:355]
        a=cv2.resize(a,(550,330),interpolation=cv2.INTER_NEAREST)
        cv2.putText(a,f"frame {row['index']} {100*row['blocked_fraction']:.2f}%",(8,22),
                    cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2,cv2.LINE_AA)
        tiles.append(a)
    while len(tiles)%3:
        tiles.append(np.zeros_like(tiles[0]))
    ok, sheet=cv2.imencode('.jpg',np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
    if not ok:
        raise ValueError('Could not encode contact sheet')
    sheet.tofile(args.output/'qa_contact_sheet.jpg')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()

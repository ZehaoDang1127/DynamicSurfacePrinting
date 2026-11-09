# 100-epoch, cloth-pretrained and mixed-mask models: results (7–8 Nov 2026)

Runs follow [TRAINING_100EP_AND_MIXED_MASKS.md](TRAINING_100EP_AND_MIXED_MASKS.md)
(round 1: Steps 1–9; round 2: "Training with head-scale masks"). The 30-epoch paper
model is `output/servo_30fps_convlstm_20261105/best.pt` (epoch 26 of 30,
SHA-256 `71d5361a…`). Result folders stay local under `output/` and are not in Git.

**Lab computer:** NVIDIA GeForce RTX 5090 (32 GB, driver 621.14), Windows, Python 3.12,
PyTorch 2.11.0+cu128. About 18–22 s per servo epoch (the guide assumed ~1.5 min).

## Training runs

| Run folder | Config | Start | Epochs | Best epoch | Best validation objective | Time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `servo_30fps_convlstm_100ep_20261107` | `occlusion_d405_30fps_100ep.json` | random | 100 | 85 | 8.647 mm | 0.51 h |
| `cloth_pretrain_20261107` | `occlusion_cloth_pretrain_5fps.json` | random | 37 (early stop) | 29 | 51.226 mm | 0.27 h |
| `servo_30fps_finetune_20261107` | `occlusion_d405_30fps_finetune.json` | cloth `best.pt` | 100 | 49 | 9.353 mm | 0.51 h |
| `servo_30fps_mixed_100ep_20261108` | `occlusion_d405_30fps_mixed_100ep.json` | random | 100 | 98 | 9.996 mm | 0.60 h |
| `servo_30fps_mixed_finetune_20261108` | `occlusion_d405_30fps_mixed_finetune.json` | cloth `best.pt` | 100 | 72 | 10.752 mm | 0.55 h |

Mixed-mask runs validate with mixed masks, so their validation objectives are not
comparable with the Gaussian-mask runs. Both fine-tuned runs record
`pretrained_weights_only` from `cloth_pretrain_20261107/best.pt`, source shape [32, 32].
GPU arithmetic is not repeatable: two identical one-epoch runs scored 25.5 and 42.7 mm
after the first epoch. Each configuration was trained once.

## Synthetic occlusion, paper masks (Table 1)

Held-out test patterns 18 and 20; 1740 windows; mask seed 44 for every model. Hidden-cell
MAE in camera depth (mm). "Same support" scores the model only where persistence has an
observation, matching the README's published numbers (2.59 / 3.47 mm for the 30-epoch
model; the current code gives 2.60 / 3.49 mm).

| Model | Current, same support | 200 ms, same support | Current, all hidden cells | 200 ms, all hidden cells |
| --- | ---: | ---: | ---: | ---: |
| 30 epochs (paper model) | 2.60 | 3.49 | 2.68 | 2.98 |
| 100 epochs | 2.47 | 3.17 | 2.56 | 2.76 |
| 100 epochs, cloth-pretrained | 2.48 | 3.51 | 2.74 | 3.12 |
| Mixed masks, 100 epochs | 2.48 | 3.20 | 2.56 | 2.79 |
| Mixed masks, cloth-pretrained | 2.49 | 3.50 | 2.73 | 3.10 |
| Persistence (last observation) | 3.58 | 4.40 | – | – |

## Synthetic occlusion, large stationary masks

Same 1740 windows and mask seed; `configs/masks/large_stationary.json` hides 70–90% of the
grid with a stationary blob. True labels exist; hidden cells are never observed in the
25-frame history, so persistence has no support. Hidden-cell MAE in camera depth (mm).

| Model | Current | 200 ms |
| --- | ---: | ---: |
| 30 epochs (paper model) | 20.61 | 20.41 |
| 100 epochs | 36.31 | 36.57 |
| 100 epochs, cloth-pretrained | 28.08 | 19.68 |
| Mixed masks, 100 epochs | 3.28 | 3.29 |
| Mixed masks, cloth-pretrained | 3.61 | 3.65 |

## Real print head (Table 2)

Patterns 21–25 with the head in view, compared with the separate repeat recording made
about 45 minutes earlier (not simultaneous ground truth). Hidden region, current frame /
200 ms. Camera depth is measured along the D405's line of sight. Printer height is the
predicted point's printer Z minus the reference surface's Z at the same printer XY, from
`python -m tools.rescore_real_head_printer` with the 8 Nov registration (below). Bias is
prediction minus reference; in printer Z, negative means too low.

| Model | Camera depth | Camera bias | Printer height | Printer bias |
| --- | ---: | ---: | ---: | ---: |
| 30 epochs (paper model, 6 Nov) | 5.79 / 5.21 | +0.45 | 2.64 / 2.35 | −0.22 |
| 100 epochs | 33.80 / 36.07 | −33.73 | 15.85 / 16.84 | +15.81 |
| 100 epochs, cloth-pretrained | 14.50 / 11.21 | +11.83 | 6.59 / 5.22 | −5.40 |
| Mixed masks, 100 epochs | 8.82 / 7.92 | +7.71 | 4.02 / 3.61 | −3.54 |
| Mixed masks, cloth-pretrained | 6.61 / 5.07 | +5.70 | 2.97 / 2.27 | −2.61 |
| The two recordings, cells visible in both | 5.45 / 5.44 | +4.95 | 2.92 / 2.91 | −2.68 |

The last row is last-observed (persistence) values against the repeat on visible cells:
how much the two recordings disagree. In printer height the mixed-mask, cloth-pretrained
model's hidden-region error and bias match that disagreement, so this reference cannot
separate it from a perfect prediction. The 30-epoch model was re-scored on the laptop
from its 6 Nov predictions (8 Nov, same tool and registration); it is also at that floor
(2.64 / 2.35 mm), so the real-head test cannot rank these two models, whereas the
large-mask test separates them (20.61 vs 3.61 mm).

## Printer-frame registration used

`output/registration_20261108/transform.json` (local), D405 serial 260522272645: four
printed dots, SVD rigid fit of SDK metric depth, 0.288 mm RMS and 0.357 mm maximum
residual, 0.702 mm maximum leave-one-out error. Status: provisional; validation on an
independent fifth point is pending. The camera's optical axis is about 61° from vertical,
so a camera-depth error corresponds to roughly 0.48 times that error vertically on a
level surface.

The registration was made on 8 Nov; the recordings are from 5–6 Nov. Under it, the
training patterns (18, 20) and the real-head repeats map to a surface at printer
Z ≈ 149–154 mm, within about 2° of level, at the height of the registration dots
(149.7–150.4 mm). This supports, but does not prove, an unchanged camera pose.

## Reading the results

- The 30-epoch and 100-epoch Gaussian-mask models fail when most of the grid is hidden
  (about 20–37 mm on large masks). Training with mixed masks removes that failure
  (3.3–3.6 mm) with no loss on the paper's masks.
- On the real head, the mixed-mask, cloth-pretrained model is the only new model close
  to the 30-epoch model in camera depth (6.61 / 5.07 vs 5.79 / 5.21 mm); both are at
  the repeat-reference floor in printer height (2.97 and 2.64 mm against 2.92 mm).
- The paper designates the mixed-mask model trained from scratch as the primary model,
  because it has the lower validation objective of the two mixed-mask models
  (10.00 vs 10.75 mm).
- Single runs and an unregistered repeat reference limit these comparisons. Retraining
  the mixed-mask models with additional seeds would show whether the gap between them
  holds.

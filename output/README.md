# Main-line results

Each folder is one stage of the occlusion-aware ConvLSTM + U-Net pipeline (see the
root `README.md` for the commands). Folder names and locations are kept exactly as they
were created. The checkpoint, `run.json` and `metrics.json` record absolute paths to
`servo_30fps_camera_maps_20261105/splits.json`, and
`tools/evaluate_occlusion_heldout.py` checks that path, so **do not rename or move these
folders**.

The 7–8 Nov runs were made on the lab desktop (RTX 5090), and their reports record
paths under `C:\Users\haim_\Desktop\DynamicSurfacePrinting\output\`. They are
kept here under the same folder names, so that layout matches this folder. Use the
lab numbers for every model: evaluation masks are seeded from the session path, so a
laptop re-run gives slightly different values.

| Stage | Folder | Contents |
| --- | --- | --- |
| 2 · Height maps | `servo_30fps_camera_maps_20261105/` | 20 patterns × ~900 frames of 23×45 camera-depth maps (`frame_*.npz`, `manifest.json`) and `splits.json` (train 16 / validation 15, 19 / test 18, 20) |
| 3 · Model | `servo_30fps_convlstm_20261105/` | `best.pt` (epoch 26), `history.json`, `run.json`, `RESULTS.md` (summary of all evaluations) |
| 4 · Synthetic-mask test | `servo_30fps_internal_synthetic_eval_20261106/` | `metrics.json`: 1,740 windows from held-out patterns 18 and 20, Gaussian masks (seed 44) |
| 5a · Real-head preparation | `real_occlusion_prior_repeats_20261105/` | Per-pattern NPZ (blocked maps, conservative head mask, repeat-reference pairing) and `report.json` |
| 5b · Real-head evaluation | `real_occlusion_repeat_evaluation_20261106/` | Per-pattern predictions, `metrics.json`, `TEST_RESULTS.md`, `example.png` |
| 5c · Repeat registration (exploratory) | `servo30_registration/` | `registration.json`: SIFT/RANSAC rigid fit between a Test and Test_groundTruth pair; did not pass spatial hold-out checks |

**Round 1 (7 Nov): 100 epochs and cloth pretraining, standard Gaussian masks**

| Stage | Folder | Contents |
| --- | --- | --- |
| Cloth pretraining | `cloth_pretrain_20261107/` | EPFL cloth, 5 FPS, history 5 / horizon 1; `best.pt` (epoch 29) |
| Model G100 | `servo_30fps_convlstm_100ep_20261107/` | 100 epochs from scratch, best epoch 85 |
| Model G100-P | `servo_30fps_finetune_20261107/` | Fine-tuned from the cloth weights, lr 3e-4, best epoch 49 |
| Synthetic test | `servo_30fps_100ep_synthetic_eval_20261107/`, `servo_30fps_finetune_synthetic_eval_20261107/` | Paper masks, held-out 18 and 20 |
| Real-head test | `real_occlusion_100ep_eval_20261107/`, `real_occlusion_finetune_eval_20261107/` | Patterns 21–25, camera depth |
| Logs | `run_logs_round1_20261107/` | Training/eval logs, `summary.json`, the lab run script `run_100ep_training.ps1` |

**Round 2 (8 Nov): mixed (standard + head-scale) masks**

| Stage | Folder | Contents |
| --- | --- | --- |
| Model M100 (**primary**) | `servo_30fps_mixed_100ep_20261108/` | 100 epochs from scratch, mixed masks, best epoch 98 |
| Model M100-P | `servo_30fps_mixed_finetune_20261108/` | Fine-tuned from the cloth weights, mixed masks, best epoch 72 |
| Evaluations | `eval_round2_20261108/` | `<model>_paper_masks`, `<model>_large_masks` (70–90 %, stationary), `mixed_*_real_head` |
| Camera–printer registration | `registration_20261108/` | Four-dot SVD rigid fit, 0.288 mm RMS (provisional); `transform.json`, `REPORT.md` |
| Printer-height rescore | `printer_frame_rescore_20261108/` | `printer_frame_real_head_rescore.json` (lab: the four 100-epoch models + repeat reference); `printer_frame_real_head_rescore_30ep.json` (laptop: the 30-epoch model) |
| Logs | `run_logs_round2_20261108/` | Logs, `summary.json`, `run_round2_training.ps1` |

The comparison of all five models is in
`../docs/occlusion_forecasting/RESULTS_100EP_MIXED_MASKS_20261108.md`.

The raw inputs for every stage are in `../Dataset/`. Earlier and side experiments are in
`../archive/`.

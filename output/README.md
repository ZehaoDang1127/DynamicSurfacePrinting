# Main-line results

Each folder is one stage of the occlusion-aware ConvLSTM + U-Net pipeline (see the
root `README.md` for the commands). Folder names and locations are kept exactly as they
were created. The checkpoint, `run.json` and `metrics.json` record absolute paths to
`servo_30fps_camera_maps_20261105/splits.json`, and
`tools/evaluate_occlusion_heldout.py` checks that path, so **do not rename or move these
folders**.

| Stage | Folder | Contents |
| --- | --- | --- |
| 2 · Height maps | `servo_30fps_camera_maps_20261105/` | 20 patterns × ~900 frames of 23×45 camera-depth maps (`frame_*.npz`, `manifest.json`) and `splits.json` (train 16 / validation 15, 19 / test 18, 20) |
| 3 · Model | `servo_30fps_convlstm_20261105/` | `best.pt` (epoch 26), `history.json`, `run.json`, `RESULTS.md` (summary of all evaluations) |
| 4 · Synthetic-mask test | `servo_30fps_internal_synthetic_eval_20261106/` | `metrics.json`: 1,740 windows from held-out patterns 18 and 20, Gaussian masks (seed 44) |
| 5a · Real-head preparation | `real_occlusion_prior_repeats_20261105/` | Per-pattern NPZ (blocked maps, conservative head mask, repeat-reference pairing) and `report.json` |
| 5b · Real-head evaluation | `real_occlusion_repeat_evaluation_20261106/` | Per-pattern predictions, `metrics.json`, `TEST_RESULTS.md`, `example.png` |
| 5c · Repeat registration (exploratory) | `servo30_registration/` | `registration.json`: SIFT/RANSAC rigid fit between a Test and Test_groundTruth pair; did not pass spatial hold-out checks |

The raw inputs for every stage are in `../Dataset/`. Earlier and side experiments are in
`../archive/`.

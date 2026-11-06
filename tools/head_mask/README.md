# Print-head mask tools (2 Nov 2026 pilot)

These are image-space tools for describing print-head occlusion in RGB. They were
developed on the first recording with the head in view, now in
`archive/head_occlusion_pilot_2026-11-02/` (that folder's `README.md` has the original
instructions and analysis notes).

| Script | Purpose |
| --- | --- |
| `analyze_blocking.py` | Estimate head coverage of a manually defined surface ROI from RGB (crop, surface mask, head polygon, overlays). Its coordinates were hand-selected for the pilot recording only. |
| `synthesize_blocking.py` | Paste the measured head silhouette onto a **clean** recording to create paired clean/blocked crops and masks. |

Both need `numpy` and `opencv-python`. Re-label the crop and head geometry before using
either script on a different camera pose. The current main-line real-head mask is
implemented separately in `tools/prepare_real_occlusion_repeats.py` (padded RGB box
projected through a training-only depth envelope).

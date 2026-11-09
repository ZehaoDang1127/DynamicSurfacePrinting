# 100-Epoch ConvLSTM + U-Net Training Guide

Nov 7, 2026 · @Zehao

> **Code updates now in the repository.** The fine-tuning code (Step 1) and the round-2
> mixed-mask code (`occlusion_code_update_v2.zip`) are committed, so the `Expand-Archive` steps
> are no longer needed with an up-to-date clone. Results of both rounds:
> [RESULTS_100EP_MIXED_MASKS_20261108.md](RESULTS_100EP_MIXED_MASKS_20261108.md).

Train two versions of the occlusion-aware ConvLSTM + U-Net on the lab computer and compare them on the paper's tests: one trained from scratch for 100 epochs, and one pretrained on the EPFL cloth data and then fine-tuned on the 20 servo patterns. Each 100-epoch servo run should take about 2.5 hours on a GPU similar to the one that ran the 30-epoch model in 45 minutes; the cloth pretraining is shorter. Apart from the epoch limit, early stopping, the fine-tuning learning rate and the starting weights, every setting matches the model in the paper.

## Before you start

You need a Windows lab computer with an NVIDIA GPU and about 1 GB of free disk space. The training itself only needs the prepared height maps, not the raw videos in `Dataset/`.

- [ ] NVIDIA GPU with a recent driver (`nvidia-smi` in PowerShell should print the GPU and a CUDA version of 12.8 or higher)
- [ ] Python 3.11 or 3.12 (`py -0` lists installed versions)
- [ ] Git, and access to the GitHub repository `ZehaoDang1127/DynamicSurfacePrinting`
- [ ] A USB drive or network share to move about 250 MB of data from the laptop
- [ ] The computer can stay awake for about 6–7 hours in total (the runs can be split across days)

At the end you will have these new folders:

| Folder | Contents |
| --- | --- |
| `output\servo_30fps_convlstm_100ep_<date>` | From-scratch model: `best.pt`, `history.json` (per-epoch errors), `run.json` (settings and best epoch) |
| `output\cloth_pretrain_<date>` | Model pretrained on the EPFL cloth data |
| `output\servo_30fps_finetune_<date>` | Cloth-pretrained model fine-tuned on the servo data |
| `output\servo_30fps_100ep_synthetic_eval_<date>` and `output\servo_30fps_finetune_synthetic_eval_<date>` | Held-out synthetic-occlusion test of each servo model (Table 1, Fig. 1) |
| `output\real_occlusion_100ep_eval_<date>` and `output\real_occlusion_finetune_eval_<date>` | Real print-head test of each servo model (Table 2) |

All commands below run in **PowerShell from the repository root** (the folder that contains `README.md`).

## Step 1: Get the code

Clone the repository. The reorganized code is already merged into `main`.

```powershell
cd D:\
git clone https://github.com/ZehaoDang1127/DynamicSurfacePrinting.git
cd DynamicSurfacePrinting
git log --oneline -1
```

The last line should show the merge of pull request #1. If the repository is private, Git asks you to sign in to GitHub the first time.

You can clone to any drive. The examples below assume `D:\DynamicSurfacePrinting`; keep the clone in the same place for the whole procedure, because the trained model records the absolute path of its training data.

**Apply the fine-tuning code update.** The pretraining and fine-tuning code is not on GitHub yet. It comes as `occlusion_finetune_code_update.zip` in the laptop's project folder (`D:\HAIM_Lab\Dynamic_Surface_Printer`): copy it to the lab computer with the data in Step 2, then unpack it over the clone:

```powershell
cd D:\DynamicSurfacePrinting
Expand-Archive -Path <path to>\occlusion_finetune_code_update.zip -DestinationPath . -Force
git status --short
```

`git status` should list `prediction/occlusion.py` and `tests/test_occlusion_pipeline.py` as modified and three new files in `configs/`. Once the code is pushed to GitHub, this step is no longer needed.

## Step 2: Copy the prepared data from the laptop

Three folders are needed. Both are local data, not part of the Git repository.

| Folder on the laptop | Size | Used for |
| --- | --- | --- |
| `output\servo_30fps_camera_maps_20261105` | 146 MB, about 18,000 files | Servo training, validation and synthetic test (20 patterns + `splits.json`) |
| `output\real_occlusion_prior_repeats_20261105` | 14 MB, 6 files | Real print-head test (5 prepared patterns + `report.json`) |
| `archive\cloth_tds_benchmark\tds_cloth\prepared` | 92 MB, about 15,800 files | Cloth pretraining (18 EPFL cloth sequences as height maps + `splits.json`) |

**On the laptop**, zip both folders. Copying one ZIP is much faster than 18,000 small files.

```powershell
cd D:\HAIM_Lab\Dynamic_Surface_Printer
Compress-Archive -Path output\servo_30fps_camera_maps_20261105, output\real_occlusion_prior_repeats_20261105 -DestinationPath $env:USERPROFILE\Desktop\occlusion_training_data.zip
Compress-Archive -Path archive\cloth_tds_benchmark\tds_cloth\prepared\* -DestinationPath $env:USERPROFILE\Desktop\cloth_tds_prepared.zip
```

Copy `occlusion_training_data.zip` and `cloth_tds_prepared.zip` from the laptop's Desktop to the lab computer, together with `occlusion_finetune_code_update.zip` from the project folder (applied in Step 1).

**On the lab computer**, unzip it into the repository's `output` folder:

```powershell
cd D:\DynamicSurfacePrinting
Expand-Archive -Path <path to>\occlusion_training_data.zip -DestinationPath output
Expand-Archive -Path <path to>\cloth_tds_prepared.zip -DestinationPath output\cloth_tds_prepared
```

Check that the data arrived complete:

```powershell
(Get-ChildItem output\servo_30fps_camera_maps_20261105 -Directory).Count              # expect 20
Test-Path output\servo_30fps_camera_maps_20261105\splits.json                       # expect True
(Get-ChildItem output\real_occlusion_prior_repeats_20261105 -Filter *.npz).Count    # expect 5
(Get-ChildItem output\cloth_tds_prepared\sessions -Directory).Count                  # expect 18
Test-Path output\cloth_tds_prepared\splits.json                                     # expect True
```

Do not rename or move these folders after training starts.

## Step 3: Set up the Python environment

Create a fresh virtual environment, install the CUDA build of PyTorch first, then the project requirements.

```powershell
cd D:\DynamicSurfacePrinting
py -3.12 -m venv .venv-train
$py = ".\.venv-train\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements-tds-benchmark.txt
```

If the lab GPU is an older generation (for example GTX 10-series) or its driver cannot be updated to CUDA 12.8, install from `https://download.pytorch.org/whl/cu126` instead.

Use `py -3.11` if Python 3.12 is not installed. The `$py` shortcut is used in every later step; if you open a new PowerShell window, set it again with the `$py = ...` line.

Confirm that PyTorch sees the GPU:

```powershell
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The second value must be `True`. If it is `False`, see Troubleshooting.

Run the model's unit tests (about a minute):

```powershell
& $py -m unittest tests.test_occlusion_model tests.test_occlusion_data tests.test_occlusion_pipeline tests.test_real_occlusion_evaluation -v
```

All tests should end with `OK`.

## Step 4: Check the three configurations

The repository includes one configuration per run. Each is the paper's configuration (`configs\occlusion_d405_30fps_20261105.json`) with only the settings below changed:

| Run | Config file | Changes from the paper's configuration |
| --- | --- | --- |
| From scratch | `configs\occlusion_d405_30fps_100ep.json` | `epochs` 100, `patience` 100 (early stopping never triggers) |
| Cloth pretraining | `configs\occlusion_cloth_pretrain_5fps.json` | 5 FPS timing: `history` 5 frames (800 ms), `horizon` 1 (200 ms), `expected_dt_ms` 200; mask moves every frame (`gaussian_step_interval` 1); `epochs` 50, `patience` 8; every window used (`train_window_stride` 1) |
| Fine-tuning | `configs\occlusion_d405_30fps_finetune.json` | Same as from scratch, plus `learning_rate` 0.0003 instead of 0.001 |

All three keep the same network size (16 base channels, 3 × 3 kernels), batch size 8, hidden-region weight 2, seed 42 and Gaussian masks covering 20% of the grid. The network size must match for the cloth weights to load.

With `patience` at 100, both servo runs train for all 100 epochs, and `best.pt` is still the epoch with the lowest validation objective. Both servo runs use the same epoch budget, so the comparison between them is fair.

Open any file to check it, for example `Get-Content configs\occlusion_d405_30fps_finetune.json`.

## Step 5: Run a one-epoch smoke test

Before committing 2.5 hours, train for one epoch to catch any setup problem in a few minutes:

```powershell
& $py -c "import json; c=json.load(open('configs/occlusion_d405_30fps_100ep.json')); c.update(epochs=1); json.dump(c, open('configs/smoke_1ep.json','w'), indent=2)"
& $py -m prediction.occlusion train --splits output\servo_30fps_camera_maps_20261105\splits.json --config configs\smoke_1ep.json --output output\smoke_1ep --device cuda
```

It should finish with a line like `epoch 1/1: validation objective 19.9 mm` (the first epoch of the earlier run scored 19.87 mm). Note how long it took: 100 epochs will take roughly 100 times as long, minus a little, because the first epoch also reads all frames into memory.

Then delete the test output:

```powershell
Remove-Item -Recurse output\smoke_1ep, configs\smoke_1ep.json
```

## Step 6a: Train from scratch for 100 epochs

First stop the computer from sleeping while plugged in (training stops if it sleeps):

```powershell
powercfg /change standby-timeout-ac 0
```

Start training. The console output is also saved to a log file next to the run folder:

```powershell
$run = "output\servo_30fps_convlstm_100ep_$(Get-Date -Format yyyyMMdd)"
& $py -m prediction.occlusion train --splits output\servo_30fps_camera_maps_20261105\splits.json --config configs\occlusion_d405_30fps_100ep.json --output $run --device cuda 2>&1 | Tee-Object -FilePath "$run.log"
```

Each epoch prints one line, for example `epoch 12/100: validation objective 11.92 mm`. Keep this PowerShell window open until the last epoch finishes.

**To watch progress** from a second PowerShell window:

```powershell
cd D:\DynamicSurfacePrinting
Get-Content (Get-ChildItem output\servo_30fps_convlstm_100ep_*.log | Select-Object -Last 1) -Wait -Tail 5
nvidia-smi
```

`history.json` in the run folder is rewritten after every epoch, and `best.pt` is overwritten whenever the validation objective reaches a new minimum.

**If training is interrupted** (window closed, crash, power loss), `run.json` records `interrupted` or `failed` and the best checkpoint so far is kept. Training cannot resume: start again with a new output folder name.

## Step 6b: Pretrain on the EPFL cloth data

This trains the same ConvLSTM + U-Net on the cloth sequences, with the same synthetic Gaussian masks, at the cloth data's 5 FPS timing. Steps 6a and 6b are independent and can run in either order; run them one after the other, not at the same time on one GPU.

```powershell
$cloth = "output\cloth_pretrain_$(Get-Date -Format yyyyMMdd)"
& $py -m prediction.occlusion train --splits output\cloth_tds_prepared\splits.json --config configs\occlusion_cloth_pretrain_5fps.json --output $cloth --device cuda 2>&1 | Tee-Object -FilePath "$cloth.log"
```

It uses the 12 training and 3 validation cloth sequences and stops after 8 epochs without improvement (at most 50). The 3 cloth test sequences are not used. Each epoch prints a validation objective in millimeters; it is much larger than on the servo data because the cloth moves more and the Kinect depth is noisier.

## Step 6c: Fine-tune on the servo data

Start from the cloth model's `best.pt`. Only its weights are copied: normalization is recomputed on the servo training data, the optimizer starts fresh, and the learning rate is 0.0003.

```powershell
$ft = "output\servo_30fps_finetune_$(Get-Date -Format yyyyMMdd)"
& $py -m prediction.occlusion train --splits output\servo_30fps_camera_maps_20261105\splits.json --config configs\occlusion_d405_30fps_finetune.json --output $ft --device cuda --initialize-from "$cloth\best.pt" 2>&1 | Tee-Object -FilePath "$ft.log"
```

This run uses the same servo split, test windows and 100-epoch budget as Step 6a. Its `run.json` and `best.pt` record the cloth checkpoint's path and SHA-256 under `initialization`, so the paper can state exactly which pretrained weights were used.

If `$cloth` is no longer set (new PowerShell window), set it to the cloth run's folder first, for example `$cloth = "output\cloth_pretrain_20261108"`.

When all three training runs (Steps 6a–6c) have finished, restore the sleep setting (30 minutes is a common default):

```powershell
powercfg /change standby-timeout-ac 30
```

## Step 7: Check the training run

Print the run's status, number of epochs and best epoch:

```powershell
& $py -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['status'], r['epochs_completed'], 'best epoch', r['best_epoch'], 'objective', round(r['best_validation_objective_mm'],3), 'mm')" "$run\run.json"
```

Expected: `complete 100 best epoch <N> objective <value> mm`.

Check the other two runs the same way, replacing `"$run\run.json"` with `"$cloth\run.json"` and `"$ft\run.json"`. The cloth run normally ends with fewer than 50 epochs, because early stopping is on.

Confirm that the fine-tuned run started from the cloth weights:

```powershell
& $py -c "import json,sys; i=json.load(open(sys.argv[1]))['initialization']; print(i['kind'], i['source_checkpoint'], i['source_shape'])" "$ft\run.json"
```

Expected: `pretrained_weights_only <path to the cloth best.pt> [32, 32]`.

The run is on a different GPU, and GPU arithmetic is not exactly repeatable, so even the first 30 epochs will not match the earlier run digit for digit. Compare the final results, not individual epochs.

If you opened a new PowerShell window, set `$run` again to the exact folder name first, for example `$run = "output\servo_30fps_convlstm_100ep_20261108"`.

## Step 8: Evaluate the new model

Run both evaluations with the new `best.pt`. Each takes minutes, not hours.

**Synthetic occlusion on the held-out patterns** (Table 1 and Fig. 1). The `--splits` path must be exactly the one used for training; the script refuses any other.

```powershell
$syn = "output\servo_30fps_100ep_synthetic_eval_$(Get-Date -Format yyyyMMdd)"
& $py -m tools.evaluate_occlusion_heldout --checkpoint "$run\best.pt" --splits output\servo_30fps_camera_maps_20261105\splits.json --output $syn --device cuda
```

It prints `"windows": 1740` and `"mask_seed": 44`, the same test windows and masks as the paper.

**Real print-head test** (Table 2). This reuses the prepared real-head data, so the head mask and frame pairing are identical to the paper's:

```powershell
$real = "output\real_occlusion_100ep_eval_$(Get-Date -Format yyyyMMdd)"
& $py -m tools.evaluate_real_occlusion_repeats --prepared output\real_occlusion_prior_repeats_20261105 --checkpoint "$run\best.pt" --output $real --device cuda
& $py -m tools.visualize_real_occlusion_repeat --prepared output\real_occlusion_prior_repeats_20261105\21_amplitude_beats.npz --predictions "$real\21_amplitude_beats.npz" --output "$real\example.png"
```

The real-head evaluation reads the training data's calibration through the path stored in `best.pt`, so run it on the lab computer, where that data is.

**Repeat both evaluations for the fine-tuned model:**

```powershell
$ftsyn = "output\servo_30fps_finetune_synthetic_eval_$(Get-Date -Format yyyyMMdd)"
& $py -m tools.evaluate_occlusion_heldout --checkpoint "$ft\best.pt" --splits output\servo_30fps_camera_maps_20261105\splits.json --output $ftsyn --device cuda
$ftreal = "output\real_occlusion_finetune_eval_$(Get-Date -Format yyyyMMdd)"
& $py -m tools.evaluate_real_occlusion_repeats --prepared output\real_occlusion_prior_repeats_20261105 --checkpoint "$ft\best.pt" --output $ftreal --device cuda
& $py -m tools.visualize_real_occlusion_repeat --prepared output\real_occlusion_prior_repeats_20261105\21_amplitude_beats.npz --predictions "$ftreal\21_amplitude_beats.npz" --output "$ftreal\example.png"
```

The cloth model itself is not evaluated on these tests: it was trained for 5 FPS on a different grid, and the evaluation scripts reject it.

## Step 9: Bring the results back and update the paper

Zip all result folders and the training logs:

```powershell
Compress-Archive -Path $run, "$run.log", $syn, $real, $cloth, "$cloth.log", $ft, "$ft.log", $ftsyn, $ftreal -DestinationPath $env:USERPROFILE\Desktop\occlusion_100ep_results.zip
```

On the laptop, unzip it into `D:\HAIM_Lab\Dynamic_Surface_Printer\output`. Then ask Claude to update the paper from those folders. These parts depend on the new run:

| Paper part | What changes |
| --- | --- |
| Methodology, Model Selection | The paragraph on training from random weights becomes pretraining on the EPFL cloth data and fine-tuning on the servo data (if the fine-tuned model is kept), with the dataset citation |
| Methodology, training paragraph | 100 epochs with early stopping disabled, best checkpoint kept; cloth pretraining settings and the fine-tuning learning rate |
| Results, Model Training | Best epochs, validation objectives, learning curves |
| Table 1 and Fig. 1 | All synthetic-occlusion numbers, possibly with a column per model |
| Table 2 and the real print-head text | All model columns (the "Before prediction" column does not change) |
| Real print-head example figure | The new `example.png` of the model you report |

Keep the 30-epoch folders on the laptop. Report whichever comparison the numbers support: if fine-tuning does not beat training from scratch, or 100 epochs do not beat 30, that is still a result worth stating.

## Round 2: Training with head-scale masks

Round 1 showed that 100 epochs improve the synthetic test but fail on the real print head (34 mm from scratch, 14.5 mm fine-tuned, against 5.8 mm for the 30-epoch model). The training masks hide 20% of the grid, while the real head hides about 80%. Round 2 trains with a mix of both: each training window gets either the paper's moving Gaussian (20%) or a large Gaussian blob covering 40–90% of the grid that stays still half the time. Validation uses the same mix, so the selected epoch must also handle large occlusions.

It needs about 1 hour of GPU time and the data and cloth model already on the lab computer from round 1.

**1. Apply the round-2 code.** Copy `occlusion_code_update_v2.zip` from the laptop's project folder (it replaces the round-1 zip and contains all its changes). Also copy the paper model `output\servo_30fps_convlstm_20261105\best.pt` from the laptop to the same path on the lab computer.

```powershell
cd C:\Users\haim_\Desktop\DynamicSurfacePrinting
Expand-Archive -Path <path to>\occlusion_code_update_v2.zip -DestinationPath . -Force
& $py -m unittest tests.test_occlusion_data tests.test_occlusion_pipeline
```

**2. Train the two mixed-mask models.** The configs differ from round 1 only in their masks:

| Config | Masks | Start | Learning rate |
| --- | --- | --- | --- |
| `configs\occlusion_d405_30fps_mixed_100ep.json` | Mixed: 50% paper Gaussian (20%), 50% large blob (40–90%, stationary half the time) | Random | 0.001 |
| `configs\occlusion_d405_30fps_mixed_finetune.json` | Same | Round-1 cloth model | 0.0003 |

```powershell
$d = Get-Date -Format yyyyMMdd
$splits = "output\servo_30fps_camera_maps_20261105\splits.json"
$mix = "output\servo_30fps_mixed_100ep_$d"
& $py -m prediction.occlusion train --splits $splits --config configs\occlusion_d405_30fps_mixed_100ep.json --output $mix --device cuda 2>&1 | Tee-Object -FilePath "$mix.log"
$mixft = "output\servo_30fps_mixed_finetune_$d"
& $py -m prediction.occlusion train --splits $splits --config configs\occlusion_d405_30fps_mixed_finetune.json --output $mixft --device cuda --initialize-from output\cloth_pretrain_20261107\best.pt 2>&1 | Tee-Object -FilePath "$mixft.log"
```

**3. Evaluate.** A new synthetic test with large stationary masks (70–90% of the grid) has true labels, unlike the real-head test, so it is run for all five servo models. The two new models also get the paper's synthetic test and the real print-head test.

```powershell
$out = "output\eval_round2_$d"
$models = [ordered]@{ "30ep" = "output\servo_30fps_convlstm_20261105\best.pt"; "100ep" = "output\servo_30fps_convlstm_100ep_20261107\best.pt"; "finetune" = "output\servo_30fps_finetune_20261107\best.pt"; "mixed_100ep" = "$mix\best.pt"; "mixed_finetune" = "$mixft\best.pt" }
foreach ($name in $models.Keys) {
  & $py -m prediction.occlusion evaluate --checkpoint $models[$name] --splits $splits --split test --mask-config configs\masks\large_stationary.json --output "$out\${name}_large_masks" --device cuda > $null
}
foreach ($name in "mixed_100ep", "mixed_finetune") {
  & $py -m prediction.occlusion evaluate --checkpoint $models[$name] --splits $splits --split test --mask-config configs\masks\paper_gaussian20.json --output "$out\${name}_paper_masks" --device cuda > $null
  & $py -m tools.evaluate_real_occlusion_repeats --prepared output\real_occlusion_prior_repeats_20261105 --checkpoint $models[$name] --output "$out\${name}_real_head" --device cuda
  & $py -m tools.visualize_real_occlusion_repeat --prepared output\real_occlusion_prior_repeats_20261105\21_amplitude_beats.npz --predictions "$out\${name}_real_head\21_amplitude_beats.npz" --output "$out\${name}_real_head\example.png"
}
```

The 30-epoch model's real-head test cannot be rerun on the lab computer (it needs the laptop's data paths); its November 6 result is used instead.

**4. Bring the results back:**

```powershell
Compress-Archive -Path $mix, "$mix.log", $mixft, "$mixft.log", $out -DestinationPath $env:USERPROFILE\Desktop\occlusion_round2_results.zip
```

## Troubleshooting

| Message or symptom | Cause | Fix |
| --- | --- | --- |
| `torch.cuda.is_available()` prints `False`, or `Torch not compiled with CUDA enabled` | The CPU build of PyTorch was installed, or the NVIDIA driver is too old for CUDA 12.8 | `& $py -m pip uninstall -y torch`, update the NVIDIA driver, then repeat the `cu128` install from Step 3 |
| `FileExistsError: output must be a new or empty directory` | The output folder already exists | Use a new folder name, for example add `_b` to `$run` |
| `No module named 'prediction'` or `'tools'` | PowerShell is not in the repository root | `cd D:\DynamicSurfacePrinting` and run again |
| `split path differs from the original training split` | `--splits` in Step 8 is not the path used for training | Use `output\servo_30fps_camera_maps_20261105\splits.json` exactly, from the same clone |
| `FileNotFoundError` for a `manifest.json` during the real-head test | The height-map folder was moved or renamed after training | Put it back where it was during training |
| `unknown configuration fields` | A typo in the configuration file | Recreate it with the one-line command in Step 4 |
| `CUDA out of memory` | Another program is using the GPU | Close it (check with `nvidia-smi`). Do not change the batch size, because that changes the experiment |
| Red `NativeCommandError` lines around Python warnings | Windows PowerShell shows anything Python writes to its error stream in red | Harmless if the epoch lines keep appearing |
| Training stopped overnight | The computer slept or restarted | Run `powercfg` as in Step 6 and start again with a new folder name |
| `initialization architecture differs from target: base_channels` | The fine-tuning config has a different network size from the cloth model | Use the configs from the repository unchanged; base\_channels and kernel\_size must match |
| `unrecognized arguments: --initialize-from` | The fine-tuning code update was not applied | Apply the code update zip from Step 1 |

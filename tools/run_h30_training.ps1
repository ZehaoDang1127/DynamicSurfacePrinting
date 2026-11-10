<#
1 s forecast horizon: train, evaluate, and score predicted motion against the commanded servo motion.

Run from PowerShell in the repository root on the lab desktop (RTX 5090), with the
existing output\ folder in place (round-1/2 models, height maps, real-head data):

    powershell -ExecutionPolicy Bypass -File .\tools\run_h30_training.ps1 -Recordings <folder>

-Recordings is the folder that holds the 20 servo recording folders
(01_triangle_sweep ... 20_biased_oscillation), each with frames.csv and trajectory.csv,
e.g. ...\Dataset\recordings\servo_patterns_20_30fps_20261105. Only those two CSV files are read.

Steps (finished steps are detected and skipped when the script is started again):
  1  calibrate the commanded-angle -> camera-depth map and servo lag on the 16 training recordings
  2  train mixed-mask models with a 30-step (1 s) horizon: from scratch, and fine-tuned from the cloth model
  3  synthetic test (paper masks and large stationary masks), all leads 0-30
  4  real print-head predictions at 6, 15 and 30 steps (h30 models and the current 200 ms primary model)
  5  motion-delta scoring (synthetic and real head) for the h30 models and the current primary model
  6  results zip on the Desktop (occlusion_h30_results.zip)

Progress: output\run_h30_logs\status.log
#>
param(
    [Parameter(Mandatory = $true)][string]$Recordings,
    [int]$Epochs = 0
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONUNBUFFERED = '1'
$LogDir = Join-Path $Root 'output\run_h30_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$StatusLog = Join-Path $LogDir 'status.log'

$Py = Join-Path $Root '.venv-train\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Py)) { $Py = 'python' }
$Splits = 'output\servo_30fps_camera_maps_20261105\splits.json'
$RealData = 'output\real_occlusion_prior_repeats_20261105'
$ClothBest = 'output\cloth_pretrain_20261107\best.pt'
$PrimaryH6 = 'output\servo_30fps_mixed_100ep_20261108\best.pt'
$Out = 'output\h30_eval'
$Cal = 'output\motion_calibration'

function Log([string]$Message) {
    $line = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line -ForegroundColor Cyan
    Add-Content -LiteralPath $StatusLog -Value $line -Encoding UTF8
}

function Run([string]$Name, [string[]]$Arguments) {
    Log "START $Name"
    & $script:Py @Arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $LogDir "$Name.log") -Append | Out-Host
    if ($LASTEXITCODE -ne 0) { Log "FAILED $Name (exit $LASTEXITCODE)"; throw "$Name failed" }
    Log "DONE  $Name"
}

function Done([string]$Path) { return (Test-Path -LiteralPath $Path) }

foreach ($p in @($Splits, $RealData, $ClothBest, $PrimaryH6, (Join-Path $Recordings '18_stop_start\trajectory.csv'))) {
    if (-not (Test-Path -LiteralPath $p)) { Log "MISSING $p"; throw "missing $p" }
}
Run 'unittests' @('-m', 'unittest', 'tests.test_motion_delta', 'tests.test_real_occlusion_evaluation', 'tests.test_occlusion_pipeline')

# 1  calibration (training recordings only)
if (-not (Done "$Cal\calibration.npz")) {
    Run 'calibrate' @('-m', 'tools.evaluate_motion_delta', 'calibrate', '--splits', $Splits, '--recordings', $Recordings, '--output', $Cal)
}

# 2  training
$models = [ordered]@{}
foreach ($spec in @(@{name = 'h30_mixed'; config = 'configs\occlusion_d405_30fps_mixed_100ep_h30.json'; init = $null},
                    @{name = 'h30_mixed_finetune'; config = 'configs\occlusion_d405_30fps_mixed_finetune_h30.json'; init = $ClothBest})) {
    $dir = "output\servo_30fps_$($spec.name)"
    $config = $spec.config
    if ($Epochs -gt 0) {
        $tmp = Join-Path $LogDir "$($spec.name)_config.json"
        $c = Get-Content -LiteralPath $config -Raw | ConvertFrom-Json
        $c.epochs = $Epochs; $c.patience = $Epochs
        $c | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $tmp -Encoding UTF8
        $config = $tmp
    }
    $run = if (Done "$dir\run.json") { Get-Content -LiteralPath "$dir\run.json" -Raw | ConvertFrom-Json } else { $null }
    if (-not $run -or $run.status -ne 'complete') {
        if (Done $dir) { $dir = "$dir`_$(Get-Date -Format HHmmss)" }
        $trainArgs = @('-m', 'prediction.occlusion', 'train', '--splits', $Splits, '--config', $config, '--output', $dir, '--device', 'cuda')
        if ($spec.init) { $trainArgs += @('--initialize-from', $spec.init) }
        Run "train_$($spec.name)" $trainArgs
    }
    $models[$spec.name] = "$dir\best.pt"
}
$models['h6_primary'] = $PrimaryH6

# 3  synthetic test, all leads of each model's own horizon
foreach ($name in @('h30_mixed', 'h30_mixed_finetune')) {
    foreach ($mask in @(@{tag = 'paper_masks'; cfg = 'configs\masks\paper_gaussian20.json'}, @{tag = 'large_masks'; cfg = 'configs\masks\large_stationary.json'})) {
        $dir = "$Out\${name}_$($mask.tag)"
        if (-not (Done "$dir\metrics.json")) {
            Run "eval_${name}_$($mask.tag)" @('-m', 'prediction.occlusion', 'evaluate', '--checkpoint', $models[$name], '--splits', $Splits,
                '--split', 'test', '--mask-config', $mask.cfg, '--output', $dir, '--device', 'cuda')
        }
    }
}

# 4  real print head, predictions saved at 6, 15 and 30 steps
foreach ($name in $models.Keys) {
    $dir = "$Out\${name}_real_head"
    if (-not (Done "$dir\metrics.json")) {
        Run "real_head_$name" @('-m', 'tools.evaluate_real_occlusion_repeats', '--prepared', $RealData, '--checkpoint', $models[$name],
            '--output', $dir, '--device', 'cuda', '--steps', '6', '15', '30')
    }
}

# 5  motion-delta scoring
foreach ($name in $models.Keys) {
    foreach ($mask in @(@{tag = 'paper_masks'; cfg = 'configs\masks\paper_gaussian20.json'}, @{tag = 'large_masks'; cfg = 'configs\masks\large_stationary.json'})) {
        $dir = "$Out\${name}_motion_$($mask.tag)"
        if (-not (Done "$dir\motion_delta.json")) {
            Run "motion_${name}_$($mask.tag)" @('-m', 'tools.evaluate_motion_delta', 'synthetic', '--checkpoint', $models[$name], '--splits', $Splits,
                '--recordings', $Recordings, '--calibration', "$Cal\calibration.npz", '--output', $dir, '--device', 'cuda',
                '--mask-config', $mask.cfg, '--steps', '6', '15', '30')
        }
    }
    $dir = "$Out\${name}_motion_real_head"
    if (-not (Done "$dir\motion_delta_real_head.json")) {
        Run "motion_${name}_real_head" @('-m', 'tools.evaluate_motion_delta', 'real-head', '--predictions', "$Out\${name}_real_head",
            '--prepared', $RealData, '--calibration', "$Cal\calibration.npz", '--output', $dir)
    }
}

# 6  results zip (JSON, logs and run metadata; no checkpoints or prediction arrays)
$zip = Join-Path ([Environment]::GetFolderPath('Desktop')) 'occlusion_h30_results.zip'
$stage = Join-Path $env:TEMP 'occlusion_h30_results'
if (Test-Path -LiteralPath $stage) { Remove-Item -Recurse -Force -LiteralPath $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
foreach ($src in @($Out, $Cal, $LogDir) + @($models.Values | ForEach-Object { Split-Path -Parent $_ })) {
    Get-ChildItem -LiteralPath $src -Recurse -File -Include *.json, *.log, *.png | ForEach-Object {
        $rel = $_.FullName.Substring($Root.Length + 1)
        $dest = Join-Path $stage $rel
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $dest
    }
}
Compress-Archive -Path "$stage\*" -DestinationPath $zip -Force
Log "ALL DONE  results: $zip"

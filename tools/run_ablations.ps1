# Train and evaluate the architecture ablations described in
# docs/occlusion_forecasting/ABLATIONS.md. Run from the repository root in PowerShell.
# A run whose folder already holds a completed run.json is not retrained, and an
# evaluation whose folder already holds metrics.json is skipped, so the script
# can be restarted after an interruption (delete the interrupted run folder first).
param(
    [string]$Python = ".\.venv-train\Scripts\python.exe",
    [string]$Splits = "output\servo_30fps_camera_maps_20261105\splits.json",
    [string]$RealHead = "output\real_occlusion_prior_repeats_20261105",
    [string]$Out = "output\ablations_$(Get-Date -Format yyyyMMdd)",
    # Round-2 evaluations of the primary model, added to the summary when present.
    [string]$Primary = "output\eval_round2_20261108\mixed_100ep",
    [string[]]$Runs = @("depth_mask", "stacked_cells", "primary_repeat", "depth_only", "depth_mask_age", "single_cell")
)

$configs = @{
    "primary_repeat" = "configs\occlusion_d405_30fps_mixed_100ep.json"
    "depth_only"     = "configs\ablations\mixed_100ep_depth_only.json"
    "depth_mask"     = "configs\ablations\mixed_100ep_depth_mask.json"
    "depth_mask_age" = "configs\ablations\mixed_100ep_depth_mask_age.json"
    "single_cell"    = "configs\ablations\mixed_100ep_single_cell.json"
    "stacked_cells"  = "configs\ablations\mixed_100ep_stacked_cells.json"
}

New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($name in $Runs) {
    if (-not $configs.ContainsKey($name)) { Write-Error "unknown run $name"; exit 1 }
    $run = Join-Path $Out $name
    $complete = (Test-Path "$run\run.json") -and ((Get-Content "$run\run.json" -Raw | ConvertFrom-Json).status -eq "complete")
    if (-not $complete) {
        Write-Host "=== training $name ($($configs[$name])) ==="
        & $Python -m prediction.occlusion train --splits $Splits --config $configs[$name] --output $run --device cuda 2>&1 | Tee-Object -FilePath "$run.log"
        if ($LASTEXITCODE -ne 0) { Write-Error "training $name failed; see $run.log"; exit 1 }
    }
    foreach ($masks in @(@("paper_masks", "paper_gaussian20"), @("large_masks", "large_stationary"))) {
        $eval = Join-Path $Out "${name}_$($masks[0])"
        if (-not (Test-Path "$eval\metrics.json")) {
            Write-Host "=== evaluating $name with $($masks[1]) masks ==="
            & $Python -m prediction.occlusion evaluate --checkpoint "$run\best.pt" --splits $Splits --split test --mask-config "configs\masks\$($masks[1]).json" --output $eval --device cuda > $null
            if ($LASTEXITCODE -ne 0) { Write-Error "evaluating $name failed"; exit 1 }
        }
    }
    $real = Join-Path $Out "${name}_real_head"
    if ((Test-Path $RealHead) -and -not (Test-Path "$real\metrics.json")) {
        Write-Host "=== real print-head test of $name ==="
        & $Python -m tools.evaluate_real_occlusion_repeats --prepared $RealHead --checkpoint "$run\best.pt" --output $real --device cuda > "$real.log"
        if ($LASTEXITCODE -ne 0) { Write-Error "real-head test of $name failed; see $real.log"; exit 1 }
    }
}

$summary = @("-m", "tools.summarize_ablations", "--root", $Out, "--output", (Join-Path $Out "summary"))
if (Test-Path "$Primary`_paper_masks\metrics.json") { $summary += @("--model", "primary=$Primary") }
& $Python @summary

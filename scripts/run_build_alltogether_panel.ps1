# Build 2x4 "alltogether" panels (needs only Pillow + numpy; no torch required).
# If you extend the script later to use torch, use the same venv block as run_export_local_assets.ps1.

$Root = "C:\TFG\TFGpractice"
Set-Location $Root

# Optional: same venv as export if you use a single `python` on PATH
$VenvDir = if ($env:TFG_VENV_ROOT) { $env:TFG_VENV_ROOT } else { Join-Path $Root ".venv" }
$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $Activate) {
    . $Activate
    Write-Host "[venv] Activated: $VenvDir"
}

python scripts/build_alltogether_panel.py `
  --data-root "D:\Dataset_Imagenes" `
  --split test `
  --hdf5 "$Root\Datasets\CKM_Dataset_180326_antenna_height.h5"
# --spread-label defaults to auto (picks predictions_*_delay_angular for this split).
# Must match export: same --split and you need --spread-checkpoint in export for delay/angular tiles.

# Optional: limit samples while testing
#   --limit 20
# Optional: larger tiles
#   --cell-w 384 --cell-h 384 --label-bar-h 64

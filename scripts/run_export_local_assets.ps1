# Full export: HDF5 -> raw_hdf5 + model predictions (Ninth + Second).
# Predictions-only (skip re-exporting 10k+ PNGs): use run_export_predictions_only.ps1 or add --skip-dataset-export.
# Requires PyTorch. Edit $VenvDir if your env is not TFGpractice\.venv

$Root = "C:\TFG\TFGpractice"
Set-Location $Root

$VenvDir = if ($env:TFG_VENV_ROOT) { $env:TFG_VENV_ROOT } else { Join-Path $Root ".venv" }
$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $Activate) {
    . $Activate
    Write-Host "[venv] Activated: $VenvDir"
} else {
    Write-Warning "No Activate.ps1 at $Activate — activate your PyTorch venv manually, or set env TFG_VENV_ROOT."
}

python scripts/export_dataset_and_predictions.py `
  --hdf5 "$Root\Datasets\CKM_Dataset_180326_antenna_height.h5" `
  --scalar-csv "$Root\Datasets\CKM_180326_antenna_height.csv" `
  --dataset-out "D:\Dataset_Imagenes" `
  --device auto `
  --ninth-root "$Root\TFGNinthTry9" `
  --ninth-config "$Root\TFGNinthTry9\configs\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9.yaml" `
  --ninth-checkpoint "$Root\cluster_outputs\TFGNinthTry9\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9_ddp2_ninthtry9\best_cgan.pt" `
  --spread-try second `
  --spread-config "$Root\TFGSecondTry2\configs\cgan_unet_hdf5_amd_midvram.yaml" `
  --spread-checkpoint "$Root\TFGSecondTry2\outputs\cgan_unet_hdf5_amd_midvram\best_cgan.pt"

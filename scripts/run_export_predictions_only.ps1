# Run ONLY model inference (path loss + delay/angular GT|Pred PNGs).
# Does NOT re-export raw_hdf5 from HDF5 — use after a full export or if raw_hdf5 already exists.
# Still needs --hdf5 for the dataloader split / samples (same file as training).
# --scalar-csv matches cluster Slurm norms for NinthTry9 path loss (with antenna_height HDF5).
# Change --device to cpu or auto if needed (DirectML is fine on AMD once scalar norms match training).

$Root = "C:\TFG\TFGpractice"
Set-Location $Root

$VenvDir = if ($env:TFG_VENV_ROOT) { $env:TFG_VENV_ROOT } else { Join-Path $Root ".venv" }
$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $Activate) {
    . $Activate
    Write-Host "[venv] Activated: $VenvDir"
} else {
    Write-Warning "No venv at $VenvDir — activate manually or set TFG_VENV_ROOT."
}

python scripts/export_dataset_and_predictions.py `
  --skip-dataset-export `
  --hdf5 "$Root\Datasets\CKM_Dataset_180326_antenna_height.h5" `
  --scalar-csv "$Root\Datasets\CKM_180326_antenna_height.csv" `
  --dataset-out "D:\Dataset_Imagenes" `
  --device directml `
  --split all `
  --ninth-root "$Root\TFGNinthTry9" `
  --ninth-config "$Root\TFGNinthTry9\configs\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9.yaml" `
  --ninth-checkpoint "$Root\cluster_outputs\TFGNinthTry9\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_ninthtry9_ddp2_ninthtry9\best_cgan.pt" `
  --spread-try second `
  --spread-config "$Root\TFGSecondTry2\configs\cgan_unet_hdf5_amd_midvram.yaml" `
  --spread-checkpoint "$Root\TFGSecondTry2\outputs\cgan_unet_hdf5_amd_midvram\best_cgan.pt"

# Optional: --limit 50 --split val

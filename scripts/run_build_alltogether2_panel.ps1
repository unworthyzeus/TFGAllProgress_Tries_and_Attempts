$Root = "C:\TFG\TFGpractice"
Set-Location $Root

$VenvDir = if ($env:TFG_VENV_ROOT) { $env:TFG_VENV_ROOT } else { Join-Path $Root ".venv" }
$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $Activate) {
    . $Activate
    Write-Host "[venv] Activated: $VenvDir"
} else {
    Write-Warning "No venv at $VenvDir - activate manually or set TFG_VENV_ROOT."
}

python scripts/build_alltogether2_panel.py `
  --data-root "D:\Dataset_Imagenes" `
  --split all `
  --path-loss-label "twentyeighthtry28" `
  --spread-label "twentysixthtry26"

$ErrorActionPreference = 'Stop'

$workDir = 'C:\TFG\TFGpractice\TFGSeventyFifthTry75'
$outDir = Join-Path $workDir 'outputs\try75_expert_allcity_los_local_ep2'
$resumeConfig = 'experiments/seventyfifth_try75_experts/try75_expert_allcity_los_local_directml_resume_ep5.yaml'
$pythonExe = 'C:\TFG\.venv\Scripts\python.exe'
$stdoutLog = Join-Path $outDir 'resume_ep5_stdout.log'
$stderrLog = Join-Path $outDir 'resume_ep5_stderr.log'
$sentinel = Join-Path $outDir 'resume_ep5_queued.flag'

Set-Location $workDir
Set-Content -LiteralPath $sentinel -Value ("queued_at=" + (Get-Date).ToString('s'))

while ($true) {
    $active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -like 'python*' -and
        $_.CommandLine -like '*train_partitioned_pathloss_expert.py*' -and
        $_.CommandLine -like '*try75_expert_allcity_los_local_directml_resume_ep3.yaml*'
    }
    if (-not $active) { break }
    Start-Sleep -Seconds 30
}

$alreadyRunning = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*' -and
    $_.CommandLine -like '*train_partitioned_pathloss_expert.py*' -and
    $_.CommandLine -like '*try75_expert_allcity_los_local_directml_resume_ep5.yaml*'
}

if (-not $alreadyRunning) {
    if (Test-Path $stdoutLog) { Remove-Item -LiteralPath $stdoutLog -Force }
    if (Test-Path $stderrLog) { Remove-Item -LiteralPath $stderrLog -Force }
    Start-Process -FilePath $pythonExe `
        -ArgumentList @('train_partitioned_pathloss_expert.py', '--config', $resumeConfig) `
        -WorkingDirectory $workDir `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog | Out-Null
}

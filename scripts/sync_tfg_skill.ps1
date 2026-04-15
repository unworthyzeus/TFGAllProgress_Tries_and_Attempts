<#
.SYNOPSIS
    Sincroniza el skill del TFG (path loss CKM) y, opcionalmente, settings.local.json entre herramientas.

.DESCRIPTION
    - Copia CLAUDE.md desde la raiz del repo a .cursor/skills/tfg-pathloss-project/CLAUDE.md
    - Copia SKILL.md y AGENTS.md desde .cursor/skills/tfg-pathloss-project/ al resto de destinos
    - Con -IncludeSettings: copia .claude/settings.local.json -> .codex/ y .cursor/

    Raiz del repo = dos niveles por encima de este script (TFGpractice/scripts -> TFG).

.PARAMETER UserSkills
    Tambien copia a $env:USERPROFILE\.claude\skills y .codex\skills (skills globales).

.PARAMETER IncludeSettings
    Replica settings.local.json desde .claude hacia .codex y .cursor en la raiz del repo.

.EXAMPLE
    powershell -NoProfile -File TFGpractice/scripts/sync_tfg_skill.ps1

.EXAMPLE
    powershell -NoProfile -File TFGpractice/scripts/sync_tfg_skill.ps1 -UserSkills -IncludeSettings
#>
[CmdletBinding()]
param(
    [switch]$UserSkills,
    [switch]$IncludeSettings
)

$ErrorActionPreference = 'Stop'
$SkillName = 'tfg-pathloss-project'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$SkillSrc = Join-Path $RepoRoot (Join-Path '.cursor' (Join-Path 'skills' $SkillName))
$ClaudeRoot = Join-Path $RepoRoot 'CLAUDE.md'

if (-not (Test-Path -LiteralPath $ClaudeRoot)) {
    Write-Error "No existe CLAUDE.md en la raiz: $ClaudeRoot"
}
if (-not (Test-Path -LiteralPath $SkillSrc)) {
    Write-Error "No existe carpeta skill fuente: $SkillSrc"
}

$files = @('SKILL.md', 'AGENTS.md', 'CLAUDE.md')

# 1) CLAUDE.md del repo -> skill fuente (.cursor)
Copy-Item -LiteralPath $ClaudeRoot -Destination (Join-Path $SkillSrc 'CLAUDE.md') -Force
Write-Host "[ok] CLAUDE.md -> $SkillSrc\CLAUDE.md"

# Fuente de SKILL.md / AGENTS.md: solo .cursor/skills/... (no se copia a si misma)
$dests = [System.Collections.Generic.List[string]]::new()
foreach ($p in @(
        (Join-Path $RepoRoot (Join-Path '.claude' (Join-Path 'skills' $SkillName))),
        (Join-Path $RepoRoot (Join-Path '.codex' (Join-Path 'skills' $SkillName)))
    )) {
    $dests.Add($p) | Out-Null
}
if ($UserSkills) {
    foreach ($p in @(
            (Join-Path $env:USERPROFILE (Join-Path '.claude' (Join-Path 'skills' $SkillName))),
            (Join-Path $env:USERPROFILE (Join-Path '.codex' (Join-Path 'skills' $SkillName)))
        )) {
        $dests.Add($p) | Out-Null
    }
}

foreach ($d in $dests) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    foreach ($f in $files) {
        $from = Join-Path $SkillSrc $f
        if (-not (Test-Path -LiteralPath $from)) {
            Write-Error "Falta archivo en fuente: $from"
        }
        Copy-Item -LiteralPath $from -Destination (Join-Path $d $f) -Force
    }
    Write-Host "[ok] skill -> $d"
}

if ($IncludeSettings) {
    $srcSettings = Join-Path $RepoRoot (Join-Path '.claude' 'settings.local.json')
    if (-not (Test-Path -LiteralPath $srcSettings)) {
        Write-Error "No existe $srcSettings (canonico para permisos)."
    }
    foreach ($toolDir in @('.codex', '.cursor')) {
        $dir = Join-Path $RepoRoot $toolDir
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $dst = Join-Path $dir 'settings.local.json'
        Copy-Item -LiteralPath $srcSettings -Destination $dst -Force
        Write-Host "[ok] settings.local.json -> $dst"
    }
}

Write-Host "Listo. Raiz: $RepoRoot"

param()

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Submodule = Join-Path $Project 'source\dppo_v0.6'
$Patch = Join-Path $Project 'reports\PERFORMANCE_PATCH.diff'
$Expected = 'dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc'

if (-not (Test-Path -LiteralPath (Join-Path $Submodule '.git'))) {
    throw "Missing submodule: $Submodule"
}
if (-not (Test-Path -LiteralPath $Patch)) {
    throw "Missing patch: $Patch"
}

$Sha = (git -C $Submodule rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $Sha -ne $Expected) {
    throw "Official submodule SHA check failed. Expected $Expected, got $Sha"
}

$AlreadyApplied = $false
try {
    git -C $Submodule apply --reverse --check -- $Patch 2>$null
    $AlreadyApplied = ($LASTEXITCODE -eq 0)
} catch {
    # A failed reverse check is the expected result for a clean submodule.
    $AlreadyApplied = $false
}
if ($AlreadyApplied) {
    Write-Output "Performance patch already applied: $Submodule"
    exit 0
}

git -C $Submodule apply --check -- $Patch
if ($LASTEXITCODE -ne 0) {
    throw 'Patch is neither cleanly applied nor cleanly applicable.'
}
git -C $Submodule apply -- $Patch
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to apply performance patch.'
}
Write-Output "Performance patch applied: $Submodule"

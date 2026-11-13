param(
    [switch]$PreflightOnly,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot

$ManifestPath = Join-Path $RepositoryRoot 'experiments\manifests\phase1-test-analysis.json'
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json

function Resolve-RepositoryPath([string]$RelativePath) {
    return Join-Path $RepositoryRoot ($RelativePath -replace '/', '\')
}

function Assert-FileHash(
    [string]$Label,
    [string]$RelativePath,
    [string]$ExpectedHash
) {
    $Path = Resolve-RepositoryPath $RelativePath
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label does not exist: $Path"
    }
    $ActualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $ExpectedHash.ToLowerInvariant()) {
        throw "$Label hash mismatch: expected $ExpectedHash, found $ActualHash"
    }
    Write-Host "OK $Label $ActualHash"
}

$GitStatus = @(git status --porcelain --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect Git status'
}
if ($GitStatus.Count -ne 0) {
    throw "The repository must be clean before test access:`n$($GitStatus -join "`n")"
}

Assert-FileHash 'test metadata' $Manifest.dataset.metadata $Manifest.dataset.metadata_sha256
Assert-FileHash 'evaluation driver' $Manifest.evaluation.driver $Manifest.evaluation.driver_sha256
Assert-FileHash 'aggregation driver' $Manifest.evaluation.aggregator $Manifest.evaluation.aggregator_sha256
Assert-FileHash 'test launcher' $Manifest.evaluation.launcher $Manifest.evaluation.launcher_sha256
Assert-FileHash 'C config' $Manifest.configs.C.path $Manifest.configs.C.sha256
Assert-FileHash 'D config' $Manifest.configs.D.path $Manifest.configs.D.sha256

$Seeds = @('20261108', '20261109', '20261110')
foreach ($Seed in $Seeds) {
    $Run = $Manifest.seeds.$Seed
    Assert-FileHash "C checkpoint $Seed" $Run.C_checkpoint $Run.C_checkpoint_sha256
    Assert-FileHash "D checkpoint $Seed" $Run.D_checkpoint $Run.D_checkpoint_sha256
}

$Commit = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to resolve the Git revision'
}
Write-Host "Preflight passed at clean commit $Commit"

if ($PreflightOnly) {
    exit 0
}

$Python = Resolve-RepositoryPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment does not exist: $Python"
}

$Evaluator = Resolve-RepositoryPath $Manifest.evaluation.driver
$Dataset = Resolve-RepositoryPath $Manifest.dataset.root
$Metadata = Resolve-RepositoryPath $Manifest.dataset.metadata
$CConfig = Resolve-RepositoryPath $Manifest.configs.C.path
$DConfig = Resolve-RepositoryPath $Manifest.configs.D.path

foreach ($Seed in $Seeds) {
    $Run = $Manifest.seeds.$Seed
    $Output = Resolve-RepositoryPath $Run.output
    $ResumeThisSeed = $false
    if ($Resume -and (Test-Path -LiteralPath $Output -PathType Leaf)) {
        $Saved = Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
        if ($Saved.complete -eq $true) {
            Write-Host "Skipping completed seed ${Seed}: $Output"
            continue
        }
        $ResumeThisSeed = $true
    }
    $Arguments = @(
        $Evaluator,
        '--dataset-path', $Dataset,
        '--metadata', $Metadata,
        '--split', 'test',
        '--left-model', 'structured',
        '--left-config', $CConfig,
        '--left-checkpoint', (Resolve-RepositoryPath $Run.C_checkpoint),
        '--left-label', 'C',
        '--right-model', 'structured',
        '--right-config', $DConfig,
        '--right-checkpoint', (Resolve-RepositoryPath $Run.D_checkpoint),
        '--right-label', 'D',
        '--output', $Output,
        '--device', $Manifest.evaluation.device
    )
    if ($ResumeThisSeed) {
        $Arguments += '--resume'
    }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Test evaluation failed for seed $Seed"
    }
}

$Aggregator = Resolve-RepositoryPath $Manifest.evaluation.aggregator
$AggregateArguments = @($Aggregator)
foreach ($Seed in $Seeds) {
    $AggregateArguments += Resolve-RepositoryPath $Manifest.seeds.$Seed.output
}
$AggregateArguments += @(
    '--output', (Resolve-RepositoryPath $Manifest.aggregate_output)
)
& $Python @AggregateArguments
if ($LASTEXITCODE -ne 0) {
    throw 'Test aggregation failed'
}

Write-Host "Completed frozen Phase 1 C-versus-D MAESTRO test analysis at commit $Commit"
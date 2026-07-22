param(
    [switch]$SkipTests
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$mavenArgs = @("clean", "verify", "-Pnative")

if ($SkipTests) {
    $mavenArgs += "-DskipTests"
}

Push-Location $projectRoot
try {
    & mvn @mavenArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

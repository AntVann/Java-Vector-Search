$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $projectRoot
try {
    & mvn test
    exit $LASTEXITCODE
} finally {
    Pop-Location
}


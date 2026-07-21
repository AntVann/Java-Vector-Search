param(
    [ValidateSet("cpu", "cuda", "cuvs")]
    [string]$Backend = "cpu",
    [int]$Vectors = 10000,
    [int]$Dimensions = 128,
    [int]$Queries = 10,
    [int]$K = 10,
    [ValidateSet("euclidean", "cosine", "dot_product")]
    [string]$Metric = "euclidean"
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$jarPath = Join-Path $projectRoot "vectorforge-demo\target\vectorforge-demo.jar"

Push-Location $projectRoot
try {
    & mvn -pl vectorforge-demo -am package -DskipTests
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & java -jar $jarPath --backend $Backend --vectors $Vectors --dimensions $Dimensions --queries $Queries --k $K --metric $Metric
    exit $LASTEXITCODE
} finally {
    Pop-Location
}


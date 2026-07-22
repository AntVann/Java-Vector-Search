param(
    [ValidateSet("cpu", "native", "cuda", "cuvs")]
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
$nativeLibraryDir = Join-Path $projectRoot "vectorforge-native\target\native-lib"

Push-Location $projectRoot
try {
    $mavenArgs = @("-pl", "vectorforge-demo", "-am", "package", "-DskipTests")
    if ($Backend -eq "native") {
        $mavenArgs += "-Pnative"
    } elseif ($Backend -eq "cuda") {
        $mavenArgs += "-Pcuda"
    }

    & mvn @mavenArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $javaArgs = @("-jar", $jarPath, "--backend", $Backend, "--vectors", $Vectors, "--dimensions", $Dimensions, "--queries", $Queries, "--k", $K, "--metric", $Metric)
    if ($Backend -eq "native" -or $Backend -eq "cuda") {
        & java "-Dvectorforge.native.library.dir=$nativeLibraryDir" @javaArgs
    } else {
        & java @javaArgs
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

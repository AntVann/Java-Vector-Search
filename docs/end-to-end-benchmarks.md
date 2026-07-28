# End-to-End Benchmark Runs

`EndToEndBenchmarkRunner` profiles complete Java batch-search calls across the Java CPU, native C++ CPU, custom CUDA, and cuVS backends. It does not replace JMH.

## Output

The runner writes JSON Lines:

- one `metadata` record describing the OS, JVM, CPU, GPU, driver, CUDA, compiler, cuVS, Maven, CMake, Git SHA, and dirty-worktree state where detectable
- one `result` record per executed backend/scenario
- one `skip` record when a backend, metric, or dataset is unavailable
- one `error` record followed by a nonzero process exit when a detected backend fails during execution

Result records include:

- separate index-build time
- raw end-to-end batch samples
- average, p50, p95, and p99 batch latency
- queries per second
- Recall@k against the exact CPU result IDs
- approximate Java heap, process RSS, and GPU-memory deltas where available
- custom CUDA H2D, kernel, and D2H averages where exposed

Memory snapshots are observational. JVM allocation, native allocator caching, other GPU processes, and `nvidia-smi` sampling can affect them.

## Build

For a CPU-only report:

```bash
mvn -pl vectorforge-benchmarks -am package
```

For all GPU backends, first build the optional cuVS profile:

```bash
conda activate cuvs
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CUDAToolkit_ROOT="$CONDA_PREFIX/targets/x86_64-linux"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mvn clean verify -Pcuvs
```

## Local Small Run

```bash
mkdir -p benchmark-results
java -Dvectorforge.native.library.dir=vectorforge-native/target/native-lib \
  -cp vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  com.vectorforge.benchmarks.EndToEndBenchmarkRunner \
  --mode small --output benchmark-results/small.jsonl

python scripts/benchmark-jsonl-to-markdown.py \
  benchmark-results/small.jsonl \
  --output benchmark-results/small.md
```

The `small` preset covers 10,000 vectors, dimensions 128 and 384, batches 1/8/32, `k` 1/10, and all VectorForge metrics. Unsupported custom-CUDA metrics produce skip records.

## GPU Large Run

```bash
java -Xmx12g \
  -Dvectorforge.native.library.dir=vectorforge-native/target/native-lib \
  -cp vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  com.vectorforge.benchmarks.EndToEndBenchmarkRunner \
  --mode large --max-dataset-bytes 4294967296 \
  --output benchmark-results/large.jsonl
```

The `large` preset defines 100,000 and 1,000,000 vectors, dimensions 384/768, batches 32/128, and `k` 10/100. Run only configurations appropriate for available host RAM, GPU memory, and time. Use comma-separated overrides to narrow a run:

```bash
--vectors 100000 --dimensions 384 --batches 32 --k 10 \
--metrics DOT_PRODUCT --backends cpu,native,cuda,cuvs
```

The `matrix` preset exposes the full requested matrix: vector counts 10k/100k/1m, dimensions 128/384/768, batches 1/8/32/128, and `k` 1/10/100. It is intentionally not the default because the Cartesian product is expensive.

Configuration is strict: unknown or duplicate options, modes, backends, or metrics; duplicate/empty list values; non-positive sizes; invalid warm-up/iteration counts; and `k` values larger than a configured vector count fail before allocation. Checked size arithmetic, `--max-dataset-bytes`, and a conservative Java-heap estimate protect CPU ground-truth allocation. Budget and capability exclusions are skips; detected-backend execution failures are errors and make the process exit nonzero.

Executable validation checks:

```bash
bash scripts/check-end-to-end-benchmark.sh
```

## Reproducibility and Interpretation

- Generated vectors, queries, and IDs use a recorded deterministic seed.
- Each backend receives identical inputs for a scenario.
- Recall is based on result-ID set overlap with CPU exact brute force.
- Build time is never included in query latency.
- The runner executes backends sequentially in one JVM; use multiple runs and retain raw artifacts.
- With the smoke preset's three measurements, p95 and p99 are only order statistics from a tiny sample; they are not stable tail-latency estimates.
- Latency percentiles describe complete batch operations, not individual query latency.
- GPU memory is an observational process-external `nvidia-smi` snapshot; allocator caching and unrelated GPU activity can change it.
- Do not infer general backend superiority from a single machine, preset, or power/thermal state.
- `--max-dataset-bytes` produces explicit skip records instead of risking an oversized allocation.

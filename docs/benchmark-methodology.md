# Benchmark Methodology

## Status

This document defines the benchmark rules VectorForge follows and records
machine-specific results. The repository contains JMH benchmarks, a dedicated
CUDA profiling harness, a resident-backend end-to-end runner, and a separate
disk-IVF harness.

The harnesses are not interchangeable: disk-IVF includes partition selection,
I/O, candidate loading, and reranking, while resident-backend measurements do
not. Smoke results are functional evidence, not generalized performance claims.

## Verified Environment

| Attribute | Value |
| --- | --- |
| Date | July 22, 2026 |
| OS | Windows 11 `10.0` `amd64` |
| CPU | `AMD Ryzen 9 3900X 12-Core Processor` |
| Logical processors | `24` |
| GPU | `NVIDIA GeForce RTX 3070` |
| GPU driver | `610.74` |
| CUDA UMD | `13.3` |
| Java | `OpenJDK 21.0.7 Temurin 21.0.7+6-LTS` |
| Maven | `Apache Maven 3.9.9` |

## Available Benchmark Harnesses

- `CpuBruteForceSearchBenchmark.searchTopK`
- `CudaBackendProfileRunner`
  - profiles both the high-level `CudaBruteForceIndex` path and the raw `NativeBindings.searchCuda` path
  - reports end-to-end latency plus internal CUDA phase timings
  - distinguishes single-query and batched-query behavior
- `BackendComparisonRunner`
  - uses the same generated vectors, queries, IDs, metric, and `k` for CPU Java, custom CUDA, and cuVS
  - reports build time, raw batch-search samples, p50/p95/p99, average, and whether result IDs match the CPU baseline
  - is a smoke/profile harness rather than JMH

## JVM Warm-Up

- Use JMH warm-up iterations before recording measurements
- Separate build/setup cost from query measurements
- Avoid mixing classloading and JIT compilation effects into steady-state search numbers

## Dataset Generation

- Use fixed random seeds
- Record dataset size, dimensions, batch size, metric, and `k`
- Keep generated inputs identical across backends for fair comparison

## Latency Measurement

- Report p50, p95, and p99 for end-to-end query latency
- Distinguish single-query and batched-query behavior
- Include tail latency because GPU batching often improves throughput while hurting single-query latency

## GPU Synchronization

- Synchronize explicitly around timed regions
- Measure host-to-device transfer, kernel execution, and device-to-host transfer separately
- Report total end-to-end latency in addition to kernel-only timing

## Recall Calculation

- Use the CPU brute-force backend as the correctness baseline
- Compute Recall@k for any approximate backend
- Never compare approximate and exact backends using throughput alone

## Fair CPU/GPU Comparison

- Keep index build time separate from search time
- Keep the GPU index resident between queries when that is how the backend is designed to run
- Do not force the CPU backend to include unrelated setup work that the GPU backend excludes
- Prefer isolated JVMs or rotate backend execution order for serious comparisons;
  current smoke runners use a fixed order and can be affected by JIT, cache,
  power, and thermal state
- Exact-backend validation should compare ordered IDs and score tolerances in
  addition to set-based Recall@k

## Verified CPU JMH Benchmark

Command:

```powershell
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar com.vectorforge.benchmarks.CpuBruteForceSearchBenchmark.searchTopK -wi 3 -i 5 -f 1
```

Configuration:

| Parameter | Value |
| --- | --- |
| Benchmark | `CpuBruteForceSearchBenchmark.searchTopK` |
| `dimensions` | `128` |
| `k` | `10` |
| `vectorCount` | `10000` |
| Warmup | `3 x 10s` |
| Measurement | `5 x 10s` |
| Forks | `1` |
| Threads | `1` |

Observed result:

| Metric | Value |
| --- | --- |
| Score | `930.850 +- 10.293 us/op` |
| Min | `927.279 us/op` |
| Avg | `930.850 us/op` |
| Max | `933.774 us/op` |

## Verified CUDA Profile Harness

Command:

```powershell
java -Dvectorforge.native.library.dir=vectorforge-native/target/native-lib -cp vectorforge-benchmarks/target/vectorforge-benchmarks.jar com.vectorforge.benchmarks.CudaBackendProfileRunner --vectors 10000 --dimensions 128 --k 10 --warmup 30 --iterations 120 --small-queries 1 --batch-queries 32
java -Dvectorforge.native.library.dir=vectorforge-native/target/native-lib -cp vectorforge-benchmarks/target/vectorforge-benchmarks.jar com.vectorforge.benchmarks.CudaBackendProfileRunner --vectors 50000 --dimensions 384 --k 10 --warmup 10 --iterations 20 --small-queries 1 --batch-queries 16
```

### Baseline Before Optimization

This section records implementation-phase measurements from the earlier row-major kernel. These numbers are historical and were not re-run in the current review pass.

The pre-optimization CUDA backend used:

- row-major device vectors
- a one-dimensional kernel with one thread per score
- no shared-memory query staging

Measured baseline results:

| Dataset | Path | Scenario | End-to-end avg | Kernel avg | H2D avg | D2H avg | Native total avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `10000 x 128` | `high_level` | `single` | `216.025 us` | `0.025 ms` | `0.003 ms` | `0.041 ms` | `0.174 ms` |
| `10000 x 128` | `high_level` | `batch x32` | `1954.015 us` | `0.975 ms` | `0.023 ms` | `0.239 ms` | `1.828 ms` |
| `10000 x 128` | `raw_native` | `single` | `180.603 us` | `0.025 ms` | `0.004 ms` | `0.046 ms` | `0.176 ms` |
| `10000 x 128` | `raw_native` | `batch x32` | `1601.630 us` | `0.840 ms` | `0.021 ms` | `0.218 ms` | `1.594 ms` |
| `50000 x 384` | `high_level` | `single` | `1381.225 us` | `0.986 ms` | `0.006 ms` | `0.073 ms` | `1.303 ms` |
| `50000 x 384` | `high_level` | `batch x16` | `34984.130 us` | `33.073 ms` | `0.035 ms` | `0.613 ms` | `34.773 ms` |
| `50000 x 384` | `raw_native` | `single` | `1243.060 us` | `0.931 ms` | `0.005 ms` | `0.081 ms` | `1.238 ms` |
| `50000 x 384` | `raw_native` | `batch x16` | `35761.955 us` | `33.845 ms` | `0.044 ms` | `0.671 ms` | `35.732 ms` |

### Final Results After Optimization

The kept optimization transposes vectors at build time and uses a two-dimensional tiled kernel that stages query data in shared memory.

Measured final results:

| Dataset | Path | Scenario | End-to-end avg | Kernel avg | H2D avg | D2H avg | Native total avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `10000 x 128` | `high_level` | `single` | `191.667 us` | `0.018 ms` | `0.004 ms` | `0.040 ms` | `0.153 ms` |
| `10000 x 128` | `high_level` | `batch x32` | `1050.008 us` | `0.109 ms` | `0.026 ms` | `0.267 ms` | `0.958 ms` |
| `10000 x 128` | `raw_native` | `single` | `186.880 us` | `0.019 ms` | `0.004 ms` | `0.055 ms` | `0.183 ms` |
| `10000 x 128` | `raw_native` | `batch x32` | `2020.038 us` | `0.098 ms` | `0.020 ms` | `0.243 ms` | `2.013 ms` |
| `50000 x 384` | `high_level` | `single` | `504.095 us` | `0.202 ms` | `0.005 ms` | `0.073 ms` | `0.438 ms` |
| `50000 x 384` | `high_level` | `batch x16` | `6516.940 us` | `4.788 ms` | `0.026 ms` | `0.535 ms` | `6.343 ms` |
| `50000 x 384` | `raw_native` | `single` | `582.675 us` | `0.189 ms` | `0.004 ms` | `0.076 ms` | `0.579 ms` |
| `50000 x 384` | `raw_native` | `batch x16` | `6075.985 us` | `4.622 ms` | `0.026 ms` | `0.521 ms` | `6.058 ms` |

Interpretation note:

- The optimized kernel improvement is still clear in absolute kernel and end-to-end times on the current checkout.
- The `high_level` versus `raw_native` gap remains noisy on smaller scenarios, so the derived Java-overhead estimate should be treated as directional rather than stable.

### Rejected Java-Side Optimization

A Java-side direct-buffer reuse experiment was benchmarked against the optimized native backend and then reverted because it did not produce a stable improvement:

| Reuse Mode | Scenario | End-to-end avg |
| --- | --- | --- |
| `disabled` | `10000 x 128 single` | `196.828 us` |
| `enabled` | `10000 x 128 single` | `234.013 us` |
| `disabled` | `10000 x 128 batch x32` | `842.875 us` |
| `enabled` | `10000 x 128 batch x32` | `956.236 us` |

## Verified End-to-End Demo Runs

Commands:

```powershell
java -jar vectorforge-demo/target/vectorforge-demo.jar --backend cpu --vectors 100000 --dimensions 384 --queries 100 --k 10 --metric dot_product
java -Dvectorforge.native.library.dir=vectorforge-native/target/native-lib -jar vectorforge-demo/target/vectorforge-demo.jar --backend native --vectors 100000 --dimensions 384 --queries 100 --k 10 --metric dot_product
java -Dvectorforge.native.library.dir=vectorforge-native/target/native-lib -jar vectorforge-demo/target/vectorforge-demo.jar --backend cuda --vectors 100000 --dimensions 384 --queries 100 --k 10 --metric dot_product
```

Observed summary:

| Backend | `build_ms` | `search_ms` | `avg_query_us` |
| --- | --- | --- | --- |
| `cpu` | `88.015` | `3043.580` | `30435.804` |
| `native` | `154.011` | `2930.033` | `29300.326` |
| `cuda` | `477.857` | `101.818` | `1018.183` |

Observed CUDA phase timings:

| Metric | Value |
| --- | --- |
| `cuda_h2d_ms` | `0.052` |
| `cuda_kernel_ms` | `76.459` |
| `cuda_d2h_ms` | `6.267` |
| `cuda_total_ms` | `98.155` |

These demo numbers are not a substitute for JMH. They are included as verified end-to-end examples that exercise the current CPU, JNI, and CUDA paths under the same dataset shape.

## CPU, Custom CUDA, and cuVS Comparison

Build the cuVS-enabled native library and benchmark jar first, then run:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
java -Dvectorforge.native.library.dir=vectorforge-native/target/native-lib \
  -cp vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  com.vectorforge.benchmarks.BackendComparisonRunner \
  --vectors 10000 --dimensions 128 --queries 16 --k 10 --warmup 5 --iterations 10
```

The runner intentionally uses dot product because it is the common metric supported by all three compared backends. Its wall-clock measurements include Java packing, JNI, transfers, synchronization, native result conversion, and Java result materialization. Build time is reported separately. Raw samples should be retained when recording results.

Results from one laptop run belong here only as machine-specific smoke evidence. They must not be interpreted as a performance-superiority claim: the harness has no JMH forks, a short warm-up, sequential backend execution, and different backend implementation strategies.

### Laptop Smoke Results

Recorded July 27, 2026 in Ubuntu under WSL2 using OpenJDK 21, cuVS 26.06, CUDA 12.9, and an NVIDIA GeForce RTX 3070. The native library came from the same clean `-Pcuvs` reactor run whose CPU, native, custom-CUDA, and cuVS tests passed.

Exact command:

```bash
export LD_LIBRARY_PATH=/home/antho/miniforge3/envs/cuvs/lib
/home/antho/miniforge3/envs/cuvs/bin/java \
  -Dvectorforge.native.library.dir=/mnt/c/Users/antho/Desktop/Java-Vector-Search/vectorforge-native/target/native-lib \
  -cp /mnt/c/Users/antho/Desktop/Java-Vector-Search/vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  com.vectorforge.benchmarks.BackendComparisonRunner \
  --vectors 10000 --dimensions 128 --queries 16 --k 10 --warmup 5 --iterations 10
```

Observed summary:

| Backend | Build | CPU result IDs matched | Batch average | p50 | p95 | p99 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CPU Java | `8.235 ms` | `true` | `20894.565 us` | `19809.089 us` | `23939.029 us` | `23939.029 us` |
| Custom CUDA | `330.708 ms` | `true` | `1212.033 us` | `1042.474 us` | `2710.555 us` | `2710.555 us` |
| cuVS | `170.074 ms` | `true` | `486.459 us` | `459.480 us` | `857.841 us` | `857.841 us` |

Raw end-to-end batch samples in execution order:

```text
cpu_us=[19809.089,19009.228,23939.029,23347.119,19597.634,23875.297,20434.819,20492.942,19359.114,19081.378]
cuda_us=[937.404,1229.561,1063.912,970.462,1119.834,1118.412,933.807,993.907,1042.474,2710.555]
cuvs_us=[482.931,461.049,475.937,437.262,352.140,315.813,643.897,857.841,459.480,378.235]
```

These samples cover complete 16-query batch calls, not individual queries. All compared backends returned the same ordered result IDs as the CPU reference for this seeded input. Score agreement is covered separately by correctness tests.

This is smoke evidence only. Ten measurements after five warm-up calls are insufficient for generalized performance conclusions. The backends ran sequentially in one JVM, build implementations differ, WSL and laptop power/thermal state can affect results, and this harness does not use JMH forks or confidence intervals.

The cuVS demo path was also smoke-tested with:

```bash
/home/antho/miniforge3/envs/cuvs/bin/java \
  -Dvectorforge.native.library.dir=/mnt/c/Users/antho/Desktop/Java-Vector-Search/vectorforge-native/target/native-lib \
  -jar /mnt/c/Users/antho/Desktop/Java-Vector-Search/vectorforge-demo/target/vectorforge-demo.jar \
  --backend cuvs --vectors 1000 --dimensions 32 --queries 2 --k 5 --metric dot_product
```

It completed successfully and returned five results per query. Its observed `build_ms=266.138` and `search_ms=94.328` include first-use startup effects and are recorded only as functional smoke output.

## Why Kernel-Only Timing Is Insufficient

Kernel execution alone hides real deployment costs:

- PCIe or NVLink transfer overhead
- Host-side batching and marshaling
- Synchronization and result materialization
- JVM/native boundary costs

For that reason, VectorForge will always report both component timings and total observed query time.

# Benchmark Methodology

## Status

This document defines the benchmark rules VectorForge follows and records the currently verified CPU, native, and CUDA results from the latest review pass. The repository contains one JMH benchmark today, for the CPU backend. Native and CUDA timings below come from verified demo executions because there is not yet a dedicated JMH harness for those backends.

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
- No dedicated JMH benchmark exists yet for the JNI or CUDA backends

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

When GPU benchmarks are added:

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
| Score | `953.574 +- 32.782 us/op` |
| Min | `943.592 us/op` |
| Avg | `953.574 us/op` |
| Max | `966.763 us/op` |

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
| `cpu` | `86.855` | `2963.671` | `29636.713` |
| `native` | `119.914` | `2793.152` | `27931.515` |
| `cuda` | `305.865` | `438.178` | `4381.775` |

Observed CUDA phase timings:

| Metric | Value |
| --- | --- |
| `cuda_h2d_ms` | `0.042` |
| `cuda_kernel_ms` | `414.848` |
| `cuda_d2h_ms` | `5.493` |
| `cuda_total_ms` | `434.427` |

These demo numbers are not a substitute for JMH. They are included as verified end-to-end examples that exercise the current CPU, JNI, and CUDA paths under the same dataset shape.

## Skipped Validation

- cuVS validation was skipped because the `cuvs` backend is not implemented

## Why Kernel-Only Timing Is Insufficient

Kernel execution alone hides real deployment costs:

- PCIe or NVLink transfer overhead
- Host-side batching and marshaling
- Synchronization and result materialization
- JVM/native boundary costs

For that reason, VectorForge will always report both component timings and total observed query time.

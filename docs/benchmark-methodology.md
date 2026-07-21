# Benchmark Methodology

## Status

This document defines the benchmark rules VectorForge follows and records the currently verified CPU baseline. All results in this file were collected on July 21, 2026. No GPU or cuVS numbers are claimed because those backends are not implemented in the current repository state.

## Verified Environment

| Attribute | Value |
| --- | --- |
| Date | July 21, 2026 |
| OS | Windows 11 `10.0` `amd64` |
| RuntimeInformation.OSDescription | `Microsoft Windows 10.0.26200` |
| CPU | `AMD Ryzen 9 3900X 12-Core Processor` |
| Logical processors | `24` |
| JVM MaxRAM | `137438953472` bytes (`~128 GiB`) |
| GPU present on host | `NVIDIA GeForce RTX 3070` |
| GPU driver | `610.74` |
| CUDA UMD | `13.3` |
| GPU VRAM | `8192 MiB` |
| Java | `OpenJDK 21.0.7 Temurin 21.0.7+6-LTS` |
| Maven | `Apache Maven 3.9.9` |

The installed GPU hardware is recorded for completeness only. It was not exercised by this repository because the `cuda` and `cuvs` backends are not available yet.

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

## Verified CPU Benchmark

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
| Score | `894.760 +- 36.378 us/op` |
| Min | `881.203 us/op` |
| Avg | `894.760 us/op` |
| Max | `904.962 us/op` |

This measurement isolates steady-state `searchTopK` cost. Build/setup work is intentionally excluded from the benchmarked method.

## Verified Demo Run

Command:

```powershell
java -jar vectorforge-demo/target/vectorforge-demo.jar --backend cpu --vectors 100000 --dimensions 384 --queries 100 --k 10 --metric cosine
```

Observed output summary:

| Metric | Value |
| --- | --- |
| `build_ms` | `83.728` |
| `search_ms` | `2908.539` |
| `avg_query_us` | `29085.390` |

This run is not a substitute for JMH. It is included as a reproducible end-to-end example that combines index build and search in the demo CLI.

## Skipped Validation

- GPU backend benchmarks were skipped because the `cuda` backend is not implemented.
- cuVS validation was skipped because the `cuvs` backend is not implemented.

## Why Kernel-Only Timing Is Insufficient

Kernel execution alone hides real deployment costs:

- PCIe or NVLink transfer overhead
- Host-side batching and marshaling
- Synchronization and result materialization
- JVM/native boundary costs

For that reason, VectorForge will always report both component timings and total observed query time.

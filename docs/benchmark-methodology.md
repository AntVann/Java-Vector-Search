# Benchmark Methodology

## Status

This document defines the benchmark rules the repository will follow. Milestone 1 provides only a CPU benchmark scaffold and does not claim any measured results.

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

## Why Kernel-Only Timing Is Insufficient

Kernel execution alone hides real deployment costs:

- PCIe or NVLink transfer overhead
- Host-side batching and marshaling
- Synchronization and result materialization
- JVM/native boundary costs

For that reason, VectorForge will always report both component timings and total observed query time.


# VectorForge End-to-End Benchmark

Generated from machine-readable JSON Lines. These measurements are local observations, not generalized performance claims.

## System

| Attribute | Value |
| --- | --- |
| `timestamp_utc` | 2026-08-03T22:27:14.900257879Z |
| `mode` | smoke |
| `os_name` | Linux |
| `os_version` | 5.15.153.1-microsoft-standard-WSL2 |
| `os_arch` | amd64 |
| `jvm` | OpenJDK 64-Bit Server VM |
| `java_version` | 21.0.10-internal |
| `processors` | 16 |
| `cpu` | 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz |
| `max_heap_bytes` | 8589934592 |
| `compiler` | /home/antho/miniforge3/envs/cuvs/bin/x86_64-conda-linux-gnu-c++ |
| `compiler_version` | x86_64-conda-linux-gnu-c++ (conda-forge gcc 14.3.0-20) 14.3.0<br>Copyright (C) 2024 Free Software Foundation, Inc.<br>This is free software; see the source for copying conditions.  There is NO<br>warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. |
| `cuda_toolkit_root` | n/a |
| `cuda_version` | nvcc: NVIDIA (R) Cuda compiler driver<br>Copyright (c) 2005-2025 NVIDIA Corporation<br>Built on Tue_May_27_02:21:03_PDT_2025<br>Cuda compilation tools, release 12.9, V12.9.86<br>Build cuda_12.9.r12.9/compiler.36037853_0 |
| `cuda_device_count` | 1 |
| `cuvs_version` | 26.6.0 |
| `gpu` | n/a |
| `git_sha` | b3fe66e3828f37fbc141b9cd606da37ac1301ac0 |
| `git_dirty` | False |
| `maven_version` | Apache Maven 3.9.16 (50e2c7eb1a9cd50ab041c0f7591d943eeb409f68) |
| `cmake_version` | cmake version 4.4.0 |

## Results

| Backend | Vectors | Dims | Batch | k | Metric | Build ms | Batch avg ms | p50 ms | p95 ms | p99 ms | QPS | Recall@k | Heap delta | RSS delta | GPU delta | CUDA H2D / kernel / D2H ms |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cpu | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 6.922 | 1.105 | 1.031 | 1.371 | 1.402 | 904.880 | 1.000000 | 10029720 | 10727424 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 17.403 | 0.841 | 0.817 | 1.056 | 1.097 | 1188.720 | 1.000000 | 1546520 | 10641408 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 1004.236 | 0.267 | 0.228 | 0.413 | 0.454 | 3745.840 | 1.000000 | 1641136 | 114798592 | n/a | 0.012 / 0.042 / 0.077 |
| cuvs | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 126.380 | 0.436 | 0.410 | 0.582 | 0.695 | 2292.468 | 1.000000 | 2013376 | 140259328 | n/a | n/a / n/a / n/a |

## Caveats

- Build time is separate from measured query batches.
- QPS is derived from completed queries divided by summed end-to-end batch time.
- Heap, process RSS, and `nvidia-smi` deltas are approximate snapshots and may be unavailable.
- CUDA phase timings are emitted only by the custom CUDA backend.
- p95/p99 are order statistics from the configured sample count.
- Review the JSON Lines artifact for raw samples and explicit skip records.

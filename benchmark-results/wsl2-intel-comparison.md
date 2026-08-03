# VectorForge End-to-End Benchmark

Generated from machine-readable JSON Lines. These measurements are local observations, not generalized performance claims.

## System

| Attribute | Value |
| --- | --- |
| `timestamp_utc` | 2026-08-03T21:48:30.771513385Z |
| `mode` | smoke |
| `os_name` | Linux |
| `os_version` | 5.15.153.1-microsoft-standard-WSL2 |
| `os_arch` | amd64 |
| `jvm` | OpenJDK 64-Bit Server VM |
| `java_version` | 21.0.10-internal |
| `processors` | 16 |
| `cpu` | 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz |
| `max_heap_bytes` | 8371830784 |
| `compiler` | /home/antho/miniforge3/envs/cuvs/bin/x86_64-conda-linux-gnu-c++ |
| `compiler_version` | x86_64-conda-linux-gnu-c++ (conda-forge gcc 14.3.0-20) 14.3.0<br>Copyright (C) 2024 Free Software Foundation, Inc.<br>This is free software; see the source for copying conditions.  There is NO<br>warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. |
| `cuda_toolkit_root` | /home/antho/miniforge3/envs/cuvs/targets/x86_64-linux |
| `cuda_version` | nvcc: NVIDIA (R) Cuda compiler driver<br>Copyright (c) 2005-2025 NVIDIA Corporation<br>Built on Tue_May_27_02:21:03_PDT_2025<br>Cuda compilation tools, release 12.9, V12.9.86<br>Build cuda_12.9.r12.9/compiler.36037853_0 |
| `cuda_device_count` | 1 |
| `cuvs_version` | 26.6.0 |
| `gpu` | n/a |
| `git_sha` | 4df4f3e47a0f5dd9dfbaa230ba175ef1dc9342ce |
| `git_dirty` | True |
| `maven_version` | Apache Maven 3.9.16 (50e2c7eb1a9cd50ab041c0f7591d943eeb409f68) |
| `cmake_version` | cmake version 4.4.0 |

## Results

| Backend | Vectors | Dims | Batch | k | Metric | Build ms | Batch avg ms | p50 ms | p95 ms | p99 ms | QPS | Recall@k | Heap delta | RSS delta | GPU delta | CUDA H2D / kernel / D2H ms |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cpu | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 7.902 | 1.135 | 1.095 | 1.390 | 1.419 | 880.733 | 1.000000 | 10184272 | 10670080 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 19.660 | 0.853 | 0.832 | 0.976 | 1.074 | 1171.794 | 1.000000 | 1466488 | 10788864 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 962.793 | 0.161 | 0.141 | 0.211 | 0.326 | 6199.386 | 1.000000 | 1641128 | 113844224 | n/a | 0.006 / 0.035 / 0.037 |
| cuvs | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 111.005 | 0.324 | 0.306 | 0.455 | 0.471 | 3087.717 | 1.000000 | 2013400 | 140177408 | n/a | n/a / n/a / n/a |

## Caveats

- Build time is separate from measured query batches.
- QPS is derived from completed queries divided by summed end-to-end batch time.
- Heap, process RSS, and `nvidia-smi` deltas are approximate snapshots and may be unavailable.
- CUDA phase timings are emitted only by the custom CUDA backend.
- p95/p99 are order statistics from the configured sample count.
- Review the JSON Lines artifact for raw samples and explicit skip records.

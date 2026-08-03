# VectorForge End-to-End Benchmark

Generated from machine-readable JSON Lines. These measurements are local observations, not generalized performance claims.

## System

| Attribute | Value |
| --- | --- |
| `timestamp_utc` | 2026-08-03T21:58:01.460198628Z |
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
| `git_sha` | d974aaae94fe8982fe14d055998cfd806eb6d1f6 |
| `git_dirty` | True |
| `maven_version` | Apache Maven 3.9.16 (50e2c7eb1a9cd50ab041c0f7591d943eeb409f68) |
| `cmake_version` | cmake version 4.4.0 |

## Results

| Backend | Vectors | Dims | Batch | k | Metric | Build ms | Batch avg ms | p50 ms | p95 ms | p99 ms | QPS | Recall@k | Heap delta | RSS delta | GPU delta | CUDA H2D / kernel / D2H ms |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cpu | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 6.431 | 1.165 | 1.082 | 1.436 | 1.615 | 858.010 | 1.000000 | 10194832 | 10645504 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 18.220 | 0.819 | 0.804 | 0.925 | 0.925 | 1220.799 | 1.000000 | 1466496 | 12742656 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 1002.783 | 0.171 | 0.147 | 0.275 | 0.329 | 5861.553 | 1.000000 | 1641128 | 114860032 | n/a | 0.006 / 0.035 / 0.040 |
| cuvs | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 137.309 | 0.405 | 0.372 | 0.523 | 0.589 | 2471.400 | 1.000000 | 2013376 | 140283904 | n/a | n/a / n/a / n/a |
| cpu | 10000 | 128 | 8 | 10 | DOT_PRODUCT | 3.797 | 6.480 | 6.413 | 7.050 | 7.373 | 1234.476 | 1.000000 | 9858064 | 876544 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 8 | 10 | DOT_PRODUCT | 4.190 | 6.192 | 6.122 | 6.488 | 7.019 | 1292.024 | 1.000000 | 1381584 | 4096 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 8 | 10 | DOT_PRODUCT | 93.470 | 0.499 | 0.476 | 0.563 | 0.598 | 16024.659 | 1.000000 | 1685992 | 89927680 | n/a | 0.009 / 0.223 / 0.105 |
| cuvs | 10000 | 128 | 8 | 10 | DOT_PRODUCT | 7.985 | 0.333 | 0.332 | 0.386 | 0.394 | 24014.425 | 1.000000 | 2088880 | 5533696 | n/a | n/a / n/a / n/a |
| cpu | 10000 | 128 | 32 | 10 | DOT_PRODUCT | 4.512 | 25.908 | 25.494 | 28.157 | 29.129 | 1235.136 | 1.000000 | 9512664 | 8683520 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 32 | 10 | DOT_PRODUCT | 6.608 | 23.678 | 23.117 | 25.837 | 25.982 | 1351.454 | 1.000000 | 1258360 | 5062656 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 32 | 10 | DOT_PRODUCT | 72.653 | 1.653 | 1.643 | 1.760 | 1.784 | 19357.226 | 1.000000 | 1255528 | 90288128 | n/a | 0.016 / 0.860 / 0.286 |
| cuvs | 10000 | 128 | 32 | 10 | DOT_PRODUCT | 7.681 | 0.513 | 0.503 | 0.701 | 0.775 | 62383.166 | 1.000000 | 2097048 | 0 | n/a | n/a / n/a / n/a |
| cpu | 10000 | 128 | 128 | 10 | DOT_PRODUCT | 8.953 | 103.456 | 102.457 | 112.772 | 115.821 | 1237.246 | 1.000000 | -62034432 | 10891264 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 128 | 10 | DOT_PRODUCT | 4.188 | 98.159 | 97.619 | 101.736 | 103.287 | 1304.004 | 1.000000 | 1425896 | 0 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 128 | 10 | DOT_PRODUCT | 81.632 | 5.836 | 5.809 | 6.016 | 6.059 | 21934.401 | 1.000000 | 1510848 | 79888384 | n/a | 0.030 / 3.338 / 0.747 |
| cuvs | 10000 | 128 | 128 | 10 | DOT_PRODUCT | 5.719 | 1.201 | 1.181 | 1.318 | 1.336 | 106604.667 | 1.000000 | 1839808 | 0 | n/a | n/a / n/a / n/a |
| cpu | 100000 | 128 | 1 | 10 | DOT_PRODUCT | 42.400 | 8.637 | 8.391 | 9.668 | 9.677 | 115.778 | 1.000000 | 66437992 | 67137536 | n/a | n/a / n/a / n/a |
| native | 100000 | 128 | 1 | 10 | DOT_PRODUCT | 76.940 | 8.764 | 8.601 | 9.960 | 10.125 | 114.108 | 1.000000 | 14157376 | 118235136 | n/a | n/a / n/a / n/a |
| cuda | 100000 | 128 | 1 | 10 | DOT_PRODUCT | 155.110 | 0.605 | 0.562 | 0.734 | 1.036 | 1652.752 | 1.000000 | 13757376 | 144273408 | n/a | 0.009 / 0.276 / 0.135 |
| cuvs | 100000 | 128 | 1 | 10 | DOT_PRODUCT | 70.391 | 0.415 | 0.381 | 0.596 | 0.610 | 2409.128 | 1.000000 | 19584360 | 70078464 | n/a | n/a / n/a / n/a |
| cpu | 100000 | 128 | 8 | 10 | DOT_PRODUCT | 42.582 | 67.360 | 67.330 | 70.082 | 73.093 | 118.766 | 1.000000 | 66437992 | 67215360 | n/a | n/a / n/a / n/a |
| native | 100000 | 128 | 8 | 10 | DOT_PRODUCT | 63.803 | 64.405 | 64.128 | 67.042 | 68.052 | 124.214 | 1.000000 | 11137576 | 112889856 | n/a | n/a / n/a / n/a |
| cuda | 100000 | 128 | 8 | 10 | DOT_PRODUCT | 137.064 | 3.545 | 3.555 | 3.693 | 3.744 | 2256.663 | 1.000000 | 11537584 | 142557184 | n/a | 0.013 / 2.096 / 0.599 |
| cuvs | 100000 | 128 | 8 | 10 | DOT_PRODUCT | 51.754 | 0.985 | 0.974 | 1.051 | 1.231 | 8120.439 | 1.000000 | 12370136 | 64888832 | n/a | n/a / n/a / n/a |
| cpu | 100000 | 128 | 32 | 10 | DOT_PRODUCT | 19.653 | 272.413 | 269.048 | 285.036 | 296.445 | 117.469 | 1.000000 | 67108864 | 0 | n/a | n/a / n/a / n/a |
| native | 100000 | 128 | 32 | 10 | DOT_PRODUCT | 60.922 | 249.763 | 249.379 | 254.329 | 255.882 | 128.121 | 1.000000 | 11285776 | 102408192 | n/a | n/a / n/a / n/a |
| cuda | 100000 | 128 | 32 | 10 | DOT_PRODUCT | 141.769 | 12.957 | 12.867 | 13.519 | 13.894 | 2469.650 | 1.000000 | 11782896 | 130203648 | n/a | 0.022 / 8.150 / 1.684 |
| cuvs | 100000 | 128 | 32 | 10 | DOT_PRODUCT | 45.659 | 1.378 | 1.358 | 1.432 | 1.523 | 23218.970 | 1.000000 | 10485760 | 51204096 | n/a | n/a / n/a / n/a |
| cpu | 100000 | 128 | 128 | 10 | DOT_PRODUCT | 16.312 | 1089.370 | 1069.387 | 1157.661 | 1206.726 | 117.499 | 1.000000 | 67108864 | 28672 | n/a | n/a / n/a / n/a |
| native | 100000 | 128 | 128 | 10 | DOT_PRODUCT | 59.842 | 1012.005 | 1006.893 | 1061.360 | 1064.445 | 126.482 | 1.000000 | 11285776 | 102154240 | n/a | n/a / n/a / n/a |
| cuda | 100000 | 128 | 128 | 10 | DOT_PRODUCT | 140.211 | 50.123 | 49.852 | 51.417 | 51.554 | 2553.704 | 1.000000 | 10485760 | 128860160 | n/a | 0.041 / 32.129 / 5.934 |
| cuvs | 100000 | 128 | 128 | 10 | DOT_PRODUCT | 44.651 | 2.981 | 2.967 | 3.051 | 3.324 | 42942.756 | 1.000000 | 12582912 | 51204096 | n/a | n/a / n/a / n/a |

## Caveats

- Build time is separate from measured query batches.
- QPS is derived from completed queries divided by summed end-to-end batch time.
- Heap, process RSS, and `nvidia-smi` deltas are approximate snapshots and may be unavailable.
- CUDA phase timings are emitted only by the custom CUDA backend.
- p95/p99 are order statistics from the configured sample count.
- Review the JSON Lines artifact for raw samples and explicit skip records.

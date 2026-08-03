# VectorForge End-to-End Benchmark

Generated from machine-readable JSON Lines. These measurements are local observations, not generalized performance claims.

## System

| Attribute | Value |
| --- | --- |
| `timestamp_utc` | 2026-08-03T22:27:19.859792052Z |
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
| cpu | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 8.120 | 1.204 | 1.139 | 1.405 | 1.610 | 830.690 | 1.000000 | 9600864 | 10657792 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 18.643 | 0.816 | 0.796 | 0.946 | 0.956 | 1225.282 | 1.000000 | 1086712 | 12218368 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 972.551 | 0.171 | 0.156 | 0.217 | 0.261 | 5851.029 | 1.000000 | 1535200 | 113844224 | n/a | 0.006 / 0.038 / 0.042 |
| cuvs | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 107.643 | 0.478 | 0.468 | 0.583 | 0.696 | 2091.805 | 1.000000 | 2013392 | 140193792 | n/a | n/a / n/a / n/a |
| cpu | 10000 | 128 | 8 | 10 | DOT_PRODUCT | 3.147 | 7.023 | 6.606 | 9.066 | 11.049 | 1139.092 | 1.000000 | 10082792 | 647168 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 8 | 10 | DOT_PRODUCT | 4.999 | 6.118 | 6.099 | 6.557 | 6.778 | 1307.620 | 1.000000 | 1530136 | 4096 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 8 | 10 | DOT_PRODUCT | 83.025 | 0.471 | 0.454 | 0.555 | 0.555 | 16999.541 | 1.000000 | 1258360 | 89931776 | n/a | 0.007 / 0.225 / 0.082 |
| cuvs | 10000 | 128 | 8 | 10 | DOT_PRODUCT | 7.824 | 0.317 | 0.307 | 0.369 | 0.420 | 25198.299 | 1.000000 | 1677520 | 5533696 | n/a | n/a / n/a / n/a |
| cpu | 10000 | 128 | 32 | 10 | DOT_PRODUCT | 4.182 | 25.853 | 25.591 | 26.444 | 30.670 | 1237.780 | 1.000000 | 9853104 | 8552448 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 32 | 10 | DOT_PRODUCT | 7.819 | 24.198 | 24.011 | 25.374 | 25.489 | 1322.397 | 1.000000 | 1389456 | 7495680 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 32 | 10 | DOT_PRODUCT | 84.329 | 1.669 | 1.633 | 1.773 | 1.886 | 19170.614 | 1.000000 | 1204504 | 89706496 | n/a | 0.021 / 0.863 / 0.290 |
| cuvs | 10000 | 128 | 32 | 10 | DOT_PRODUCT | 7.873 | 0.434 | 0.394 | 0.493 | 1.018 | 73738.674 | 1.000000 | 2097312 | 10059776 | n/a | n/a / n/a / n/a |
| cpu | 10000 | 128 | 128 | 10 | DOT_PRODUCT | 2.189 | 102.587 | 101.889 | 106.961 | 107.739 | 1247.727 | 1.000000 | 9145864 | 225280 | n/a | n/a / n/a / n/a |
| native | 10000 | 128 | 128 | 10 | DOT_PRODUCT | 4.091 | 94.918 | 95.120 | 100.330 | 101.509 | 1348.529 | 1.000000 | 1385568 | 0 | n/a | n/a / n/a / n/a |
| cuda | 10000 | 128 | 128 | 10 | DOT_PRODUCT | 85.996 | 5.822 | 5.739 | 6.073 | 6.163 | 21986.959 | 1.000000 | 1174448 | 80154624 | n/a | 0.026 / 3.337 / 0.747 |
| cuvs | 10000 | 128 | 128 | 10 | DOT_PRODUCT | 5.691 | 1.255 | 1.243 | 1.460 | 1.498 | 102012.898 | 1.000000 | 1761720 | 0 | n/a | n/a / n/a / n/a |
| cpu | 100000 | 128 | 1 | 10 | DOT_PRODUCT | 37.072 | 8.696 | 8.606 | 9.263 | 10.202 | 114.989 | 1.000000 | 66521624 | 65118208 | n/a | n/a / n/a / n/a |
| native | 100000 | 128 | 1 | 10 | DOT_PRODUCT | 67.685 | 8.146 | 8.042 | 8.508 | 9.263 | 122.766 | 1.000000 | 14241016 | 118980608 | n/a | n/a / n/a / n/a |
| cuda | 100000 | 128 | 1 | 10 | DOT_PRODUCT | 124.843 | 0.530 | 0.511 | 0.593 | 0.610 | 1887.441 | 1.000000 | 13757360 | 143699968 | n/a | 0.009 / 0.256 / 0.112 |
| cuvs | 100000 | 128 | 1 | 10 | DOT_PRODUCT | 63.161 | 0.359 | 0.350 | 0.431 | 0.434 | 2787.869 | 1.000000 | 20171584 | 70078464 | n/a | n/a / n/a / n/a |
| cpu | 100000 | 128 | 8 | 10 | DOT_PRODUCT | 41.594 | 68.522 | 67.978 | 72.295 | 72.785 | 116.750 | 1.000000 | -284124192 | -96882688 | n/a | n/a / n/a / n/a |
| native | 100000 | 128 | 8 | 10 | DOT_PRODUCT | 67.571 | 64.111 | 63.313 | 67.917 | 71.093 | 124.783 | 1.000000 | 10485760 | 102408192 | n/a | n/a / n/a / n/a |
| cuda | 100000 | 128 | 8 | 10 | DOT_PRODUCT | 134.477 | 3.545 | 3.515 | 3.806 | 3.835 | 2256.816 | 1.000000 | 11285776 | 128819200 | n/a | 0.017 / 2.097 / 0.568 |
| cuvs | 100000 | 128 | 8 | 10 | DOT_PRODUCT | 57.129 | 0.990 | 0.980 | 1.094 | 1.160 | 8082.338 | 1.000000 | 12582912 | 51470336 | n/a | n/a / n/a / n/a |
| cpu | 100000 | 128 | 32 | 10 | DOT_PRODUCT | 33.975 | 274.770 | 273.449 | 289.395 | 297.015 | 116.461 | 1.000000 | 1577176 | 0 | n/a | n/a / n/a / n/a |
| native | 100000 | 128 | 32 | 10 | DOT_PRODUCT | 62.352 | 258.785 | 259.573 | 265.999 | 269.602 | 123.655 | 1.000000 | 11008464 | 102408192 | n/a | n/a / n/a / n/a |
| cuda | 100000 | 128 | 32 | 10 | DOT_PRODUCT | 142.137 | 13.251 | 13.195 | 13.662 | 14.121 | 2414.874 | 1.000000 | 11405936 | 129335296 | n/a | 0.042 / 8.225 / 1.807 |
| cuvs | 100000 | 128 | 32 | 10 | DOT_PRODUCT | 49.886 | 1.608 | 1.578 | 1.904 | 1.905 | 19903.648 | 1.000000 | 11408456 | 51204096 | n/a | n/a / n/a / n/a |
| cpu | 100000 | 128 | 128 | 10 | DOT_PRODUCT | 22.163 | 1092.575 | 1090.982 | 1111.816 | 1123.183 | 117.154 | 1.000000 | 65934440 | 4096 | n/a | n/a / n/a / n/a |
| native | 100000 | 128 | 128 | 10 | DOT_PRODUCT | 56.727 | 1035.053 | 1031.856 | 1060.230 | 1066.812 | 123.665 | 1.000000 | 11324704 | 102408192 | n/a | n/a / n/a / n/a |
| cuda | 100000 | 128 | 128 | 10 | DOT_PRODUCT | 139.087 | 50.602 | 50.333 | 52.258 | 54.125 | 2529.561 | 1.000000 | 11408496 | 128798720 | n/a | 0.052 / 32.159 / 6.278 |
| cuvs | 100000 | 128 | 128 | 10 | DOT_PRODUCT | 47.555 | 3.094 | 3.071 | 3.259 | 3.383 | 41370.473 | 1.000000 | -148266408 | -23941120 | n/a | n/a / n/a / n/a |

## Caveats

- Build time is separate from measured query batches.
- QPS is derived from completed queries divided by summed end-to-end batch time.
- Heap, process RSS, and `nvidia-smi` deltas are approximate snapshots and may be unavailable.
- CUDA phase timings are emitted only by the custom CUDA backend.
- p95/p99 are order statistics from the configured sample count.
- Review the JSON Lines artifact for raw samples and explicit skip records.

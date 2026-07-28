# VectorForge End-to-End Benchmark

Generated from machine-readable JSON Lines. These measurements are local observations, not generalized performance claims.

## System

| Attribute | Value |
| --- | --- |
| `timestamp_utc` | 2026-07-28T06:21:55.714756281Z |
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
| `cuda_toolkit_root` | n/a |
| `cuda_version` | nvcc: NVIDIA (R) Cuda compiler driver<br>Copyright (c) 2005-2025 NVIDIA Corporation<br>Built on Tue_May_27_02:21:03_PDT_2025<br>Cuda compilation tools, release 12.9, V12.9.86<br>Build cuda_12.9.r12.9/compiler.36037853_0 |
| `cuda_device_count` | 1 |
| `cuvs_version` | 26.6.0 |
| `gpu` | NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB, 581.83 |
| `git_sha` | bbb825a6537e0ab7a3fcbaf67becf56b3196ac1e |
| `git_dirty` | False |
| `maven_version` | Picked up JAVA_TOOL_OPTIONS: -Dvectorforge.native.library.dir=vectorforge-native/target/native-lib |
| `cmake_version` | cmake version 4.4.0 |

## Results

| Backend | Vectors | Dims | Batch | k | Metric | Build ms | Batch avg ms | p50 ms | p95 ms | p99 ms | QPS | Recall@k | Heap delta | RSS delta | GPU delta | CUDA H2D / kernel / D2H ms |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cpu | 10000 | 128 | 1 | 10 | EUCLIDEAN | 4.658 | 2.085 | 2.243 | 2.291 | 2.291 | 479.558 | 1.000000 | 8891952 | 8470528 | 0 | n/a / n/a / n/a |
| native | 10000 | 128 | 1 | 10 | EUCLIDEAN | 14.314 | 0.831 | 0.821 | 0.855 | 0.855 | 1203.578 | 1.000000 | 1137744 | 10665984 | 0 | n/a / n/a / n/a |
| cuvs | 10000 | 128 | 1 | 10 | EUCLIDEAN | 237.764 | 0.341 | 0.327 | 0.380 | 0.380 | 2931.422 | 1.000000 | 1510048 | 154877952 | 154140672 | n/a / n/a / n/a |
| cpu | 10000 | 128 | 1 | 10 | COSINE | 2.839 | 1.527 | 1.546 | 1.552 | 1.552 | 654.958 | 1.000000 | 8891952 | 9768960 | 0 | n/a / n/a / n/a |
| native | 10000 | 128 | 1 | 10 | COSINE | 5.144 | 0.850 | 0.856 | 0.898 | 0.898 | 1176.855 | 1.000000 | 503336 | 13705216 | 0 | n/a / n/a / n/a |
| cuvs | 10000 | 128 | 1 | 10 | COSINE | 5.785 | 0.331 | 0.334 | 0.362 | 0.362 | 3017.392 | 1.000000 | 1534624 | 10059776 | 8388608 | n/a / n/a / n/a |
| cpu | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 1.895 | 1.669 | 1.680 | 1.691 | 1.691 | 599.291 | 1.000000 | 9437248 | 946176 | 0 | n/a / n/a / n/a |
| native | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 5.095 | 0.922 | 0.958 | 0.965 | 0.965 | 1084.605 | 1.000000 | 838904 | 0 | 0 | n/a / n/a / n/a |
| cuda | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 159.696 | 0.146 | 0.152 | 0.154 | 0.154 | 6834.600 | 1.000000 | 1046912 | 108425216 | 149946368 | 0.005 / 0.034 / 0.037 |
| cuvs | 10000 | 128 | 1 | 10 | DOT_PRODUCT | 6.169 | 0.445 | 0.363 | 0.664 | 0.664 | 2246.484 | 1.000000 | 1274760 | 4993024 | 8388608 | n/a / n/a / n/a |

## Skips

| Backend | Vectors | Dims | Batch | k | Metric | Reason |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| cuda | 10000 | 128 | 1 | 10 | EUCLIDEAN | metric_not_supported |
| cuda | 10000 | 128 | 1 | 10 | COSINE | metric_not_supported |

## Caveats

- Build time is separate from measured query batches.
- QPS is derived from completed queries divided by summed end-to-end batch time.
- Heap, process RSS, and `nvidia-smi` deltas are approximate snapshots and may be unavailable.
- CUDA phase timings are emitted only by the custom CUDA backend.
- p95/p99 are order statistics from a tiny sample and are not stable tail estimates.
- Review the JSON Lines artifact for raw samples and explicit skip records.

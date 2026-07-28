# cuVS Integration Plan

## Status

Date of replacement-laptop inspection: July 27, 2026.

The earlier missing-installation blocker has been cleared. A verified cuVS
26.06.00 installation is available in Ubuntu under WSL2, and the exact
brute-force adapter, JNI entry points, Java wrapper, tests, demo, and smoke
comparison runner are implemented.

The implementation must use the exact installed API surface described below;
it must not invent or guess cuVS calls.

## Verified Local Environment

- Platform: Ubuntu under WSL2
- Conda environment: `/home/antho/miniforge3/envs/cuvs`
- OpenJDK: 21.0.10
- Maven: 3.9.16
- CMake: 4.4.0
- Ninja: 1.13.2
- Conda GCC/G++: 14.3.0
  (`/home/antho/miniforge3/envs/cuvs/bin/x86_64-conda-linux-gnu-c++`)
- CUDA toolkit/runtime: 12.9
- cuVS: 26.06.00
- DLPack: installed
- GPU: NVIDIA GeForce RTX 3070, compute capability 8.6

Verified installation artifacts:

- cuVS headers under the `cuvs` environment
- cuVS CMake package configuration under the `cuvs` environment
- `/home/antho/miniforge3/envs/cuvs/lib/libcuvs.so`
- `/home/antho/miniforge3/envs/cuvs/lib/libcuvs_c.so`
- all inspected runtime dependencies resolved

A standalone program using the installed cuVS C API was compiled, linked, and
run successfully. The repository integration is independently covered by the
profile-gated correctness and lifecycle tests described below.

## Chosen API Direction

The adapter uses the installed cuVS 26.06.00 C API surface validated by the
direct probe and recorded in this document. Do not substitute remembered APIs
from another cuVS release.

Selected header and functions:

- `<cuvs/neighbors/brute_force.h>`
- `cuvsResourcesCreate`, `cuvsResourcesDestroy`, and `cuvsStreamSync`
- `cuvsBruteForceIndexCreate` and `cuvsBruteForceIndexDestroy`
- `cuvsBruteForceBuild` and `cuvsBruteForceSearch`
- `cuvsRMMAlloc` and `cuvsRMMFree`
- `cuvsGetLastErrorText` for failure details

Explicit metric mapping:

- `EUCLIDEAN` -> `L2Expanded`
- `COSINE` -> `CosineExpanded`, converted from distance to similarity with
  `1.0f - distance`
- `DOT_PRODUCT` -> `InnerProduct`

The selected index is exact brute-force search, so Recall@k is not applicable;
tests compare ordered IDs and scores directly with `CpuBruteForceIndex`.

## Current Build Scaffolding

The repository now exposes:

- Maven profile: `-Pcuvs`
- CMake flag: `VECTORFORGE_ENABLE_CUVS`

Current repository behavior when `VECTORFORGE_ENABLE_CUVS=ON` is requested:

- CMake requires the verified `cuvs` C API package and links `cuvs::c_api`
  plus `CUDA::cudart`
- the isolated `native_cuvs_index.cpp` adapter is compiled
- the Java `CuvsVectorIndex` and opt-in correctness tests are enabled

## Build Validation

- Default full reactor: `mvn clean verify` passed.
- Native profile: `mvn -Pnative clean verify` passed, including 5/5 native
  tests. The WSL portability pass replaces the Windows-only generator
  assumption with a Linux Ninja path; preserve the Windows MinGW behavior.
- CUDA profile: `mvn -Pcuda verify` passed after a clean configure/build,
  including CPU 9/9, native 5/5, and GPU 4/4 tests on the RTX 3070.
- cuVS profile: `mvn -Pcuvs clean verify` passed against cuVS 26.06.00:
  CPU 9/9, native 5/5, custom CUDA 4/4, and cuVS 5/5.
- Windows native profile: `mvn -Pnative clean verify` passed, including 5/5
  native tests.

## Next Implementation Steps

1. Preserve the passing default, native, CUDA, and cuVS profile matrix.
2. Consider larger JMH-quality comparisons with forks, longer warmups, and
   confidence intervals before making performance claims.
3. Consider additional cuVS index families only after inspecting their exact
   installed APIs and adding Recall@k coverage where search is approximate.

## Required Inputs

The formerly missing local inputs are now present: headers, shared libraries,
package/CMake metadata, DLPack, CUDA 12.9, and a successful direct C API probe.
The adapter now uses the installed `cuvs::c_api` target, device-resident
DLPack tensors, metric-specific brute-force indexes, and the existing JNI
handle registry.

## Unsupported Functionality In The Current State

- approximate cuVS index families
- persistence and serialization of cuVS indexes
- generalized performance claims from the short smoke comparison
- Windows cuVS builds; the verified cuVS environment is Ubuntu under WSL2

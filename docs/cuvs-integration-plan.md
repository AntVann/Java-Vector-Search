# cuVS Integration Plan

## Status

Date of local inspection: Wednesday, July 22, 2026.

The cuVS integration path is blocked locally because no cuVS installation was detected on this machine.

In the current repository state, the implementation stops at:

- verified local detection results
- optional Maven/CMake scaffolding
- a concrete integration plan

It does not add invented cuVS adapter calls, guessed headers, guessed JNI entry points, or guessed integration code.

## Local Detection Results

The following locations were inspected:

- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
- `C:\Program Files`
- `C:\Users\Anthony`
- environment variables matching `CUDA`, `CUVS`, `RAPIDS`, `CONDA`

Observed environment:

- `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
- `CUDA_PATH_V12_4=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`

Searches performed:

- `cuvs-config.cmake`
- `cuvs*.dll`
- `cuvs*.lib`
- `libcuvs*`
- `cuvs*.h`
- `cuvs*.hpp`
- `raft*` under the CUDA toolkit tree

Detected cuVS version:

- none

Detected headers:

- none

Detected libraries:

- none

Detected CMake package files:

- none

Detected examples or documentation:

- none

## Chosen API

No cuVS C or C++ API was chosen because no local cuVS installation was found and therefore no installed API surface could be verified.

The future implementation must begin by inspecting the installed headers and examples for the exact version present on the target machine.

## Current Build Scaffolding

The repository now exposes:

- Maven profile: `-Pcuvs`
- CMake flag: `VECTORFORGE_ENABLE_CUVS`

Current behavior when `VECTORFORGE_ENABLE_CUVS=ON` is requested:

- the native build stops immediately with a clear error directing the user to this plan document

This is intentional. It prevents the build from implying that cuVS integration exists when no verified local SDK is present.

## Planned Integration Steps Once cuVS Is Installed

1. Detect and record the exact cuVS version from the installed package metadata, headers, or CMake config files.
2. Inspect the installed headers, examples, and docs to select one supported index type.
3. Choose the smallest viable native adapter surface:
   - create index
   - build or train if required
   - search batch
   - destroy
4. Keep the cuVS-specific code isolated in a dedicated native adapter translation unit.
5. Reuse the existing opaque-handle JNI registry and exception translation path.
6. Add a Java `CuvsVectorIndex` only after the native entry points and supported metric mapping are verified from the installed API.
7. Add correctness tests against `CpuBruteForceIndex`.
8. If the selected cuVS index is approximate, add Recall@k tests against exact CPU results.
9. Extend the demo to compare:
   - CPU Java
   - custom CUDA
   - cuVS

## Required Inputs Before Real Integration

To proceed beyond scaffolding, the machine needs a locally installed cuVS distribution that provides at least:

- headers
- import libraries or shared libraries
- package metadata or CMake config files
- examples or docs sufficient to verify the intended API

## Unsupported Functionality In The Current State

- `CuvsVectorIndex` Java implementation
- cuVS JNI entry points
- cuVS native adapter
- cuVS correctness tests
- cuVS recall tests
- cuVS demo path
- cuVS benchmark claims

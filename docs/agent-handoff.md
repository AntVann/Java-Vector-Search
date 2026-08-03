# VectorForge Agent Handoff

Last updated: August 3, 2026

This file records the repository-backed continuation state. Git history and the
current documentation are authoritative; old Codex task state is not required.

## Repository state

- Repository: `https://github.com/AntVann/Java-Vector-Search.git`
- Release-candidate branch: `feature/portfolio-readiness`
- Release-candidate commit before final audit: `a0d22baf95250d17e15d2326a5ec646441ba35d9`
- Target branch after final verification: `main`

The release candidate is a linear descendant of remote `main`. It includes the
cuVS integration, benchmark suite, standalone Lucene adapter, experimental
disk-backed IVF prototype, production-readiness fixes, and portfolio
documentation.

## Implemented state

- Java CPU exact brute-force search is the default backend.
- Native C++ exact brute-force search is available through the `native` Maven
  profile and the shared JNI handle/lifecycle design.
- Custom CUDA exact dot-product search is experimental and available through the
  `cuda` profile.
- cuVS 26.06 exact brute-force search is experimental and available through the
  `cuvs` profile on the verified Linux/WSL2 environment.
- The Lucene module is a standalone adapter with explicit rebuild semantics; it
  does not modify Lucene internals.
- Disk-backed IVF is an immutable research prototype with documented recovery,
  cache, consistency, and format limitations. It is not a database engine.
- JMH and end-to-end benchmark layers produce machine-readable artifacts. The
  checked-in measurements are local observations, not general speed claims.

See `README.md`, `docs/architecture.md`, `docs/limitations.md`, and the
feature-specific documents for current behavior and constraints.

## Verified laptop environment

- Ubuntu under WSL2
- Conda environment: `/home/antho/miniforge3/envs/cuvs`
- OpenJDK 21.0.10
- Maven 3.9.16
- CMake 4.4.0 and Ninja 1.13.2
- Conda GCC/G++ 14.3.0
- CUDA toolkit/runtime 12.9
- cuVS 26.06.00
- NVIDIA GPU passthrough with compute capability 8.6

Verified cuVS artifacts include `libcuvs.so`, `libcuvs_c.so`, installed headers,
and CMake package configuration under the Conda environment. The adapter uses
only APIs verified against that installation; do not infer APIs for other cuVS
versions.

## Release verification

Before promoting a continuation branch, run:

```bash
mvn clean verify
mvn clean verify -Pnative
```

On the configured WSL2 GPU environment, also run:

```bash
conda activate cuvs
export CMAKE_GENERATOR=Ninja
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX/targets/x86_64-linux"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
mvn clean verify -Pcuvs
```

The `cuvs` profile builds and tests the native CPU, custom CUDA, and cuVS paths.
Confirm that CUDA and cuVS tests executed rather than being skipped. The current
CI definitions are in `.github/workflows/ci.yml` and
`.github/workflows/gpu-self-hosted.yml`.

## Continuation rules

1. Preserve the default CPU-only build without CUDA or cuVS.
2. Keep optional native/GPU dependencies isolated behind Maven/CMake features.
3. Preserve exactly-once native cleanup and meaningful Java exception mapping.
4. Add cuVS functionality only from installed, verified headers and APIs.
5. Retain raw benchmark output and disclose hardware, workload, warm-up, sample
   count, backend order, and measurement limitations.
6. Do not describe experimental Lucene, disk-IVF, CUDA, or cuVS paths as
   production-ready.

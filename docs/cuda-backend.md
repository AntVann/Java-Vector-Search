# CUDA Backend

## Scope

The current CUDA backend is an educational exact implementation. It is designed to validate correctness, GPU residency, JNI integration, and timing instrumentation before any cuVS work begins.

## Architecture

- Java entrypoint: `CudaBruteForceIndex`
- Native library: `vectorforge_jni.dll`
- Native CUDA path: runtime-loaded CUDA Driver API plus NVRTC
- Correctness baseline: `CpuBruteForceIndex`

The current implementation supports only `DOT_PRODUCT`.

## Kernel Design

- Kernel name: `vectorforge_dot_product`
- Launch shape: one-dimensional grid with `256` threads per block
- Work assignment: one thread computes one `(query, vector)` score
- Query index: `global_index / vector_count`
- Vector index: `global_index % vector_count`
- Inner loop: straightforward accumulation across all dimensions

This is intentionally simple. It is easy to reason about and compare against the CPU reference, but it is not optimized for memory reuse, shared-memory tiling, or warp-level reductions.

## Memory Layout

- Indexed vectors: contiguous row-major `float` array on device
- Queries: contiguous row-major `float` array on device
- Scores: contiguous row-major `float` score matrix on device
  - shape: `query_count x vector_count`
- IDs: retained on the host alongside the index for deterministic tie-breaking and exact top-k selection

The index vectors remain resident on the GPU after build. Queries and score buffers are resized lazily and then reused across searches.

## Synchronization Points

The current search path synchronizes explicitly at these points:

1. After the kernel launch, an end event is synchronized before kernel timing is read.
2. The device-to-host score copy happens only after kernel completion.
3. Exact top-k selection starts only after the device-to-host score transfer finishes.

These synchronization points are intentional so the reported phase timings are defensible and the returned results are fully materialized before Java observes them.

## Timing Model

Each CUDA search records:

- Host-to-device query transfer time
- Kernel execution time
- Device-to-host score transfer time
- Total end-to-end search latency

The total latency includes:

- Buffer growth when needed
- Query upload
- Kernel execution
- Score download
- Host-side exact top-k selection

The phase timings do not include Java-side input packing into direct buffers.

## Exactness Model

The backend is exact for dot-product search because:

1. Every query is scored against every indexed vector.
2. No approximate pruning or partitioning is used.
3. Top-k selection is exact on the full score matrix.
4. Ties are broken by ascending vector ID, matching the repository’s deterministic behavior.

## Known Bottlenecks

- The kernel writes the full query-by-vector score matrix, which increases device memory traffic.
- Exact top-k selection is currently performed on the host, so every score must be copied back to host memory.
- The dot-product kernel does not use shared-memory tiling or vectorized loads.
- Searches are serialized per index so query and score buffers can be reused safely.
- The implementation currently supports only one metric.

These are conscious tradeoffs for the current implementation. They keep the code small, auditable, and correctness-focused.

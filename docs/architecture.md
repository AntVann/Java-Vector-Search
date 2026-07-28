# VectorForge Architecture

## Scope

The repository is a modular research and comparison project. The pure-Java CPU
backend is the portability baseline; native, CUDA, cuVS, Lucene, benchmark, and
disk-IVF capabilities are isolated in separate Maven modules.

## Java API Layer

`vectorforge-api` defines:

- `VectorIndex` as the backend-independent lifecycle, scalar-search, and batch-search contract
- Immutable value objects for results, parameters, and metrics
- `DistanceMetric` as the algorithm-selection enum

The API intentionally keeps search inputs small and explicit so JNI and future off-heap transfers can map onto it cleanly.

## CPU Backend

`vectorforge-cpu` implements `CpuBruteForceIndex`:

- Input vectors are validated and flattened into a contiguous `float[]`
- IDs are stored in a parallel `long[]`
- Vector norms are precomputed once for cosine similarity
- Search uses a bounded heap to maintain only the best `k` results
- Tie-breaking is deterministic: equal scores are ordered by ascending ID
- Searches are safe to run concurrently after a successful build because index state is immutable once published

## JNI Boundary

`vectorforge-native` now contains:

- `NativeBruteForceIndex`, which implements `VectorIndex`
- `NativeBindings`, a small JNI entrypoint surface
- `NativeLibraryLoader`, which loads the shared library from a configured path
- A C++17 shared library built with CMake

The boundary model is:

- Java owns backend selection and public lifecycle
- Java stores opaque `long` handles, not native pointers
- Native code owns the actual index objects in a handle table
- Native code uses RAII-managed `shared_ptr<NativeIndex>` instances
- Errors cross the boundary as structured Java exceptions

JNI entrypoints currently support:

- Native index creation from direct buffers
- Batched search from direct buffers
- Native index destruction
- CUDA index creation through the same opaque-handle model
- CUDA search with an explicit timing buffer

## Native Resource Ownership

The detailed ownership model is documented in [native-memory-model.md](native-memory-model.md). In the current JNI path:

- Java arrays are validated and packed into direct buffers
- Native code copies those buffers into a C++ `NativeIndex`
- Native index lifetime is bound to the Java wrapper handle lifecycle

## CUDA Backend

`vectorforge-gpu` now provides `CudaBruteForceIndex`:

- Indexed vectors are copied to device memory once during build and stay resident until `close()`
- Query and score device buffers are grown lazily and then reused across searches
- Batched queries are supported through one native call
- The current kernel computes exact dot-product scores for every `(query, vector)` pair
- Exact top-k selection happens on the host after copying scores back from device memory
- Transfer, kernel, and total timing are recorded for every CUDA search

The CPU implementation remains the correctness reference for this work.

## cuVS Adapter

The cuVS integration is isolated behind a small native adapter layer so:

- Java code is not tightly coupled to the version-specific cuVS API surface
- cuVS discovery can be profile-gated
- The CPU-only build continues to work without CUDA or cuVS installed

## Error Propagation

Current behavior:

- Invalid build inputs throw `IllegalArgumentException`
- Invalid lifecycle usage throws `IllegalStateException`
- Public API records validate mandatory fields at construction time
- JNI argument validation throws `IllegalArgumentException`
- Invalid or closed native handles throw `IllegalStateException`
- Unexpected native failures throw `NativeInteropException`
- CUDA and cuVS failures will never be swallowed or converted into silent fallbacks

## Thread-Safety Model

`CpuBruteForceIndex` supports:

- Synchronized `build()` and `close()`
- Lock-free concurrent `search()` calls after build through immutable published state

`NativeBruteForceIndex` supports:

- Concurrent public searches guarded by a read lock
- Serialized `build()` and `close()` through a write lock
- Safe replacement or destruction of native handles without racing in-flight public searches

`CudaBruteForceIndex` supports:

- Serialized searches so a single GPU-resident buffer set can be reused safely
- Serialized `build()` and `close()` through the same lifecycle lock pattern
- Clear failure when CUDA support is not compiled or when no usable device is present

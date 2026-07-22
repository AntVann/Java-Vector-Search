# Native Memory Model

## Status

The current repository includes a JNI-backed CPU-native backend and a CUDA backend built on the same opaque-handle model. This document reflects the current native ownership rules and the expected GPU evolution path.

## Ownership Rules

### Java-owned allocations

- Public API objects such as search parameters and result lists
- On-heap datasets used by the CPU backend
- Direct buffers allocated by Java for JNI argument passing and result collection

### Native-owned allocations

- Handle-table entries stored behind opaque `long` IDs
- `NativeIndex` objects storing copied vectors, IDs, and precomputed norms
- Future device memory allocated for CUDA or cuVS backends

## Freeing Strategy

- The side that allocates a resource is responsible for freeing it unless ownership is explicitly transferred
- Java direct buffers are reclaimed by the JVM
- Native handles are released through explicit destroy functions invoked from `AutoCloseable.close()`
- Cleanup must be idempotent so duplicate close calls do not double-free native resources

## Handle Lifetime

- Java code holds a logical handle reference, not a raw pointer API
- A closed Java wrapper transitions permanently to a closed state
- Searches on a closed handle must fail fast with a Java exception
- Native code validates handle presence in a registry before use

## Device Memory Ownership

The current native backend keeps vectors in C++ heap memory. The planned CUDA and cuVS backends will keep index vectors resident on the GPU after build. Query-time temporary buffers may be reused across searches where thread-safety allows it.

## Exceptions

- Native allocation failures become Java exceptions
- Partial native construction rolls back through C++ RAII and `shared_ptr`
- Native validation errors map to `IllegalArgumentException` or `IllegalStateException`
- CUDA errors are surfaced directly and never ignored

## Cleanup on Failure

Every JNI entrypoint should follow the same rule:

1. Validate all arguments before allocating expensive resources where practical
2. Use RAII on the C++ side for intermediate resources
3. Transfer ownership only after full construction succeeds
4. Leave Java-visible handles in a consistent state on all failure paths

## Current JNI Flow

1. Java validates `float[][]` and `long[]` inputs
2. Java packs them into direct `ByteBuffer` instances
3. JNI validates direct-buffer capacity and shape
4. Native code copies the vectors and IDs into a `NativeIndex`
5. A new opaque `long` handle is returned to Java
6. Search calls pack queries into direct buffers and receive IDs and scores through direct output buffers
7. `close()` destroys the native handle exactly once

## Current CUDA Flow

1. Java validates vectors, IDs, and query batches and packs them into direct buffers
2. JNI copies the build-time vectors into a CUDA-backed native index
3. The CUDA index allocates device memory for the vector matrix once and keeps it resident
4. Search calls reuse device-side query and score buffers when the batch size fits previous allocations
5. Queries are transferred from Java direct buffers into device memory
6. The CUDA kernel writes the full score matrix into device memory
7. Scores are copied back to native host memory for exact top-k selection
8. Only the final IDs and scores are written back into Java-owned direct output buffers

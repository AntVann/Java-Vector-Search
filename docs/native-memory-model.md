# Native Memory Model

## Status

The repository includes JNI-backed CPU-native, custom CUDA, and cuVS brute-force backends built on the same opaque-handle model.

## Ownership Rules

### Java-owned allocations

- Public API objects such as search parameters and result lists
- On-heap datasets used by the CPU backend
- Direct buffers allocated by Java for JNI argument passing and result collection

### Native-owned allocations

- Handle-table entries stored behind opaque `long` IDs
- `NativeIndex` objects storing copied vectors, IDs, and precomputed norms
- CUDA allocations owned by the custom CUDA backend
- cuVS resources, brute-force indexes, and RMM allocations owned by the cuVS adapter

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

The CPU-native backend keeps vectors in C++ heap memory. The custom CUDA
backend keeps its vector matrix resident on the GPU and reuses query and score
buffers. The cuVS adapter copies the dataset into an adapter-owned RMM device
allocation and passes a CUDA `DLManagedTensor` view to
`cuvsBruteForceBuild`. The allocation remains alive for the lifetime of all
metric-specific indexes. At search time the adapter allocates query, neighbor,
and distance buffers with `cuvsRMMAlloc`, copies inputs and outputs on the cuVS
resource stream, synchronizes, and releases those buffers with `cuvsRMMFree`.

The cuVS adapter owns:

- one `cuvsResources_t`
- one `cuvsBruteForceIndex_t` for each supported metric because the metric is fixed at build time
- the Java-ID mapping used to translate cuVS signed 64-bit row indices back to VectorForge `long` IDs
- the persistent RMM allocation containing the dataset
- query-time RMM allocations, scoped to one synchronized search

RAII destroys brute-force indexes before their shared resource. cuVS cleanup failures in destructors cannot cross the JNI boundary.

## Exceptions

- Native allocation failures become Java exceptions
- Partial native construction rolls back through C++ RAII and `shared_ptr`
- Native validation errors map to `IllegalArgumentException` or `IllegalStateException`
- CUDA errors are surfaced directly and never ignored
- cuVS status values are checked against `CUVS_SUCCESS`; failures include `cuvsGetLastErrorText()` when available

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

## Current cuVS Flow

1. Java validates and packs vectors and IDs into direct buffers.
2. JNI copies those values into native vectors before registering a handle.
3. The adapter creates a `cuvsResources_t`, copies the dataset into persistent
   RMM device memory, and creates metric-specific brute-force indexes with
   `cuvsBruteForceIndexCreate` and `cuvsBruteForceBuild`.
4. Search allocates CUDA query, signed 64-bit neighbor, and float distance
   tensors through RMM.
5. `cuvsBruteForceSearch` runs with `cuvsFilter{0, NO_FILTER}`.
6. The resource stream is synchronized before host results are consumed.
7. Returned row indices are bounds-checked and translated through the retained `long` ID vector.
8. Search-scoped RMM buffers are freed; closing the Java wrapper destroys the registered native index and all cuVS-owned index handles.

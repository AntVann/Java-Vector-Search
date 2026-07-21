# Native Memory Model

## Status

Milestone 1 is CPU-only and keeps index state entirely on the Java heap. This document captures the intended ownership model for future native and GPU work so the project can evolve without ambiguity.

## Planned Ownership Rules

### Java-owned allocations

- Public API objects such as search parameters and result lists
- On-heap datasets used by the CPU backend
- Direct buffers allocated by Java when Java is responsible for lifecycle

### Native-owned allocations

- Opaque native index handles
- Host-side native work buffers allocated inside the JNI layer
- Device memory allocated for CUDA or cuVS backends

## Freeing Strategy

- The side that allocates a resource is responsible for freeing it unless ownership is explicitly transferred
- Native handles are released through explicit destroy functions invoked from `AutoCloseable.close()`
- Cleanup must be idempotent so duplicate close calls do not double-free native resources

## Handle Lifetime

- Java code holds a logical handle reference, not a raw pointer API
- A closed Java wrapper transitions permanently to a closed state
- Searches on a closed handle must fail fast with a Java exception

## Device Memory Ownership

The planned CUDA and cuVS backends will keep index vectors resident on the GPU after build. Query-time temporary buffers may be reused across searches where thread-safety allows it.

## Exceptions

- Native allocation failures become Java exceptions
- Partial native construction must roll back already-allocated resources
- CUDA errors are surfaced directly and never ignored

## Cleanup on Failure

Every JNI entrypoint should follow the same rule:

1. Validate all arguments before allocating expensive resources where practical
2. Use RAII on the C++ side for intermediate resources
3. Transfer ownership only after full construction succeeds
4. Leave Java-visible handles in a consistent state on all failure paths


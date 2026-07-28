# Experimental disk-backed IVF search

This module is a research prototype for searching collections that should not be resident in GPU
memory all at once. It is not a database engine and does not provide online updates, transactions,
replication, compaction, or production recovery tooling.

## API decision

The shared `VectorIndex`, `SearchParameters`, and `IndexMetrics` APIs are unchanged. IVF controls
are implementation-specific:

- `DiskIvfBuildConfig`: centroid count, fixed-seed k-means iterations, and build metric.
- `DiskIvfSearchConfig`: partitions to probe, cache-byte budget, and maximum backend-batch bytes.
- `searchDetailed()`: normal results plus disk, decode, backend-transfer/build, backend-search,
  end-to-end, candidate, and cache measurements.
- `diskMetrics()`: generation, on-disk bytes, cache occupancy, hits, misses, and evictions.

The prototype currently supports Euclidean distance only. Search rejects a metric different from
the metric recorded in the index.

## File format

An index root contains immutable generations:

```text
index-root/
├── CURRENT
└── generations/
    └── <uuid>/
        ├── READY
        ├── manifest.vfi
        ├── centroids.vfc
        └── partitions/
            ├── part-00000.vfp
            └── ...
```

All binary integers and IEEE-754 float32 values are little-endian. Every binary file ends with a
CRC32C over all preceding bytes. Headers contain an eight-byte magic value, format version `1`,
and generation UUID.

`manifest.vfi` records dimensions, centroid and partition counts, vector count, metric code,
seed, training iterations, centroid-file size/checksum, and, for each partition, its ID, vector
count, byte length, and checksum.

`centroids.vfc` stores dimensions, centroid count, encoding code (`1` = float32), and row-major
centroids.

Each `part-N.vfp` stores its partition ID, dimensions, vector count, then records:

```text
int64 vectorId
float32 vector[dimensions]
```

Records are sorted by vector ID. Empty partitions are valid header-plus-checksum files. Opening an
index eagerly validates the manifest, centroid file, every referenced partition length, headers,
UUIDs, and checksums. Missing or corrupt files fail open; they are never silently skipped.

## Construction and consistency

Construction validates dimensions, finite values, and unique IDs, trains fixed-seed k-means, and
assigns every vector to its nearest centroid. Files are written and forced inside a uniquely named
staging generation. `READY` is written last, the generation directory is renamed, and the forced
unique `CURRENT.<generation>.tmp` pointer is atomically moved over `CURRENT` where the
filesystem supports atomic moves.

The fallback for filesystems without atomic moves is a replace move and has weaker crash
guarantees. Incomplete staging and unreferenced generations are ignored. Rebuild publishes a new
generation; it does not mutate the old one. This version does not delete old generations
automatically.

The convenience `build(float[][], long[])` still requires build input in Java memory. A streaming
builder is not implemented yet, so the prototype demonstrates disk-backed query residency rather
than truly out-of-heap ingestion.

## Query path

1. Score the in-memory centroid table and select `min(nprobe, centroidCount)` centroids.
2. Visit their posting lists in centroid-distance order.
3. Reuse a flat decoded partition from the byte-weighted LRU cache, or stream records with
   positional `FileChannel` I/O.
4. Process every record in the selected partitions in batches capped by
   `maxBackendBatchBytes`.
5. Build and search a temporary supplied `VectorIndex` for each batch.
6. Merge every batch's top-k into a global metric-aware top-k, close each temporary backend, and
   return phase timings.

`CpuBruteForceIndex::new` is the default example reranker. Native, custom CUDA, or cuVS factories
can be supplied when available. Temporary backend construction is reported as
`backendTransferNanos`; it includes CPU allocation/packing and, for a GPU backend, its build and
host-to-device work. Device-to-host timing is `-1` because the generic `VectorIndex` API does not
expose that phase. The prototype does not fabricate it.

## Cache and memory bounds

The partition cache is a synchronized access-order `LinkedHashMap` storing flat `long[]` and
`float[]` arrays. Its conservative primitive-payload weight is:

```text
64 + vectorCount * (8 + dimensions * 4)
```

Entries are evicted least-recently-used until the configured byte budget is met. An entry larger
than the cache budget bypasses the cache and is streamed in backend-sized batches. Failed loads
are not cached.

Each reranker batch is conservatively limited by `maxBackendBatchBytes`, including estimated Java
row-array and reference overhead. Therefore the explicit Java working bound is approximately:

```text
centroid table + cacheBytes + about 2 * maxBackendBatchBytes + O(k)
```

The factor of roughly two covers a flat decoded batch coexisting with the row arrays submitted to
the backend. Only one bounded batch is given to a GPU backend at a time. Backend-specific copies,
workspace, and result allocation are not described by the common API and may add memory beyond
that estimate; choose a conservative backend-batch budget and measure the selected backend.
On a cache miss for an admissible partition, the encoded read buffer briefly coexists with the
flat decoded cache entry, so transient cache-fill memory can approach twice that entry's logical
weight. Cache residency itself never exceeds `cacheBytes`.

## Run tests and the smoke benchmark

```bash
mvn -pl vectorforge-disk -am test
mvn -pl vectorforge-disk -am install -DskipTests
mvn -pl vectorforge-disk exec:java \
  -Dexec.mainClass=com.vectorforge.disk.DiskIvfBenchmark
```

The benchmark generates a fixed-seed `10,000 x 32` clustered collection and 20 queries, computes
exact CPU ground truth outside timed search sections, warms the JVM/cache with three queries, and
prints JSON Lines for `nprobe=1,4,16`. It reports build time, Recall@10, centroid time, logical
partition read time, candidate decode/load time, backend build/transfer time, backend search time,
end-to-end latency, candidate count, and application-cache hits/misses.

These timings include effects of the operating-system page cache. A Java cache miss is not proof
of a physical device read, and the smoke run is not a storage-throughput benchmark.

### Observed smoke result

One actual Windows/JDK 21 run on this laptop produced:

| nprobe | Recall@10 | Avg candidates | Disk read ms | Candidate load ms | Backend transfer/build ms | Backend search ms | End-to-end ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.970 | 702.90 | 0.387 | 0.178 | 0.151 | 0.040 | 0.842 |
| 4 | 1.000 | 2,425.10 | 0.369 | 0.375 | 0.289 | 0.192 | 1.350 |
| 16 | 1.000 | 10,000.00 | 0.000 | 0.681 | 0.681 | 0.306 | 1.784 |

Index construction took `333.631 ms`. The 16-probe measured phase was fully served by the
application cache after warm-up, which is why its recorded partition read time was zero. These are
small fixed-seed smoke observations, not generalized performance or scalability results.

## Current limitations

- Euclidean distance and raw float32 vectors only.
- Simple fixed-iteration k-means; no k-means++, distributed training, compression, or quantization.
- `nprobe` controls the recall/latency tradeoff; all records in selected partitions are reranked.
- One temporary reranker index is constructed per bounded candidate batch.
- Query methods are synchronized; concurrent search optimization is not implemented.
- Partition CRCs are eagerly checked at open, which reads the whole generation once.
- Manifest and centroid files are assembled in Java byte arrays; partition files are written and
  checksummed incrementally. The in-memory centroid table must still fit a Java array.
- No streaming builder, incremental insert/delete, generation garbage collection, or repair.
- No deterministic public Java API for unmapping files is required because partition data uses
  buffered positional reads rather than long-lived memory maps.
- Results are valid only for tested datasets and hardware. No larger-than-tested scale claim is
  made.

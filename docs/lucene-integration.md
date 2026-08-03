# Lucene integration

The `vectorforge-lucene` module is a standalone adapter built on Lucene 10.5.0 public APIs.
Lucene 10 and VectorForge both require Java 21. The default implementation uses
`CpuBruteForceIndex`, so no native library, CUDA toolkit, GPU, or cuVS installation is needed.

## Architecture

Each document contains:

| Field | Lucene representation | Purpose |
| --- | --- | --- |
| `external_id` | stored/indexed `StringField` | application document identity and deletes |
| `text` | stored/indexed `TextField` | searchable/source text |
| `dense_vector` | `KnnFloatVectorField` | Lucene built-in vector comparison |
| `metadata` | stored/indexed `StringField` | exact-value filtering |
| `vectorforge_id` | stored numeric field | stable mapping to a VectorForge result ID |
| `vectorforge_vector` | stored binary field | rebuild input for the external index |

`LuceneVectorAdapter` owns a Lucene writer, near-real-time reader, and a factory for creating
`VectorIndex` snapshots. Adds, updates, and deletes enter Lucene first. `refreshAndRebuild()` opens
a new reader, walks only live stored documents, builds a fresh backend, and atomically replaces
the published VectorForge index and ID mapping only after the build succeeds.
Lucene internal document numbers are never used as durable identifiers because they change
across segments and merges.

If candidate reader or backend construction fails, the prior published snapshot remains active.
If releasing a retired reader or backend fails after publication, `refreshAndRebuild()` throws
`LuceneRefreshCleanupException`. That exception explicitly means the new snapshot is active and
reports its live-document count; it is a cleanup diagnostic, not a rollback signal.

## Search and filtering

VectorForge search returns stable `long` IDs. The adapter resolves each ID through the snapshot
mapping and returns the external ID, text, and metadata.

The simple VectorForge metadata filter runs **after vector search**. To preserve correct filtered
top-k results, the adapter requests all live vectors, retains candidates whose metadata matches,
and stops after k matches. This is correct but deliberately not scalable. A future backend-aware
adapter could maintain one index per filter value or add native filtered-search support.

For comparison, `searchLucene()` supplies a `TermQuery` to `KnnFloatVectorQuery`, so Lucene
applies its metadata filter **during vector search**. The two engines' scores are not directly
comparable because Lucene transforms raw distances/similarities into its own score convention;
compare identities and ordering instead.

## Run

```bash
mvn -pl vectorforge-lucene -am test
mvn -pl vectorforge-lucene -am package -DskipTests
java -jar vectorforge-lucene/target/vectorforge-lucene-demo.jar
```

The deterministic demo indexes four documents, rebuilds a CPU VectorForge index, searches with
a metadata filter, prints Lucene's built-in k-NN comparison, and reports raw VectorForge backend
latency separately from complete adapter latency.

## Refresh, deletes, updates, and segments

- Writes are not visible to either search path until `refreshAndRebuild()`.
- A refresh creates one consistent live-document snapshot for Lucene and VectorForge.
- Deletes remain visible in the previous snapshot until refresh, then disappear from both.
- External IDs are unique within the adapter. Duplicate `add()` calls are rejected.
- Updates use Lucene's atomic `updateDocument` operation and receive a new VectorForge ID.
- The mapping does not rely on Lucene segment-local document IDs, so segment merges do not break
  it. Rebuild cost is still proportional to all live documents.
- Stored vector copies increase index size and are required by this simple rebuild design.
- This in-memory example does not persist the next-ID counter. A production implementation must
  use a persistent, collision-free ID allocator.
- Concurrent writes/searches, incremental segment rebuilds, merge callbacks, crash recovery,
  multi-valued metadata, and pre-search VectorForge filtering are outside the current adapter.
- Dot-product indexed and query vectors are required to have unit length; the adapter validates
  this constraint. The demo uses Euclidean distance.
- Exact CPU results and Lucene's HNSW results can differ on larger datasets because Lucene vector
  search may be approximate.
- Equal-score ties may use different secondary ordering. Direct ordering comparisons are valid
  only when the compared scores are not tied.

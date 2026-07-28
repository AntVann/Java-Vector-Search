# JMH Search Benchmarks

Build the benchmark jar with:

```bash
mvn -pl vectorforge-benchmarks -am package -DskipTests
```

Run all backends usable by the current native library and hardware:

```bash
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar
```

The launcher always enables the Java CPU backend. Native CPU, custom CUDA, and cuVS
benchmarks are included only when their libraries and required hardware are available.
Build with `-Pnative`, `-Pcuda`, or `-Pcuvs` and set
`vectorforge.native.library.dir` when benchmarking an optional backend.
The default launcher prints the enabled backend names; it does not emit structured
skip records for unavailable backends. An explicit benchmark expression bypasses
automatic selection and intentionally fails fast when its requested backend is unavailable.

JMH parameters cover dataset size, dimensions, query batch size, `k`, and distance
metric. The custom CUDA backend is restricted to its supported dot-product metric.
Datasets, queries, and indexes are created once in trial setup, outside measured
search calls. Fixed seeds make every backend receive the same inputs.

Batch search is a backend-specific extension rather than part of the `VectorIndex`
interface. The Java CPU implementation may execute a batch by looping over queries,
while the native, CUDA, and cuVS implementations pass the batch through their native
search paths. Treat batch results as end-to-end backend measurements, not as proof
that every backend uses the same internal batching strategy.

`IndexBuildBenchmark` separately measures index construction and close. Its vectors
and IDs are generated once in trial setup, so input generation is not timed.

A quick CPU smoke run is:

```bash
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  '.*BackendSearchBenchmark.cpuSearch' \
  -p vectorCount=128 -p dimensions=16 -p batchSize=1 -p k=5 \
  -p metric=EUCLIDEAN -wi 1 -i 1 -f 1 -w 100ms -r 100ms
```

The checked-in functional smoke artifact was produced with:

```bash
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  '.*BackendSearchBenchmark.cpuSearch' \
  -p vectorCount=128 -p dimensions=16 -p batchSize=1 -p k=5 \
  -p metric=EUCLIDEAN -wi 1 -i 1 -f 1 -w 100ms -r 100ms \
  -rf json -rff benchmark-results/jmh-cpu-smoke.json
```

That run completed successfully at `2.001 us/op`. The JSON file is retained only
as functional evidence that the packaged JMH launcher, benchmark discovery,
parameter binding, fork, and result serialization work. Its timing is not a
performance claim or a baseline.

A quick index-build smoke run is:

```bash
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  '.*IndexBuildBenchmark.cpuBuild' \
  -p vectorCount=128 -p dimensions=16 \
  -wi 1 -i 1 -f 1 -w 100ms -r 100ms
```

## Scale runs

The supported scale values are:

- vectors: `10000`, `100000`, `1000000`
- dimensions: `128`, `384`, `768`
- query batches: `1`, `8`, `32`, `128`
- `k`: `1`, `10`, `100`

Run deliberate points rather than the full Cartesian product. For example:

```bash
# Small representative point
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  '.*BackendSearchBenchmark.cpuSearch' \
  -p vectorCount=10000 -p dimensions=128 -p batchSize=1 -p k=1

# Medium representative point
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  '.*BackendSearchBenchmark.cpuSearch' \
  -p vectorCount=100000 -p dimensions=384 -p batchSize=8 -p k=10

# Large search points; run batches independently
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  '.*BackendSearchBenchmark.cpuSearch' \
  -p vectorCount=1000000 -p dimensions=768 -p batchSize=32 -p k=100
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  '.*BackendSearchBenchmark.cpuSearch' \
  -p vectorCount=1000000 -p dimensions=768 -p batchSize=128 -p k=100

# Build-only scale sweep; each combination is a separate JMH trial
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  '.*IndexBuildBenchmark.cpuBuild' \
  -p vectorCount=10000,100000,1000000 -p dimensions=128
java -jar vectorforge-benchmarks/target/vectorforge-benchmarks.jar \
  '.*IndexBuildBenchmark.cpuBuild' \
  -p vectorCount=10000,100000 -p dimensions=384,768
```

A single `1,000,000 x 768` float dataset is about 2.86 GiB before Java row-object,
query, result, native-copy, or GPU allocation overhead. CPU/native builds may hold
multiple copies, and GPU backends also need device memory and working buffers.
Increase the JVM heap deliberately, check available RAM/VRAM first, and run the
largest cases one at a time. Avoid comma-separated values across every parameter:
JMH expands them into a large Cartesian product with long runtime and substantial
allocation pressure.

Do not compare numbers from different machines, power modes, JVM versions, or JMH
settings. GPU calls copy results back and synchronize before returning, so their
reported operation time includes completion rather than asynchronous launch alone.

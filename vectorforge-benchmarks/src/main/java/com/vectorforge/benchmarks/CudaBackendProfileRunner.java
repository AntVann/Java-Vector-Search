package com.vectorforge.benchmarks;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.gpu.CudaBruteForceIndex;
import com.vectorforge.gpu.CudaSearchTimings;
import com.vectorforge.nativeindex.NativeBindings;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * Profiles the CUDA backend using both the high-level Java API and the low-level JNI binding.
 *
 * <p>The high-level mode captures Java validation, direct-buffer allocation, JNI, native execution,
 * and result materialization. The raw mode reuses direct buffers and exercises one JNI call so its
 * gap from high-level mode approximates Java-side allocation and marshaling overhead.
 */
public final class CudaBackendProfileRunner {

    private static final SearchParameters SEARCH_PARAMETERS = new SearchParameters(DistanceMetric.DOT_PRODUCT);
    private static final int DOT_PRODUCT_METRIC_CODE = 2;

    private CudaBackendProfileRunner() {
    }

    public static void main(String[] args) {
        Config config = Config.parse(args);
        if (!CudaBruteForceIndex.isCudaAvailable()) {
            throw new IllegalStateException("CUDA backend is unavailable. Build with -Pcuda and set vectorforge.native.library.dir.");
        }

        float[][] vectors = randomVectors(config.vectorCount(), config.dimensions(), 42L);
        long[] ids = sequentialIds(config.vectorCount());
        float[][] smallQueries = randomVectors(config.smallQueryCount(), config.dimensions(), 84L);
        float[][] batchQueries = randomVectors(config.batchQueryCount(), config.dimensions(), 126L);

        System.out.println("VectorForge CUDA Profile");
        System.out.println("vectorCount=" + config.vectorCount());
        System.out.println("dimensions=" + config.dimensions());
        System.out.println("k=" + config.k());
        System.out.println("warmupIterations=" + config.warmupIterations());
        System.out.println("measurementIterations=" + config.measurementIterations());
        System.out.println("smallQueryCount=" + config.smallQueryCount());
        System.out.println("batchQueryCount=" + config.batchQueryCount());
        System.out.println();

        runScenario("single", vectors, ids, smallQueries, config);
        System.out.println();
        runScenario("batch", vectors, ids, batchQueries, config);
    }

    private static void runScenario(
            String scenarioName,
            float[][] vectors,
            long[] ids,
            float[][] queries,
            Config config
    ) {
        ScenarioSpec scenario = new ScenarioSpec(scenarioName, queries);
        ProfileSummary highLevel = profileHighLevel(vectors, ids, scenario, config);
        ProfileSummary rawNative = profileRawNative(vectors, ids, scenario, config);

        printSummary("high_level", scenarioName, highLevel);
        printSummary("raw_native", scenarioName, rawNative);

        double javaOverheadMicros = highLevel.endToEndMicros().average() - rawNative.endToEndMicros().average();
        System.out.printf(Locale.US, "comparison scenario=%s java_overhead_avg_us=%.3f%n", scenarioName, javaOverheadMicros);
    }

    private static ProfileSummary profileHighLevel(
            float[][] vectors,
            long[] ids,
            ScenarioSpec scenario,
            Config config
    ) {
        try (CudaBruteForceIndex index = new CudaBruteForceIndex()) {
            index.build(vectors, ids);
            warmUpHighLevel(index, scenario, config);

            double[] endToEndMicros = new double[config.measurementIterations()];
            double[] h2dMillis = new double[config.measurementIterations()];
            double[] kernelMillis = new double[config.measurementIterations()];
            double[] d2hMillis = new double[config.measurementIterations()];
            double[] totalMillis = new double[config.measurementIterations()];
            long resultChecksum = 0L;
            double scoreChecksum = 0.0;

            for (int i = 0; i < config.measurementIterations(); i++) {
                long start = System.nanoTime();
                SearchOutcome outcome = executeHighLevel(index, scenario.queries(), config.k());
                long end = System.nanoTime();

                CudaSearchTimings timings = Objects.requireNonNull(index.lastSearchTimings(), "missing CUDA timings");
                endToEndMicros[i] = (end - start) / 1_000.0;
                h2dMillis[i] = timings.hostToDeviceMillis();
                kernelMillis[i] = timings.kernelMillis();
                d2hMillis[i] = timings.deviceToHostMillis();
                totalMillis[i] = timings.totalMillis();
                resultChecksum += outcome.idChecksum();
                scoreChecksum += outcome.scoreChecksum();
            }

            return new ProfileSummary(
                    summarize(endToEndMicros),
                    summarize(h2dMillis),
                    summarize(kernelMillis),
                    summarize(d2hMillis),
                    summarize(totalMillis),
                    resultChecksum,
                    scoreChecksum
            );
        }
    }

    private static void warmUpHighLevel(CudaBruteForceIndex index, ScenarioSpec scenario, Config config) {
        for (int i = 0; i < config.warmupIterations(); i++) {
            executeHighLevel(index, scenario.queries(), config.k());
        }
    }

    private static SearchOutcome executeHighLevel(CudaBruteForceIndex index, float[][] queries, int k) {
        if (queries.length == 1) {
            List<SearchResult> results = index.search(queries[0], k, SEARCH_PARAMETERS);
            return checksumSingle(results);
        }
        List<List<SearchResult>> results = index.searchBatch(queries, k, SEARCH_PARAMETERS);
        return checksumBatch(results);
    }

    private static ProfileSummary profileRawNative(
            float[][] vectors,
            long[] ids,
            ScenarioSpec scenario,
            Config config
    ) {
        PackedDataset dataset = packVectors(vectors, ids);
        PackedQueries packedQueries = packQueries(scenario.queries());
        int resultCount = scenario.queries().length * config.k();
        ByteBuffer outputIds = directBuffer((long) resultCount * Long.BYTES);
        ByteBuffer outputScores = directBuffer((long) resultCount * Float.BYTES);
        ByteBuffer timingBuffer = directBuffer(4L * Double.BYTES);
        LongBuffer outputIdsView = outputIds.asLongBuffer();
        FloatBuffer outputScoresView = outputScores.asFloatBuffer();
        DoubleBuffer timingView = timingBuffer.asDoubleBuffer();

        long handle = NativeBindings.createCudaIndex(dataset.vectors(), dataset.ids(), vectors.length, vectors[0].length);
        try {
            warmUpRaw(handle, packedQueries.queries(), outputIds, outputScores, timingBuffer, scenario.queries().length, vectors[0].length, config);

            double[] endToEndMicros = new double[config.measurementIterations()];
            double[] h2dMillis = new double[config.measurementIterations()];
            double[] kernelMillis = new double[config.measurementIterations()];
            double[] d2hMillis = new double[config.measurementIterations()];
            double[] totalMillis = new double[config.measurementIterations()];
            long resultChecksum = 0L;
            double scoreChecksum = 0.0;

            for (int i = 0; i < config.measurementIterations(); i++) {
                clearForWrite(outputIds);
                clearForWrite(outputScores);
                clearForWrite(timingBuffer);

                long start = System.nanoTime();
                NativeBindings.searchCuda(
                        handle,
                        packedQueries.queries(),
                        scenario.queries().length,
                        vectors[0].length,
                        config.k(),
                        DOT_PRODUCT_METRIC_CODE,
                        outputIds,
                        outputScores,
                        timingBuffer
                );
                long end = System.nanoTime();

                endToEndMicros[i] = (end - start) / 1_000.0;
                h2dMillis[i] = timingView.get(0);
                kernelMillis[i] = timingView.get(1);
                d2hMillis[i] = timingView.get(2);
                totalMillis[i] = timingView.get(3);

                SearchOutcome outcome = checksumBuffers(outputIdsView, outputScoresView, resultCount);
                resultChecksum += outcome.idChecksum();
                scoreChecksum += outcome.scoreChecksum();
            }

            return new ProfileSummary(
                    summarize(endToEndMicros),
                    summarize(h2dMillis),
                    summarize(kernelMillis),
                    summarize(d2hMillis),
                    summarize(totalMillis),
                    resultChecksum,
                    scoreChecksum
            );
        } finally {
            NativeBindings.destroyIndex(handle);
        }
    }

    private static void warmUpRaw(
            long handle,
            ByteBuffer queryBuffer,
            ByteBuffer outputIds,
            ByteBuffer outputScores,
            ByteBuffer timingBuffer,
            int queryCount,
            int dimensions,
            Config config
    ) {
        for (int i = 0; i < config.warmupIterations(); i++) {
            clearForWrite(outputIds);
            clearForWrite(outputScores);
            clearForWrite(timingBuffer);
            NativeBindings.searchCuda(
                    handle,
                    queryBuffer,
                    queryCount,
                    dimensions,
                    config.k(),
                    DOT_PRODUCT_METRIC_CODE,
                    outputIds,
                    outputScores,
                    timingBuffer
            );
        }
    }

    private static SearchOutcome checksumSingle(List<SearchResult> results) {
        long idChecksum = 0L;
        double scoreChecksum = 0.0;
        for (SearchResult result : results) {
            idChecksum += result.id();
            scoreChecksum += result.score();
        }
        return new SearchOutcome(idChecksum, scoreChecksum);
    }

    private static SearchOutcome checksumBatch(List<List<SearchResult>> batchResults) {
        long idChecksum = 0L;
        double scoreChecksum = 0.0;
        for (List<SearchResult> results : batchResults) {
            SearchOutcome outcome = checksumSingle(results);
            idChecksum += outcome.idChecksum();
            scoreChecksum += outcome.scoreChecksum();
        }
        return new SearchOutcome(idChecksum, scoreChecksum);
    }

    private static SearchOutcome checksumBuffers(LongBuffer ids, FloatBuffer scores, int resultCount) {
        long idChecksum = 0L;
        double scoreChecksum = 0.0;
        for (int i = 0; i < resultCount; i++) {
            idChecksum += ids.get(i);
            scoreChecksum += scores.get(i);
        }
        return new SearchOutcome(idChecksum, scoreChecksum);
    }

    private static PackedDataset packVectors(float[][] vectors, long[] ids) {
        int vectorCount = vectors.length;
        int dimensions = vectors[0].length;

        ByteBuffer vectorsBuffer = directBuffer((long) vectorCount * dimensions * Float.BYTES);
        FloatBuffer floatView = vectorsBuffer.asFloatBuffer();
        for (float[] vector : vectors) {
            floatView.put(vector);
        }

        ByteBuffer idsBuffer = directBuffer((long) vectorCount * Long.BYTES);
        idsBuffer.asLongBuffer().put(ids);
        return new PackedDataset(vectorsBuffer, idsBuffer);
    }

    private static PackedQueries packQueries(float[][] queries) {
        int queryCount = queries.length;
        int dimensions = queries[0].length;
        ByteBuffer buffer = directBuffer((long) queryCount * dimensions * Float.BYTES);
        FloatBuffer floatView = buffer.asFloatBuffer();
        for (float[] query : queries) {
            floatView.put(query);
        }
        return new PackedQueries(buffer);
    }

    private static ByteBuffer directBuffer(long bytes) {
        if (bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Requested direct buffer exceeds Java limits");
        }
        return ByteBuffer.allocateDirect((int) bytes).order(ByteOrder.nativeOrder());
    }

    private static void clearForWrite(ByteBuffer buffer) {
        buffer.position(0);
        buffer.limit(buffer.capacity());
    }

    private static LatencySummary summarize(double[] values) {
        double[] sorted = Arrays.copyOf(values, values.length);
        Arrays.sort(sorted);
        double sum = 0.0;
        for (double value : sorted) {
            sum += value;
        }
        return new LatencySummary(
                sorted[0],
                percentile(sorted, 0.50),
                percentile(sorted, 0.95),
                percentile(sorted, 0.99),
                sum / sorted.length,
                sorted[sorted.length - 1]
        );
    }

    private static double percentile(double[] sorted, double percentile) {
        if (sorted.length == 1) {
            return sorted[0];
        }
        double index = percentile * (sorted.length - 1);
        int lower = (int) Math.floor(index);
        int upper = (int) Math.ceil(index);
        if (lower == upper) {
            return sorted[lower];
        }
        double fraction = index - lower;
        return sorted[lower] + ((sorted[upper] - sorted[lower]) * fraction);
    }

    private static void printSummary(String pathName, String scenarioName, ProfileSummary summary) {
        System.out.printf(
                Locale.US,
                "summary path=%s scenario=%s end_to_end_avg_us=%.3f end_to_end_p50_us=%.3f end_to_end_p95_us=%.3f end_to_end_p99_us=%.3f kernel_avg_ms=%.3f h2d_avg_ms=%.3f d2h_avg_ms=%.3f native_total_avg_ms=%.3f checksum_ids=%d checksum_scores=%.6f%n",
                pathName,
                scenarioName,
                summary.endToEndMicros().average(),
                summary.endToEndMicros().p50(),
                summary.endToEndMicros().p95(),
                summary.endToEndMicros().p99(),
                summary.kernelMillis().average(),
                summary.hostToDeviceMillis().average(),
                summary.deviceToHostMillis().average(),
                summary.nativeTotalMillis().average(),
                summary.resultChecksum(),
                summary.scoreChecksum()
        );
    }

    private static float[][] randomVectors(int count, int dimensions, long seed) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimensions];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = (random.nextFloat() * 2.0f) - 1.0f;
            }
        }
        return vectors;
    }

    private static long[] sequentialIds(int count) {
        long[] ids = new long[count];
        for (int i = 0; i < count; i++) {
            ids[i] = i + 1L;
        }
        return ids;
    }

    private record ScenarioSpec(String name, float[][] queries) {
    }

    private record SearchOutcome(long idChecksum, double scoreChecksum) {
    }

    private record PackedDataset(ByteBuffer vectors, ByteBuffer ids) {
    }

    private record PackedQueries(ByteBuffer queries) {
    }

    private record ProfileSummary(
            LatencySummary endToEndMicros,
            LatencySummary hostToDeviceMillis,
            LatencySummary kernelMillis,
            LatencySummary deviceToHostMillis,
            LatencySummary nativeTotalMillis,
            long resultChecksum,
            double scoreChecksum
    ) {
    }

    private record LatencySummary(
            double minimum,
            double p50,
            double p95,
            double p99,
            double average,
            double maximum
    ) {
    }

    private record Config(
            int vectorCount,
            int dimensions,
            int k,
            int warmupIterations,
            int measurementIterations,
            int smallQueryCount,
            int batchQueryCount
    ) {
        private static Config parse(String[] args) {
            Map<String, String> options = new LinkedHashMap<>();
            for (int i = 0; i < args.length; i += 2) {
                if (i + 1 >= args.length) {
                    throw usage("missing value for option '" + args[i] + "'");
                }
                options.put(args[i], args[i + 1]);
            }

            return new Config(
                    positiveInt(options, "--vectors", 10_000),
                    positiveInt(options, "--dimensions", 128),
                    positiveInt(options, "--k", 10),
                    positiveInt(options, "--warmup", 10),
                    positiveInt(options, "--iterations", 30),
                    positiveInt(options, "--small-queries", 1),
                    positiveInt(options, "--batch-queries", 32)
            );
        }

        private static int positiveInt(Map<String, String> options, String key, int defaultValue) {
            String value = options.get(key);
            if (value == null) {
                return defaultValue;
            }
            try {
                int parsed = Integer.parseInt(value);
                if (parsed <= 0) {
                    throw usage(key + " must be positive");
                }
                return parsed;
            } catch (NumberFormatException ex) {
                throw usage(key + " must be an integer");
            }
        }

        private static IllegalArgumentException usage(String message) {
            return new IllegalArgumentException(message + System.lineSeparator()
                    + "Usage: java -cp vectorforge-benchmarks/target/vectorforge-benchmarks.jar "
                    + "com.vectorforge.benchmarks.CudaBackendProfileRunner "
                    + "[--vectors 10000 --dimensions 128 --k 10 --warmup 10 --iterations 30 "
                    + "--small-queries 1 --batch-queries 32]");
        }
    }
}

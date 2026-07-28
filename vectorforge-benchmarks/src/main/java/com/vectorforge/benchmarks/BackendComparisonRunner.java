package com.vectorforge.benchmarks;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;
import com.vectorforge.cpu.CpuBruteForceIndex;
import com.vectorforge.gpu.CudaBruteForceIndex;
import com.vectorforge.gpu.CuvsVectorIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Small end-to-end comparison harness. It is a smoke/profile tool, not a JMH benchmark.
 */
public final class BackendComparisonRunner {

    private BackendComparisonRunner() {
    }

    public static void main(String[] args) {
        Config config = Config.parse(args);
        float[][] vectors = randomVectors(config.vectors(), config.dimensions(), 42L);
        float[][] queries = randomVectors(config.queries(), config.dimensions(), 84L);
        long[] ids = sequentialIds(config.vectors());
        SearchParameters parameters = new SearchParameters(DistanceMetric.DOT_PRODUCT);

        System.out.println("VectorForge backend comparison");
        System.out.printf(Locale.US,
                "vectors=%d dimensions=%d queries=%d k=%d warmup=%d iterations=%d metric=DOT_PRODUCT%n",
                config.vectors(), config.dimensions(), config.queries(), config.k(),
                config.warmup(), config.iterations());
        System.out.println("measurement=end-to-end Java searchBatch wall clock; build reported separately");

        List<List<SearchResult>> expected;
        try (CpuBruteForceIndex cpu = new CpuBruteForceIndex()) {
            cpu.build(vectors, ids);
            expected = cpu.searchBatch(queries, config.k(), parameters);
        }

        runBackend("cpu", CpuBruteForceIndex::new, vectors, ids, queries, parameters, expected, config);
        if (CudaBruteForceIndex.isCudaAvailable()) {
            runBackend("cuda", CudaBruteForceIndex::new, vectors, ids, queries, parameters, expected, config);
        } else {
            System.out.println("backend=cuda skipped=true reason=unavailable");
        }
        if (CuvsVectorIndex.isCuvsAvailable()) {
            runBackend("cuvs", CuvsVectorIndex::new, vectors, ids, queries, parameters, expected, config);
        } else {
            System.out.println("backend=cuvs skipped=true reason=unavailable");
        }
    }

    private static void runBackend(
            String name,
            Supplier<? extends VectorIndex> factory,
            float[][] vectors,
            long[] ids,
            float[][] queries,
            SearchParameters parameters,
            List<List<SearchResult>> expected,
            Config config
    ) {
        try (VectorIndex index = factory.get()) {
            long buildStarted = System.nanoTime();
            index.build(vectors, ids);
            double buildMillis = elapsedMillis(buildStarted);

            List<List<SearchResult>> first = searchBatch(index, queries, config.k(), parameters);
            boolean idsMatch = resultIds(first).equals(resultIds(expected));
            for (int i = 0; i < config.warmup(); i++) {
                searchBatch(index, queries, config.k(), parameters);
            }

            double[] samplesMicros = new double[config.iterations()];
            for (int i = 0; i < samplesMicros.length; i++) {
                long started = System.nanoTime();
                searchBatch(index, queries, config.k(), parameters);
                samplesMicros[i] = elapsedMicros(started);
            }
            double[] sorted = Arrays.copyOf(samplesMicros, samplesMicros.length);
            Arrays.sort(sorted);

            System.out.printf(Locale.US,
                    "backend=%s build_ms=%.3f ids_match_cpu=%s p50_us=%.3f p95_us=%.3f p99_us=%.3f avg_us=%.3f%n",
                    name, buildMillis, idsMatch, percentile(sorted, 0.50), percentile(sorted, 0.95),
                    percentile(sorted, 0.99), Arrays.stream(samplesMicros).average().orElseThrow());
            System.out.println("backend=" + name + " raw_batch_us=" + formatSamples(samplesMicros));
        }
    }

    private static List<List<SearchResult>> searchBatch(
            VectorIndex index,
            float[][] queries,
            int k,
            SearchParameters parameters
    ) {
        if (index instanceof CpuBruteForceIndex cpu) {
            return cpu.searchBatch(queries, k, parameters);
        }
        if (index instanceof CudaBruteForceIndex cuda) {
            return cuda.searchBatch(queries, k, parameters);
        }
        if (index instanceof CuvsVectorIndex cuvs) {
            return cuvs.searchBatch(queries, k, parameters);
        }
        ArrayList<List<SearchResult>> results = new ArrayList<>(queries.length);
        for (float[] query : queries) {
            results.add(index.search(query, k, parameters));
        }
        return List.copyOf(results);
    }

    private static List<List<Long>> resultIds(List<List<SearchResult>> results) {
        return results.stream()
                .map(query -> query.stream().map(SearchResult::id).toList())
                .toList();
    }

    private static double percentile(double[] sorted, double percentile) {
        int index = (int) Math.ceil(percentile * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
    }

    private static String formatSamples(double[] samples) {
        StringBuilder output = new StringBuilder("[");
        for (int i = 0; i < samples.length; i++) {
            if (i > 0) {
                output.append(',');
            }
            output.append(String.format(Locale.US, "%.3f", samples[i]));
        }
        return output.append(']').toString();
    }

    private static double elapsedMicros(long started) {
        return (System.nanoTime() - started) / 1_000.0;
    }

    private static double elapsedMillis(long started) {
        return (System.nanoTime() - started) / 1_000_000.0;
    }

    private static float[][] randomVectors(int count, int dimensions, long seed) {
        Random random = new Random(seed);
        float[][] values = new float[count][dimensions];
        for (float[] value : values) {
            for (int i = 0; i < value.length; i++) {
                value[i] = (random.nextFloat() * 2.0F) - 1.0F;
            }
        }
        return values;
    }

    private static long[] sequentialIds(int count) {
        long[] ids = new long[count];
        for (int i = 0; i < count; i++) {
            ids[i] = i + 1L;
        }
        return ids;
    }

    private record Config(int vectors, int dimensions, int queries, int k, int warmup, int iterations) {
        private static Config parse(String[] args) {
            Map<String, String> values = new LinkedHashMap<>();
            for (int i = 0; i < args.length; i += 2) {
                if (!args[i].startsWith("--") || i + 1 >= args.length) {
                    throw usage("options require --name value pairs");
                }
                values.put(args[i], args[i + 1]);
            }
            Config config = new Config(
                    positive(values, "--vectors", 10_000),
                    positive(values, "--dimensions", 128),
                    positive(values, "--queries", 16),
                    positive(values, "--k", 10),
                    nonNegative(values, "--warmup", 5),
                    positive(values, "--iterations", 10)
            );
            if (config.k() > config.vectors()) {
                throw usage("--k must be <= --vectors");
            }
            return config;
        }

        private static int positive(Map<String, String> values, String key, int fallback) {
            int value = parse(values, key, fallback);
            if (value <= 0) {
                throw usage(key + " must be positive");
            }
            return value;
        }

        private static int nonNegative(Map<String, String> values, String key, int fallback) {
            int value = parse(values, key, fallback);
            if (value < 0) {
                throw usage(key + " must be non-negative");
            }
            return value;
        }

        private static int parse(Map<String, String> values, String key, int fallback) {
            try {
                return Integer.parseInt(values.getOrDefault(key, Integer.toString(fallback)));
            } catch (NumberFormatException exception) {
                throw usage(key + " must be an integer");
            }
        }

        private static IllegalArgumentException usage(String message) {
            return new IllegalArgumentException(message + System.lineSeparator()
                    + "Usage: BackendComparisonRunner [--vectors 10000] [--dimensions 128] "
                    + "[--queries 16] [--k 10] [--warmup 5] [--iterations 10]");
        }
    }
}

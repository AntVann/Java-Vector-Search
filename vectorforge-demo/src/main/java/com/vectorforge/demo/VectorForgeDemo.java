package com.vectorforge.demo;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;
import com.vectorforge.cpu.CpuBruteForceIndex;
import com.vectorforge.gpu.CudaBruteForceIndex;
import com.vectorforge.gpu.CudaSearchTimings;
import com.vectorforge.gpu.CuvsVectorIndex;
import com.vectorforge.nativeindex.NativeBruteForceIndex;

import java.util.LinkedHashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

public final class VectorForgeDemo {

    private VectorForgeDemo() {
    }

    public static void main(String[] args) {
        DemoConfig config = DemoConfig.parse(args);

        float[][] vectors = generateVectors(config.vectors(), config.dimensions(), 42L);
        long[] ids = generateIds(config.vectors());
        float[][] queries = generateVectors(config.queries(), config.dimensions(), 84L);

        try (VectorIndex index = createIndex(config.backend())) {
            long buildStart = System.nanoTime();
            index.build(vectors, ids);
            long buildNanos = System.nanoTime() - buildStart;

            SearchParameters parameters = new SearchParameters(config.metric());
            long searchStart = System.nanoTime();
            List<List<SearchResult>> allResults = searchAll(index, queries, config.k(), parameters);
            long searchNanos = System.nanoTime() - searchStart;

            printSummary(config, buildNanos, searchNanos, allResults, cudaTimings(index));
        } catch (IllegalArgumentException | IllegalStateException ex) {
            System.err.println(ex.getMessage());
            System.exit(2);
        }
    }

    private static VectorIndex createIndex(String backend) {
        return switch (backend) {
            case "cpu" -> new CpuBruteForceIndex();
            case "native" -> new NativeBruteForceIndex();
            case "cuda" -> new CudaBruteForceIndex();
            case "cuvs" -> new CuvsVectorIndex();
            default -> throw new IllegalArgumentException(
                    "Unsupported backend '" + backend + "'. Supported backends: cpu, native, cuda, cuvs"
            );
        };
    }

    private static List<List<SearchResult>> searchAll(
            VectorIndex index,
            float[][] queries,
            int k,
            SearchParameters parameters
    ) {
        if (index instanceof CpuBruteForceIndex cpu) {
            return cpu.searchBatch(queries, k, parameters);
        }
        if (index instanceof NativeBruteForceIndex nativeIndex) {
            return nativeIndex.searchBatch(queries, k, parameters);
        }
        if (index instanceof CudaBruteForceIndex cudaIndex) {
            return cudaIndex.searchBatch(queries, k, parameters);
        }
        if (index instanceof CuvsVectorIndex cuvsIndex) {
            return cuvsIndex.searchBatch(queries, k, parameters);
        }

        ArrayList<List<SearchResult>> results = new ArrayList<>(queries.length);
        for (float[] query : queries) {
            results.add(index.search(query, k, parameters));
        }
        return List.copyOf(results);
    }

    private static void printSummary(
            DemoConfig config,
            long buildNanos,
            long searchNanos,
            List<List<SearchResult>> allResults,
            CudaSearchTimings cudaTimings
    ) {
        System.out.println("VectorForge Demo");
        System.out.println("backend=" + config.backend());
        System.out.println("metric=" + config.metric());
        System.out.println("vectors=" + config.vectors());
        System.out.println("dimensions=" + config.dimensions());
        System.out.println("queries=" + config.queries());
        System.out.println("k=" + config.k());
        System.out.printf(Locale.US, "build_ms=%.3f%n", buildNanos / 1_000_000.0);
        System.out.printf(Locale.US, "search_ms=%.3f%n", searchNanos / 1_000_000.0);
        System.out.printf(Locale.US, "avg_query_us=%.3f%n", searchNanos / 1_000.0 / config.queries());

        if (cudaTimings != null) {
            System.out.printf(Locale.US, "cuda_h2d_ms=%.3f%n", cudaTimings.hostToDeviceMillis());
            System.out.printf(Locale.US, "cuda_kernel_ms=%.3f%n", cudaTimings.kernelMillis());
            System.out.printf(Locale.US, "cuda_d2h_ms=%.3f%n", cudaTimings.deviceToHostMillis());
            System.out.printf(Locale.US, "cuda_total_ms=%.3f%n", cudaTimings.totalMillis());
        }

        if (!allResults.isEmpty()) {
            System.out.println("sample_results_query_0=");
            for (SearchResult result : allResults.getFirst()) {
                System.out.printf(Locale.US, "  id=%d score=%.6f%n", result.id(), result.score());
            }
        }
    }

    private static CudaSearchTimings cudaTimings(VectorIndex index) {
        return index instanceof CudaBruteForceIndex cudaIndex ? cudaIndex.lastSearchTimings() : null;
    }

    private static float[][] generateVectors(int count, int dimensions, long seed) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimensions];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = (random.nextFloat() * 2.0f) - 1.0f;
            }
        }
        return vectors;
    }

    private static long[] generateIds(int count) {
        long[] ids = new long[count];
        for (int i = 0; i < count; i++) {
            ids[i] = i + 1L;
        }
        return ids;
    }

    private record DemoConfig(
            String backend,
            int vectors,
            int dimensions,
            int queries,
            int k,
            DistanceMetric metric
    ) {
        private static DemoConfig parse(String[] args) {
            Map<String, String> options = new LinkedHashMap<>();
            for (int i = 0; i < args.length; i += 2) {
                if (!args[i].startsWith("--")) {
                    throw usage("expected option name but found '" + args[i] + "'");
                }
                if (i + 1 >= args.length) {
                    throw usage("missing value for option '" + args[i] + "'");
                }
                options.put(args[i], args[i + 1]);
            }

            String backend = options.getOrDefault("--backend", "cpu").toLowerCase(Locale.ROOT);
            int vectors = positiveInt(options, "--vectors", 10_000);
            int dimensions = positiveInt(options, "--dimensions", 128);
            int queries = positiveInt(options, "--queries", 10);
            int k = positiveInt(options, "--k", 10);
            DistanceMetric metric = parseMetric(options.getOrDefault("--metric", "euclidean"));

            if (k > vectors) {
                throw usage("--k must be <= --vectors");
            }

            return new DemoConfig(backend, vectors, dimensions, queries, k, metric);
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

        private static DistanceMetric parseMetric(String raw) {
            return switch (raw.toLowerCase(Locale.ROOT)) {
                case "euclidean" -> DistanceMetric.EUCLIDEAN;
                case "cosine" -> DistanceMetric.COSINE;
                case "dot_product", "dot-product", "dot" -> DistanceMetric.DOT_PRODUCT;
                default -> throw usage("unsupported metric '" + raw + "'");
            };
        }

        private static IllegalArgumentException usage(String message) {
            return new IllegalArgumentException(message + System.lineSeparator()
                    + "Usage: java -jar vectorforge-demo.jar "
                    + "--backend cpu|native|cuda|cuvs --vectors 100000 --dimensions 384 --queries 100 --k 10 [--metric euclidean|cosine|dot_product]");
        }
    }
}

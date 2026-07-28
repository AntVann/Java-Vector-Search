package com.vectorforge.cpu;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.IndexMetrics;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CpuBruteForceIndexTest {

    @Test
    void searchesKnownDatasetWithEuclideanDistance() {
        try (CpuBruteForceIndex index = new CpuBruteForceIndex()) {
            index.build(
                    new float[][]{
                            {0.0f, 0.0f},
                            {1.0f, 1.0f},
                            {2.0f, 2.0f},
                            {4.0f, 4.0f}
                    },
                    new long[]{10L, 20L, 30L, 40L}
            );

            List<SearchResult> results = index.search(
                    new float[]{1.1f, 1.1f},
                    3,
                    new SearchParameters(DistanceMetric.EUCLIDEAN)
            );

            assertEquals(List.of(20L, 30L, 10L), extractIds(results));
            assertFloatEquals(0.02f, results.get(0).score());
        }
    }

    @Test
    void searchesKnownDatasetWithCosineAndDotProduct() {
        try (CpuBruteForceIndex index = new CpuBruteForceIndex()) {
            index.build(
                    new float[][]{
                            {1.0f, 0.0f},
                            {0.0f, 1.0f},
                            {1.0f, 1.0f}
                    },
                    new long[]{1L, 2L, 3L}
            );

            List<SearchResult> cosine = index.search(
                    new float[]{1.0f, 0.0f},
                    3,
                    new SearchParameters(DistanceMetric.COSINE)
            );
            List<SearchResult> dotProduct = index.search(
                    new float[]{1.0f, 0.0f},
                    3,
                    new SearchParameters(DistanceMetric.DOT_PRODUCT)
            );

            assertEquals(List.of(1L, 3L, 2L), extractIds(cosine));
            assertEquals(List.of(1L, 3L, 2L), extractIds(dotProduct));
            assertFloatEquals(1.0f, cosine.get(0).score());
            assertFloatEquals(1.0f, dotProduct.get(0).score());
        }
    }

    @Test
    void supportsDeterministicTieOrdering() {
        try (CpuBruteForceIndex index = new CpuBruteForceIndex()) {
            index.build(
                    new float[][]{
                            {1.0f, 1.0f},
                            {1.0f, 1.0f},
                            {1.0f, 1.0f}
                    },
                    new long[]{9L, 3L, 5L}
            );

            List<SearchResult> results = index.search(
                    new float[]{1.0f, 1.0f},
                    3,
                    new SearchParameters(DistanceMetric.EUCLIDEAN)
            );

            assertEquals(List.of(3L, 5L, 9L), extractIds(results));
            assertTrue(results.stream().allMatch(result -> result.score() == 0.0f));
        }
    }

    @Test
    void matchesReferenceImplementationAcrossMetrics() {
        Random random = new Random(17L);
        float[][] vectors = randomVectors(random, 64, 16);
        long[] ids = sequentialIds(64);
        float[] query = randomVector(random, 16);

        try (CpuBruteForceIndex index = new CpuBruteForceIndex()) {
            index.build(vectors, ids);

            for (DistanceMetric metric : DistanceMetric.values()) {
                List<SearchResult> actual = index.search(query, 10, new SearchParameters(metric));
                List<SearchResult> expected = referenceSearch(vectors, ids, query, 10, metric);
                assertIterableEquals(expected, actual, "metric=" + metric);
            }
        }
    }

    @Test
    void rejectsInvalidBuildInputs() {
        try (CpuBruteForceIndex index = new CpuBruteForceIndex()) {
            assertThrows(NullPointerException.class, () -> index.build(null, new long[]{1L}));
            assertThrows(NullPointerException.class, () -> index.build(new float[][]{{1.0f}}, null));
            assertThrows(IllegalArgumentException.class, () -> index.build(new float[0][], new long[0]));
            assertThrows(IllegalArgumentException.class, () -> index.build(new float[][]{{1.0f}}, new long[]{1L, 2L}));
            assertThrows(IllegalArgumentException.class, () -> index.build(new float[][]{{1.0f}, null}, new long[]{1L, 2L}));
            assertThrows(IllegalArgumentException.class, () -> index.build(new float[][]{{1.0f}, {1.0f, 2.0f}}, new long[]{1L, 2L}));
            assertThrows(IllegalArgumentException.class, () -> index.build(new float[][]{{1.0f}, {2.0f}}, new long[]{7L, 7L}));
            assertThrows(IllegalArgumentException.class, () -> index.build(new float[][]{{Float.NaN}}, new long[]{1L}));
            assertThrows(IllegalArgumentException.class, () -> index.build(new float[][]{{Float.POSITIVE_INFINITY}}, new long[]{1L}));
        }
    }

    @Test
    void rejectsInvalidSearchInputsAndSearchBeforeBuild() {
        try (CpuBruteForceIndex index = new CpuBruteForceIndex()) {
            assertThrows(IllegalStateException.class, () -> index.search(
                    new float[]{1.0f},
                    1,
                    new SearchParameters(DistanceMetric.EUCLIDEAN)
            ));

            index.build(new float[][]{{1.0f, 2.0f}}, new long[]{1L});

            assertThrows(NullPointerException.class, () -> index.search(null, 1, new SearchParameters(DistanceMetric.EUCLIDEAN)));
            assertThrows(NullPointerException.class, () -> index.search(new float[]{1.0f, 2.0f}, 1, null));
            assertThrows(IllegalArgumentException.class, () -> index.search(new float[]{1.0f}, 1, new SearchParameters(DistanceMetric.EUCLIDEAN)));
            assertThrows(IllegalArgumentException.class, () -> index.search(new float[]{1.0f, 2.0f}, 0, new SearchParameters(DistanceMetric.EUCLIDEAN)));
            assertThrows(IllegalArgumentException.class, () -> index.search(new float[]{1.0f, 2.0f}, 2, new SearchParameters(DistanceMetric.EUCLIDEAN)));
            assertThrows(IllegalArgumentException.class, () -> index.search(new float[]{Float.NEGATIVE_INFINITY, 2.0f}, 1,
                    new SearchParameters(DistanceMetric.EUCLIDEAN)));
        }
    }

    @Test
    void supportsConcurrentSearchesAfterBuild() throws ExecutionException, InterruptedException {
        float[][] vectors = randomVectors(new Random(29L), 256, 32);
        long[] ids = sequentialIds(256);
        float[] query = randomVector(new Random(31L), 32);

        try (CpuBruteForceIndex index = new CpuBruteForceIndex()) {
            index.build(vectors, ids);
            List<SearchResult> expected = index.search(query, 8, new SearchParameters(DistanceMetric.COSINE));

            ExecutorService executor = Executors.newFixedThreadPool(8);
            try {
                List<Callable<List<SearchResult>>> tasks = new ArrayList<>();
                for (int i = 0; i < 32; i++) {
                    tasks.add(() -> index.search(query, 8, new SearchParameters(DistanceMetric.COSINE)));
                }

                List<Future<List<SearchResult>>> futures = executor.invokeAll(tasks);
                for (Future<List<SearchResult>> future : futures) {
                    assertEquals(expected, future.get());
                }
            } finally {
                executor.shutdownNow();
            }
        }
    }

    @Test
    void closeIsIdempotentAndPreventsFurtherUse() {
        CpuBruteForceIndex index = new CpuBruteForceIndex();
        index.build(new float[][]{{1.0f, 0.0f}}, new long[]{1L});

        IndexMetrics builtMetrics = index.metrics();
        assertTrue(builtMetrics.built());
        assertFalse(builtMetrics.closed());
        assertEquals(1L, builtMetrics.vectorCount());
        assertEquals(2, builtMetrics.dimensions());

        index.close();
        assertDoesNotThrow(index::close);

        IndexMetrics closedMetrics = index.metrics();
        assertFalse(closedMetrics.built());
        assertTrue(closedMetrics.closed());

        assertThrows(IllegalStateException.class, () -> index.search(
                new float[]{1.0f, 0.0f},
                1,
                new SearchParameters(DistanceMetric.DOT_PRODUCT)
        ));
        assertThrows(IllegalStateException.class, () -> index.build(new float[][]{{1.0f, 0.0f}}, new long[]{1L}));
    }

    @Test
    void supportsBatchSearch() {
        try (CpuBruteForceIndex index = new CpuBruteForceIndex()) {
            index.build(
                    new float[][]{
                            {1.0f, 0.0f},
                            {0.0f, 1.0f},
                            {1.0f, 1.0f}
                    },
                    new long[]{1L, 2L, 3L}
            );

            List<List<SearchResult>> results = index.searchBatch(
                    new float[][]{
                            {1.0f, 0.0f},
                            {0.0f, 1.0f}
                    },
                    2,
                    new SearchParameters(DistanceMetric.DOT_PRODUCT)
            );

            assertEquals(2, results.size());
            assertEquals(List.of(1L, 3L), extractIds(results.get(0)));
            assertEquals(List.of(2L, 3L), extractIds(results.get(1)));
        }
    }

    @Test
    void batchSearchUsesTheSharedValidationContract() {
        try (CpuBruteForceIndex index = new CpuBruteForceIndex()) {
            assertThrows(IllegalStateException.class, () -> index.searchBatch(
                    new float[][]{{1.0f}}, 1, new SearchParameters(DistanceMetric.DOT_PRODUCT)));
            index.build(new float[][]{{1.0f}}, new long[]{1L});
            assertThrows(IllegalArgumentException.class, () -> index.searchBatch(
                    new float[0][], 1, new SearchParameters(DistanceMetric.DOT_PRODUCT)));
            assertThrows(NullPointerException.class, () -> index.searchBatch(
                    new float[][]{{1.0f}}, 1, null));
            assertThrows(NullPointerException.class, () -> index.searchBatch(
                    new float[][]{null}, 1, new SearchParameters(DistanceMetric.DOT_PRODUCT)));
            assertThrows(IllegalArgumentException.class, () -> index.searchBatch(
                    new float[][]{{1.0f}}, 0, new SearchParameters(DistanceMetric.DOT_PRODUCT)));
        }
    }

    private static List<SearchResult> referenceSearch(float[][] vectors, long[] ids, float[] query, int k, DistanceMetric metric) {
        float queryNorm = metric == DistanceMetric.COSINE ? norm(query) : 0.0f;
        ArrayList<SearchResult> results = new ArrayList<>(vectors.length);

        for (int i = 0; i < vectors.length; i++) {
            float score = switch (metric) {
                case EUCLIDEAN -> squaredEuclideanDistance(vectors[i], query);
                case DOT_PRODUCT -> dotProduct(vectors[i], query);
                case COSINE -> cosineSimilarity(vectors[i], query, norm(vectors[i]), queryNorm);
            };
            results.add(new SearchResult(ids[i], score));
        }

        Comparator<SearchResult> comparator = switch (metric) {
            case EUCLIDEAN -> Comparator.comparing(SearchResult::score).thenComparing(SearchResult::id);
            case COSINE, DOT_PRODUCT -> Comparator.comparing(SearchResult::score).reversed().thenComparing(SearchResult::id);
        };
        results.sort(comparator);
        return List.copyOf(results.subList(0, k));
    }

    private static float[][] randomVectors(Random random, int count, int dimensions) {
        float[][] vectors = new float[count][dimensions];
        for (int i = 0; i < count; i++) {
            vectors[i] = randomVector(random, dimensions);
        }
        return vectors;
    }

    private static float[] randomVector(Random random, int dimensions) {
        float[] vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            vector[i] = (random.nextFloat() * 2.0f) - 1.0f;
        }
        return vector;
    }

    private static long[] sequentialIds(int count) {
        long[] ids = new long[count];
        for (int i = 0; i < count; i++) {
            ids[i] = i + 1L;
        }
        return ids;
    }

    private static List<Long> extractIds(List<SearchResult> results) {
        return results.stream().map(SearchResult::id).toList();
    }

    private static float squaredEuclideanDistance(float[] vector, float[] query) {
        float sum = 0.0f;
        for (int i = 0; i < vector.length; i++) {
            float delta = vector[i] - query[i];
            sum += delta * delta;
        }
        return sum;
    }

    private static float dotProduct(float[] vector, float[] query) {
        float sum = 0.0f;
        for (int i = 0; i < vector.length; i++) {
            sum += vector[i] * query[i];
        }
        return sum;
    }

    private static float cosineSimilarity(float[] vector, float[] query, float vectorNorm, float queryNorm) {
        if (vectorNorm == 0.0f || queryNorm == 0.0f) {
            return 0.0f;
        }
        return dotProduct(vector, query) / (vectorNorm * queryNorm);
    }

    private static float norm(float[] vector) {
        float sum = 0.0f;
        for (float value : vector) {
            sum += value * value;
        }
        return (float) Math.sqrt(sum);
    }

    private static void assertFloatEquals(float expected, float actual) {
        assertEquals(expected, actual, 1.0e-6f);
    }
}

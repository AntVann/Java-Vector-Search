package com.vectorforge.gpu;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.cpu.CpuBruteForceIndex;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CudaBruteForceIndexTest {

    @BeforeAll
    static void requireCudaProfileAndDevice() {
        assumeTrue(Boolean.getBoolean("vectorforge.cuda.enabled"), "CUDA tests require the -Pcuda profile");
        assertTrue(com.vectorforge.nativeindex.NativeBindings.isCudaCompiled(),
                "The -Pcuda profile must compile CUDA support");
        assumeTrue(CudaBruteForceIndex.isCudaAvailable(), "CUDA backend requires a usable GPU");
    }

    @Test
    void matchesCpuReferenceAcrossDimensionsDatasetsAndBatchSizes() {
        int[] dimensions = {8, 32, 96};
        int[] vectorCounts = {32, 128, 512};
        int[] queryCounts = {1, 4, 8};
        int[] ks = {1, 5, 10};

        long seed = 41L;
        for (int dimension : dimensions) {
            for (int vectorCount : vectorCounts) {
                for (int queryCount : queryCounts) {
                    for (int k : ks) {
                        if (k > vectorCount) {
                            continue;
                        }

                        Random random = new Random(seed++);
                        float[][] vectors = randomVectors(random, vectorCount, dimension);
                        long[] ids = sequentialIds(vectorCount);
                        float[][] queries = randomVectors(random, queryCount, dimension);

                        try (CpuBruteForceIndex cpu = new CpuBruteForceIndex();
                             CudaBruteForceIndex cuda = new CudaBruteForceIndex()) {
                            cpu.build(vectors, ids);
                            cuda.build(vectors, ids);

                            List<List<SearchResult>> expected = cpu.searchBatch(queries, k, new SearchParameters(DistanceMetric.DOT_PRODUCT));
                            List<List<SearchResult>> actual = cuda.searchBatch(queries, k, new SearchParameters(DistanceMetric.DOT_PRODUCT));

                            assertResultsMatch(expected, actual,
                                    "dimension=" + dimension + ", vectorCount=" + vectorCount + ", queryCount=" + queryCount + ", k=" + k);

                            CudaSearchTimings timings = cuda.lastSearchTimings();
                            assertNotNull(timings);
                            assertEquals(queryCount, timings.queryCount());
                            assertEquals(vectorCount, timings.vectorCount());
                            assertEquals(k, timings.k());
                        }
                    }
                }
            }
        }
    }

    @Test
    void matchesDeterministicReferenceDataset() {
        float[][] vectors = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {1.0f, 1.0f, 0.0f},
                {0.5f, 0.5f, 0.0f}
        };
        long[] ids = {11L, 22L, 33L, 44L};
        float[][] queries = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f}
        };

        try (CpuBruteForceIndex cpu = new CpuBruteForceIndex();
             CudaBruteForceIndex cuda = new CudaBruteForceIndex()) {
            cpu.build(vectors, ids);
            cuda.build(vectors, ids);

            List<List<SearchResult>> expected = cpu.searchBatch(queries, 3, new SearchParameters(DistanceMetric.DOT_PRODUCT));
            List<List<SearchResult>> actual = cuda.searchBatch(queries, 3, new SearchParameters(DistanceMetric.DOT_PRODUCT));

            assertResultsMatch(expected, actual, "deterministic dataset");
        }
    }

    @Test
    void rejectsUnsupportedMetricsAndLifecycleMisuse() {
        CudaBruteForceIndex index = new CudaBruteForceIndex();

        assertThrows(IllegalStateException.class, () -> index.search(
                new float[]{1.0f},
                1,
                new SearchParameters(DistanceMetric.DOT_PRODUCT)
        ));

        index.build(new float[][]{{1.0f, 0.0f}, {0.0f, 1.0f}}, new long[]{1L, 2L});

        assertThrows(IllegalArgumentException.class, () -> index.search(
                new float[]{1.0f, 0.0f},
                1,
                new SearchParameters(DistanceMetric.EUCLIDEAN)
        ));
        assertThrows(IllegalArgumentException.class, () -> index.searchBatch(
                new float[][]{{1.0f, 0.0f}},
                1,
                new SearchParameters(DistanceMetric.COSINE)
        ));
        assertThrows(IllegalArgumentException.class, () -> index.search(
                new float[]{1.0f},
                1,
                new SearchParameters(DistanceMetric.DOT_PRODUCT)
        ));

        index.close();
        assertDoesNotThrow(index::close);
        assertThrows(IllegalStateException.class, () -> index.build(new float[][]{{1.0f, 0.0f}}, new long[]{1L}));
    }

    @Test
    void recordsNonNegativeTimings() {
        float[][] vectors = randomVectors(new Random(99L), 256, 64);
        long[] ids = sequentialIds(256);
        float[][] queries = randomVectors(new Random(101L), 8, 64);

        try (CudaBruteForceIndex cuda = new CudaBruteForceIndex()) {
            cuda.build(vectors, ids);
            cuda.searchBatch(queries, 10, new SearchParameters(DistanceMetric.DOT_PRODUCT));

            CudaSearchTimings timings = cuda.lastSearchTimings();
            assertNotNull(timings);
            assertEquals(8, timings.queryCount());
            assertEquals(256, timings.vectorCount());
            assertEquals(10, timings.k());
            assertTrue(Double.isFinite(timings.totalMillis()) && timings.totalMillis() >= 0.0);
            assertTrue(Double.isFinite(timings.hostToDeviceMillis()) && timings.hostToDeviceMillis() >= 0.0);
            assertTrue(Double.isFinite(timings.kernelMillis()) && timings.kernelMillis() >= 0.0);
            assertTrue(Double.isFinite(timings.deviceToHostMillis()) && timings.deviceToHostMillis() >= 0.0);
        }
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

    private static void assertResultsMatch(List<List<SearchResult>> expected, List<List<SearchResult>> actual, String message) {
        assertEquals(expected.size(), actual.size(), message + ": batch size");
        for (int queryIndex = 0; queryIndex < expected.size(); queryIndex++) {
            List<SearchResult> expectedResults = expected.get(queryIndex);
            List<SearchResult> actualResults = actual.get(queryIndex);
            assertEquals(expectedResults.size(), actualResults.size(), message + ": result count for query " + queryIndex);
            for (int rank = 0; rank < expectedResults.size(); rank++) {
                SearchResult expectedResult = expectedResults.get(rank);
                SearchResult actualResult = actualResults.get(rank);
                assertEquals(expectedResult.id(), actualResult.id(), message + ": id mismatch at query " + queryIndex + ", rank " + rank);
                assertEquals(expectedResult.score(), actualResult.score(), 1.0e-5f,
                        message + ": score mismatch at query " + queryIndex + ", rank " + rank);
            }
        }
    }
}

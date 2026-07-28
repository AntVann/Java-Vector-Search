package com.vectorforge.gpu;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.cpu.CpuBruteForceIndex;
import com.vectorforge.nativeindex.NativeBindings;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CuvsVectorIndexTest {

    @BeforeAll
    static void requireCuvsProfileAndDevice() {
        assumeTrue(Boolean.getBoolean("vectorforge.cuvs.enabled"), "cuVS tests require the -Pcuvs profile");
        assertTrue(NativeBindings.isCuvsCompiled(), "-Pcuvs must produce a cuVS-enabled native library");
        assumeTrue(NativeBindings.getCudaDeviceCount() > 0, "cuVS tests require a usable CUDA device");
    }

    @Test
    void reportsVersionAndLifecycleState() {
        assertFalse(CuvsVectorIndex.cuvsVersion().isBlank());
        CuvsVectorIndex index = new CuvsVectorIndex();
        assertFalse(index.metrics().built());
        index.build(new float[][]{{0, 0}, {1, 1}}, new long[]{10, 20});
        assertTrue(index.metrics().built());
        assertTrue(index.metrics().gpuResident());
        index.close();
        assertDoesNotThrow(index::close);
        assertTrue(index.metrics().closed());
        assertThrows(IllegalStateException.class,
                () -> index.build(new float[][]{{0, 0}}, new long[]{10}));
    }

    @Test
    void exactResultsAndScoresMatchCpuReferenceAcrossMetrics() {
        float[][] vectors = {
                {0.2f, -0.7f, 0.4f},
                {1.1f, 0.3f, -0.2f},
                {-0.4f, 0.8f, 1.3f},
                {0.9f, -0.1f, 0.6f},
                {-1.2f, -0.5f, 0.2f}
        };
        long[] ids = {Long.MAX_VALUE - 7, -9_223_372_036_854_775_000L, 17, -42, 4_294_967_311L};
        float[][] queries = {
                {0.7f, -0.2f, 0.5f},
                {-0.6f, 0.9f, 0.8f}
        };

        try (CpuBruteForceIndex cpu = new CpuBruteForceIndex();
             CuvsVectorIndex cuvs = new CuvsVectorIndex()) {
            cpu.build(vectors, ids);
            cuvs.build(vectors, ids);
            for (DistanceMetric metric : DistanceMetric.values()) {
                SearchParameters parameters = new SearchParameters(metric);
                List<List<SearchResult>> expected = cpu.searchBatch(queries, 4, parameters);
                List<List<SearchResult>> actual = cuvs.searchBatch(queries, 4, parameters);
                assertResultsEqual(expected, actual);
            }
        }
    }

    @Test
    void returnsAllTiedNeighborsWithoutAssumingNativeTieOrder() {
        try (CuvsVectorIndex cuvs = new CuvsVectorIndex()) {
            cuvs.build(
                    new float[][]{{1, 0}, {1, 0}, {0, 1}},
                    new long[]{Long.MAX_VALUE, Long.MIN_VALUE + 1, 77}
            );
            List<SearchResult> results = cuvs.search(
                    new float[]{1, 0}, 2, new SearchParameters(DistanceMetric.DOT_PRODUCT));
            assertEquals(Set.of(Long.MAX_VALUE, Long.MIN_VALUE + 1),
                    Set.of(results.get(0).id(), results.get(1).id()));
            assertEquals(1.0f, results.get(0).score(), 1.0e-5f);
            assertEquals(1.0f, results.get(1).score(), 1.0e-5f);
        }
    }

    @Test
    void survivesRepeatedCreateSearchRebuildAndCloseCycles() {
        for (int iteration = 0; iteration < 20; iteration++) {
            CuvsVectorIndex index = new CuvsVectorIndex();
            index.build(
                    new float[][]{{1, 0}, {0, 1}, {-1, 0}},
                    new long[]{Long.MIN_VALUE + iteration, 0, Long.MAX_VALUE - iteration}
            );
            assertEquals(2, index.search(
                    new float[]{1, 0}, 2, new SearchParameters(DistanceMetric.COSINE)).size());

            index.build(
                    new float[][]{{0, 0}, {2, 2}},
                    new long[]{-100L - iteration, 100L + iteration}
            );
            assertEquals(100L + iteration, index.search(
                    new float[]{1.9f, 2.1f}, 1,
                    new SearchParameters(DistanceMetric.EUCLIDEAN)).getFirst().id());
            index.close();
            assertDoesNotThrow(index::close);
            assertThrows(IllegalStateException.class,
                    () -> index.build(new float[][]{{1, 1}}, new long[]{1}));
        }
    }

    @Test
    void validatesInputsAndLifecycleMisuse() {
        try (CuvsVectorIndex index = new CuvsVectorIndex()) {
            assertThrows(IllegalStateException.class,
                    () -> index.search(new float[]{1}, 1,
                            new SearchParameters(DistanceMetric.EUCLIDEAN)));
            assertThrows(NullPointerException.class, () -> index.build(null, new long[]{1}));
            assertThrows(IllegalArgumentException.class,
                    () -> index.build(new float[][]{{1}, {2, 3}}, new long[]{1, 2}));
            assertThrows(IllegalArgumentException.class,
                    () -> index.build(new float[][]{{1}, {2}}, new long[]{1, 1}));

            index.build(new float[][]{{0, 0}, {1, 1}}, new long[]{1, 2});
            assertThrows(IllegalArgumentException.class,
                    () -> index.search(new float[]{1}, 1,
                            new SearchParameters(DistanceMetric.EUCLIDEAN)));
            assertThrows(IllegalArgumentException.class,
                    () -> index.search(new float[]{1, 1}, 3,
                            new SearchParameters(DistanceMetric.EUCLIDEAN)));
            assertThrows(IllegalArgumentException.class,
                    () -> index.searchBatch(new float[0][], 1,
                            new SearchParameters(DistanceMetric.EUCLIDEAN)));
        }
    }

    private static void assertResultsEqual(
            List<List<SearchResult>> expected,
            List<List<SearchResult>> actual
    ) {
        assertEquals(expected.size(), actual.size());
        for (int query = 0; query < expected.size(); query++) {
            assertEquals(expected.get(query).size(), actual.get(query).size());
            for (int rank = 0; rank < expected.get(query).size(); rank++) {
                assertEquals(expected.get(query).get(rank).id(), actual.get(query).get(rank).id());
                assertEquals(expected.get(query).get(rank).score(),
                        actual.get(query).get(rank).score(), 1.0e-5f);
            }
        }
    }
}

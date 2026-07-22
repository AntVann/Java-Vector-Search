package com.vectorforge.nativeindex;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.IndexMetrics;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.cpu.CpuBruteForceIndex;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class NativeBruteForceIndexTest {

    @BeforeAll
    static void requireNativeProfile() {
        assumeTrue(Boolean.getBoolean("vectorforge.native.tests.enabled"), "Native tests require the -Pnative profile");
    }

    @Test
    void matchesCpuReferenceAcrossMetrics() {
        Random random = new Random(113L);
        float[][] vectors = randomVectors(random, 96, 24);
        long[] ids = sequentialIds(96);
        float[] query = randomVector(random, 24);

        try (CpuBruteForceIndex cpu = new CpuBruteForceIndex();
             NativeBruteForceIndex nativeIndex = new NativeBruteForceIndex()) {
            cpu.build(vectors, ids);
            nativeIndex.build(vectors, ids);

            for (DistanceMetric metric : DistanceMetric.values()) {
                List<SearchResult> expected = cpu.search(query, 12, new SearchParameters(metric));
                List<SearchResult> actual = nativeIndex.search(query, 12, new SearchParameters(metric));
                assertIterableEquals(expected, actual, "metric=" + metric);
            }
        }
    }

    @Test
    void matchesCpuReferenceForBatchSearch() {
        Random random = new Random(211L);
        float[][] vectors = randomVectors(random, 128, 16);
        long[] ids = sequentialIds(128);
        float[][] queries = randomVectors(random, 5, 16);

        try (CpuBruteForceIndex cpu = new CpuBruteForceIndex();
             NativeBruteForceIndex nativeIndex = new NativeBruteForceIndex()) {
            cpu.build(vectors, ids);
            nativeIndex.build(vectors, ids);

            List<List<SearchResult>> expected = cpu.searchBatch(queries, 7, new SearchParameters(DistanceMetric.COSINE));
            List<List<SearchResult>> actual = nativeIndex.searchBatch(queries, 7, new SearchParameters(DistanceMetric.COSINE));

            assertEquals(expected, actual);
        }
    }

    @Test
    void rejectsInvalidPublicInputsAndLifecycleMisuse() {
        NativeBruteForceIndex index = new NativeBruteForceIndex();

        assertThrows(NullPointerException.class, () -> index.build(null, new long[]{1L}));
        assertThrows(NullPointerException.class, () -> index.build(new float[][]{{1.0f}}, null));
        assertThrows(IllegalArgumentException.class, () -> index.build(new float[0][], new long[0]));
        assertThrows(IllegalArgumentException.class, () -> index.build(new float[][]{{1.0f}, {2.0f, 3.0f}}, new long[]{1L, 2L}));
        assertThrows(IllegalArgumentException.class, () -> index.build(new float[][]{{1.0f}, {2.0f}}, new long[]{9L, 9L}));

        assertThrows(IllegalStateException.class, () -> index.search(
                new float[]{1.0f},
                1,
                new SearchParameters(DistanceMetric.EUCLIDEAN)
        ));

        index.build(new float[][]{{1.0f, 0.0f}, {0.0f, 1.0f}}, new long[]{1L, 2L});

        assertThrows(NullPointerException.class, () -> index.search(null, 1, new SearchParameters(DistanceMetric.EUCLIDEAN)));
        assertThrows(NullPointerException.class, () -> index.search(new float[]{1.0f, 0.0f}, 1, null));
        assertThrows(IllegalArgumentException.class, () -> index.search(new float[]{1.0f}, 1, new SearchParameters(DistanceMetric.EUCLIDEAN)));
        assertThrows(IllegalArgumentException.class, () -> index.search(new float[]{1.0f, 0.0f}, 0, new SearchParameters(DistanceMetric.EUCLIDEAN)));
        assertThrows(IllegalArgumentException.class, () -> index.search(new float[]{1.0f, 0.0f}, 3, new SearchParameters(DistanceMetric.EUCLIDEAN)));
        assertThrows(IllegalArgumentException.class, () -> index.searchBatch(new float[0][], 1, new SearchParameters(DistanceMetric.EUCLIDEAN)));
        assertThrows(IllegalArgumentException.class, () -> index.searchBatch(new float[][]{{1.0f}, {1.0f, 2.0f}}, 1, new SearchParameters(DistanceMetric.EUCLIDEAN)));

        IndexMetrics builtMetrics = index.metrics();
        assertTrue(builtMetrics.built());
        assertFalse(builtMetrics.closed());
        assertEquals(2L, builtMetrics.vectorCount());

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
    void validatesRawNativeBuffersAndHandles() {
        ByteBuffer validVectors = directByteBuffer(Float.BYTES * 2L);
        validVectors.asFloatBuffer().put(new float[]{1.0f, 2.0f});
        ByteBuffer validIds = directByteBuffer(Long.BYTES);
        validIds.asLongBuffer().put(new long[]{7L});

        assertThrows(NullPointerException.class, () -> NativeBindings.createIndex(null, validIds, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> NativeBindings.createIndex(ByteBuffer.allocate(Float.BYTES * 2), validIds, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> NativeBindings.createIndex(validVectors, validIds, 0, 2));
        assertThrows(IllegalArgumentException.class, () -> NativeBindings.createIndex(validVectors, validIds, 1, 3));
        assertThrows(IllegalStateException.class, () -> NativeBindings.destroyIndex(42L));

        long handle = NativeBindings.createIndex(validVectors, validIds, 1, 2);
        ByteBuffer validQuery = directByteBuffer(Float.BYTES * 2L);
        validQuery.asFloatBuffer().put(new float[]{1.0f, 2.0f});
        ByteBuffer tooSmallIds = directByteBuffer(Integer.BYTES);
        ByteBuffer validOutputScores = directByteBuffer(Float.BYTES);

        assertThrows(IllegalArgumentException.class, () -> NativeBindings.search(
                handle,
                validQuery,
                1,
                2,
                1,
                0,
                tooSmallIds,
                validOutputScores
        ));

        NativeBindings.destroyIndex(handle);
        assertThrows(IllegalStateException.class, () -> NativeBindings.search(
                handle,
                validQuery,
                1,
                2,
                1,
                0,
                directByteBuffer(Long.BYTES),
                validOutputScores
        ));
    }

    @Test
    void survivesRepeatedCreateSearchCloseCycles() {
        for (int iteration = 0; iteration < 100; iteration++) {
            float[][] vectors = randomVectors(new Random(500L + iteration), 32, 8);
            long[] ids = sequentialIds(32);
            float[] query = randomVector(new Random(800L + iteration), 8);

            try (CpuBruteForceIndex cpu = new CpuBruteForceIndex();
                 NativeBruteForceIndex nativeIndex = new NativeBruteForceIndex()) {
                cpu.build(vectors, ids);
                nativeIndex.build(vectors, ids);

                List<SearchResult> expected = cpu.search(query, 5, new SearchParameters(DistanceMetric.DOT_PRODUCT));
                List<SearchResult> actual = nativeIndex.search(query, 5, new SearchParameters(DistanceMetric.DOT_PRODUCT));
                assertEquals(expected, actual);
            }
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

    private static ByteBuffer directByteBuffer(long bytes) {
        return ByteBuffer.allocateDirect(Math.toIntExact(bytes)).order(ByteOrder.nativeOrder());
    }
}

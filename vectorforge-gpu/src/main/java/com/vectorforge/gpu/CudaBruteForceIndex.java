package com.vectorforge.gpu;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.IndexMetrics;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;
import com.vectorforge.nativeindex.NativeBindings;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Educational exact CUDA backend that keeps the indexed vectors resident on the GPU.
 *
 * <p>The current implementation supports dot-product search only and serializes searches so it can
 * reuse query and score buffers across calls.
 */
public final class CudaBruteForceIndex implements VectorIndex {

    private static final String BACKEND_NAME = "cuda-bruteforce";
    private static final DistanceMetric SUPPORTED_METRIC = DistanceMetric.DOT_PRODUCT;

    private final ReadWriteLock lifecycleLock = new ReentrantReadWriteLock();
    private final Object searchLock = new Object();

    private NativeIndexState state;
    private boolean closed;
    private volatile CudaSearchTimings lastSearchTimings;

    public CudaBruteForceIndex() {
        ensureCudaAvailable();
    }

    public static boolean isCudaAvailable() {
        try {
            return NativeBindings.isCudaCompiled() && NativeBindings.getCudaDeviceCount() > 0;
        } catch (RuntimeException ex) {
            return false;
        }
    }

    public static int cudaDeviceCount() {
        return NativeBindings.isCudaCompiled() ? NativeBindings.getCudaDeviceCount() : 0;
    }

    public CudaSearchTimings lastSearchTimings() {
        return lastSearchTimings;
    }

    @Override
    public void build(float[][] vectors, long[] ids) {
        ensureCudaAvailable();
        ValidatedBuildInput input = validateAndPackBuildInput(vectors, ids);
        long handle = NativeBindings.createCudaIndex(input.vectorsBuffer(), input.idsBuffer(), input.vectorCount(), input.dimensions());

        long priorHandle = 0L;
        lifecycleLock.writeLock().lock();
        try {
            ensureOpenLocked();
            NativeIndexState previous = state;
            state = new NativeIndexState(handle, input.vectorCount(), input.dimensions(), input.vectorBytes());
            lastSearchTimings = null;
            if (previous != null) {
                priorHandle = previous.handle();
            }
        } catch (RuntimeException ex) {
            NativeBindings.destroyIndex(handle);
            throw ex;
        } finally {
            lifecycleLock.writeLock().unlock();
        }

        if (priorHandle != 0L) {
            NativeBindings.destroyIndex(priorHandle);
        }
    }

    @Override
    public List<SearchResult> search(float[] query, int k, SearchParameters parameters) {
        Objects.requireNonNull(parameters, "parameters must not be null");
        requireSupportedMetric(parameters.metric());

        lifecycleLock.readLock().lock();
        try {
            ensureOpenLocked();
            NativeIndexState snapshot = requireBuiltLocked();
            validateK(k, snapshot.vectorCount());
            ByteBuffer queryBuffer = packSingleQuery(query, snapshot.dimensions());
            ByteBuffer outputIdsBuffer = allocateDirectBytes((long) k * Long.BYTES, "output ids");
            ByteBuffer outputScoresBuffer = allocateDirectBytes((long) k * Float.BYTES, "output scores");
            ByteBuffer timingBuffer = allocateDirectBytes(4L * Double.BYTES, "timing values");

            synchronized (searchLock) {
                NativeBindings.searchCuda(
                        snapshot.handle(),
                        queryBuffer,
                        1,
                        snapshot.dimensions(),
                        k,
                        metricCode(parameters.metric()),
                        outputIdsBuffer,
                        outputScoresBuffer,
                        timingBuffer
                );
                lastSearchTimings = unpackTimings(timingBuffer, 1, snapshot.vectorCount(), k);
            }

            return unpackSingleQueryResults(outputIdsBuffer, outputScoresBuffer, k);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Executes a batched search through one native CUDA call.
     *
     * @param queries query vectors
     * @param k number of results per query
     * @param parameters search controls
     * @return ordered nearest-neighbor results per query
     */
    public List<List<SearchResult>> searchBatch(float[][] queries, int k, SearchParameters parameters) {
        Objects.requireNonNull(parameters, "parameters must not be null");
        requireSupportedMetric(parameters.metric());

        lifecycleLock.readLock().lock();
        try {
            ensureOpenLocked();
            NativeIndexState snapshot = requireBuiltLocked();
            validateK(k, snapshot.vectorCount());
            ValidatedQueryBatch queryBatch = validateAndPackQueryBatch(queries, snapshot.dimensions());
            long resultCount = (long) queryBatch.queryCount() * k;
            ByteBuffer outputIdsBuffer = allocateDirectBytes(resultCount * Long.BYTES, "output ids");
            ByteBuffer outputScoresBuffer = allocateDirectBytes(resultCount * Float.BYTES, "output scores");
            ByteBuffer timingBuffer = allocateDirectBytes(4L * Double.BYTES, "timing values");

            synchronized (searchLock) {
                NativeBindings.searchCuda(
                        snapshot.handle(),
                        queryBatch.queryBuffer(),
                        queryBatch.queryCount(),
                        snapshot.dimensions(),
                        k,
                        metricCode(parameters.metric()),
                        outputIdsBuffer,
                        outputScoresBuffer,
                        timingBuffer
                );
                lastSearchTimings = unpackTimings(timingBuffer, queryBatch.queryCount(), snapshot.vectorCount(), k);
            }

            return unpackBatchResults(outputIdsBuffer, outputScoresBuffer, queryBatch.queryCount(), k);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    @Override
    public IndexMetrics metrics() {
        lifecycleLock.readLock().lock();
        try {
            if (state == null) {
                return new IndexMetrics(BACKEND_NAME, false, closed, 0, 0, 0, false);
            }
            return new IndexMetrics(
                    BACKEND_NAME,
                    true,
                    closed,
                    state.vectorCount(),
                    state.dimensions(),
                    state.vectorBytes(),
                    true
            );
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    @Override
    public void close() {
        long handleToDestroy = 0L;
        lifecycleLock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            lastSearchTimings = null;
            if (state != null) {
                handleToDestroy = state.handle();
                state = null;
            }
        } finally {
            lifecycleLock.writeLock().unlock();
        }

        if (handleToDestroy != 0L) {
            NativeBindings.destroyIndex(handleToDestroy);
        }
    }

    private static void ensureCudaAvailable() {
        if (!NativeBindings.isCudaCompiled()) {
            throw new IllegalStateException("CUDA support was not compiled into the VectorForge native library. Build with -Pcuda.");
        }
        int deviceCount = NativeBindings.getCudaDeviceCount();
        if (deviceCount <= 0) {
            throw new IllegalStateException("No CUDA device is available for the VectorForge CUDA backend.");
        }
    }

    private NativeIndexState requireBuiltLocked() {
        if (state == null) {
            throw new IllegalStateException("index has not been built");
        }
        return state;
    }

    private void ensureOpenLocked() {
        if (closed) {
            throw new IllegalStateException("index has been closed");
        }
    }

    private static void requireSupportedMetric(DistanceMetric metric) {
        Objects.requireNonNull(metric, "metric must not be null");
        if (metric != SUPPORTED_METRIC) {
            throw new IllegalArgumentException("CUDA backend currently supports only DOT_PRODUCT");
        }
    }

    private static int metricCode(DistanceMetric metric) {
        return switch (metric) {
            case EUCLIDEAN -> 0;
            case COSINE -> 1;
            case DOT_PRODUCT -> 2;
        };
    }

    private static ValidatedBuildInput validateAndPackBuildInput(float[][] vectors, long[] ids) {
        Objects.requireNonNull(vectors, "vectors must not be null");
        Objects.requireNonNull(ids, "ids must not be null");
        if (vectors.length == 0) {
            throw new IllegalArgumentException("vectors must not be empty");
        }
        if (vectors.length != ids.length) {
            throw new IllegalArgumentException("vectors length must match ids length");
        }

        int dimensions = -1;
        Set<Long> seenIds = new HashSet<>(Math.max(16, vectors.length * 2));
        for (long id : ids) {
            if (!seenIds.add(id)) {
                throw new IllegalArgumentException("duplicate id detected: " + id);
            }
        }

        for (int row = 0; row < vectors.length; row++) {
            float[] vector = vectors[row];
            if (vector == null) {
                throw new IllegalArgumentException("vector at index " + row + " must not be null");
            }
            if (dimensions == -1) {
                if (vector.length == 0) {
                    throw new IllegalArgumentException("vector dimensions must be positive");
                }
                dimensions = vector.length;
            } else if (vector.length != dimensions) {
                throw new IllegalArgumentException("vector at index " + row + " has dimension " + vector.length
                        + " but expected " + dimensions);
            }
        }

        long elementCount = (long) vectors.length * dimensions;
        ByteBuffer vectorsBuffer = allocateDirectBytes(elementCount * Float.BYTES, "vector values");
        FloatBuffer floatView = vectorsBuffer.asFloatBuffer();
        for (float[] vector : vectors) {
            floatView.put(vector);
        }

        long[] copiedIds = Arrays.copyOf(ids, ids.length);
        ByteBuffer idsBuffer = allocateDirectBytes((long) copiedIds.length * Long.BYTES, "vector ids");
        idsBuffer.asLongBuffer().put(copiedIds);

        return new ValidatedBuildInput(vectorsBuffer, idsBuffer, vectors.length, dimensions, elementCount * Float.BYTES);
    }

    private static ByteBuffer packSingleQuery(float[] query, int expectedDimensions) {
        Objects.requireNonNull(query, "query must not be null");
        if (query.length != expectedDimensions) {
            throw new IllegalArgumentException("query dimension " + query.length
                    + " does not match index dimension " + expectedDimensions);
        }
        ByteBuffer queryBuffer = allocateDirectBytes((long) expectedDimensions * Float.BYTES, "query values");
        queryBuffer.asFloatBuffer().put(query);
        return queryBuffer;
    }

    private static ValidatedQueryBatch validateAndPackQueryBatch(float[][] queries, int expectedDimensions) {
        Objects.requireNonNull(queries, "queries must not be null");
        if (queries.length == 0) {
            throw new IllegalArgumentException("queries must not be empty");
        }

        long elementCount = (long) queries.length * expectedDimensions;
        ByteBuffer queryBuffer = allocateDirectBytes(elementCount * Float.BYTES, "query values");
        FloatBuffer floatView = queryBuffer.asFloatBuffer();

        for (int i = 0; i < queries.length; i++) {
            float[] query = queries[i];
            if (query == null) {
                throw new IllegalArgumentException("query at index " + i + " must not be null");
            }
            if (query.length != expectedDimensions) {
                throw new IllegalArgumentException("query at index " + i + " has dimension " + query.length
                        + " but expected " + expectedDimensions);
            }
            floatView.put(query);
        }

        return new ValidatedQueryBatch(queryBuffer, queries.length);
    }

    private static void validateK(int k, int vectorCount) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        if (k > vectorCount) {
            throw new IllegalArgumentException("k must be <= vector count");
        }
    }

    private static ByteBuffer allocateDirectBytes(long byteCount, String description) {
        if (byteCount <= 0L) {
            throw new IllegalArgumentException(description + " size must be positive");
        }
        if (byteCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(description + " size exceeds Java direct buffer limits");
        }
        return ByteBuffer.allocateDirect((int) byteCount).order(ByteOrder.nativeOrder());
    }

    private static List<SearchResult> unpackSingleQueryResults(ByteBuffer outputIdsBuffer, ByteBuffer outputScoresBuffer, int k) {
        long[] ids = new long[k];
        float[] scores = new float[k];
        outputIdsBuffer.asLongBuffer().get(ids);
        outputScoresBuffer.asFloatBuffer().get(scores);

        ArrayList<SearchResult> results = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            results.add(new SearchResult(ids[i], scores[i]));
        }
        return List.copyOf(results);
    }

    private static List<List<SearchResult>> unpackBatchResults(
            ByteBuffer outputIdsBuffer,
            ByteBuffer outputScoresBuffer,
            int queryCount,
            int k
    ) {
        long[] ids = new long[queryCount * k];
        float[] scores = new float[queryCount * k];
        outputIdsBuffer.asLongBuffer().get(ids);
        outputScoresBuffer.asFloatBuffer().get(scores);

        ArrayList<List<SearchResult>> allResults = new ArrayList<>(queryCount);
        for (int queryIndex = 0; queryIndex < queryCount; queryIndex++) {
            ArrayList<SearchResult> results = new ArrayList<>(k);
            int offset = queryIndex * k;
            for (int rank = 0; rank < k; rank++) {
                results.add(new SearchResult(ids[offset + rank], scores[offset + rank]));
            }
            allResults.add(List.copyOf(results));
        }
        return List.copyOf(allResults);
    }

    private static CudaSearchTimings unpackTimings(ByteBuffer timingBuffer, int queryCount, int vectorCount, int k) {
        DoubleBuffer timings = timingBuffer.asDoubleBuffer();
        return new CudaSearchTimings(
                timings.get(0),
                timings.get(1),
                timings.get(2),
                timings.get(3),
                queryCount,
                vectorCount,
                k
        );
    }

    private record NativeIndexState(long handle, int vectorCount, int dimensions, long vectorBytes) {
    }

    private record ValidatedBuildInput(
            ByteBuffer vectorsBuffer,
            ByteBuffer idsBuffer,
            int vectorCount,
            int dimensions,
            long vectorBytes
    ) {
    }

    private record ValidatedQueryBatch(ByteBuffer queryBuffer, int queryCount) {
    }
}

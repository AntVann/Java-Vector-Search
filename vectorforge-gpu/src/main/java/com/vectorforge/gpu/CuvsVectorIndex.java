package com.vectorforge.gpu;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.IndexMetrics;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;
import com.vectorforge.nativeindex.NativeBindings;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Exact brute-force vector index backed by the optional cuVS native integration.
 */
public final class CuvsVectorIndex implements VectorIndex {

    private static final String BACKEND_NAME = "cuvs-bruteforce";
    private final ReadWriteLock lifecycleLock = new ReentrantReadWriteLock();
    private State state;
    private boolean closed;

    public CuvsVectorIndex() {
        ensureCuvsAvailable();
    }

    public static boolean isCuvsAvailable() {
        try {
            return NativeBindings.isCuvsCompiled() && NativeBindings.getCudaDeviceCount() > 0;
        } catch (RuntimeException | LinkageError ignored) {
            return false;
        }
    }

    public static String cuvsVersion() {
        if (!NativeBindings.isCuvsCompiled()) {
            throw new IllegalStateException("cuVS support was not compiled into the VectorForge native library");
        }
        return NativeBindings.getCuvsVersion();
    }

    @Override
    public void build(float[][] vectors, long[] ids) {
        lifecycleLock.readLock().lock();
        try {
            ensureOpen();
        } finally {
            lifecycleLock.readLock().unlock();
        }

        PackedBuild input = packBuild(vectors, ids);
        long handle = NativeBindings.createCuvsIndex(
                input.vectors(), input.ids(), input.vectorCount(), input.dimensions());
        long oldHandle = 0;
        lifecycleLock.writeLock().lock();
        try {
            ensureOpen();
            if (state != null) {
                oldHandle = state.handle();
            }
            state = new State(handle, input.vectorCount(), input.dimensions(), input.vectorBytes());
        } catch (RuntimeException exception) {
            NativeBindings.destroyIndex(handle);
            throw exception;
        } finally {
            lifecycleLock.writeLock().unlock();
        }
        if (oldHandle != 0) {
            NativeBindings.destroyIndex(oldHandle);
        }
    }

    @Override
    public List<SearchResult> search(float[] query, int k, SearchParameters parameters) {
        Objects.requireNonNull(query, "query must not be null");
        Objects.requireNonNull(parameters, "parameters must not be null");
        lifecycleLock.readLock().lock();
        try {
            State current = requireBuilt();
            validateK(k, current.vectorCount());
            if (query.length != current.dimensions()) {
                throw new IllegalArgumentException("query dimension " + query.length
                        + " does not match index dimension " + current.dimensions());
            }
            return searchPacked(current, packFloats(new float[][]{query}, current.dimensions()), 1, k, parameters)
                    .getFirst();
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    @Override
    public List<List<SearchResult>> searchBatch(float[][] queries, int k, SearchParameters parameters) {
        Objects.requireNonNull(parameters, "parameters must not be null");
        lifecycleLock.readLock().lock();
        try {
            State current = requireBuilt();
            validateK(k, current.vectorCount());
            Objects.requireNonNull(queries, "queries must not be null");
            if (queries.length == 0) {
                throw new IllegalArgumentException("queries must not be empty");
            }
            return searchPacked(current, packFloats(queries, current.dimensions()), queries.length, k, parameters);
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
            return new IndexMetrics(BACKEND_NAME, true, closed, state.vectorCount(),
                    state.dimensions(), state.vectorBytes(), true);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    @Override
    public void close() {
        long handle = 0;
        lifecycleLock.writeLock().lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            if (state != null) {
                handle = state.handle();
                state = null;
            }
        } finally {
            lifecycleLock.writeLock().unlock();
        }
        if (handle != 0) {
            NativeBindings.destroyIndex(handle);
        }
    }

    private List<List<SearchResult>> searchPacked(
            State current,
            ByteBuffer queries,
            int queryCount,
            int k,
            SearchParameters parameters
    ) {
        int resultCount = Math.multiplyExact(queryCount, k);
        ByteBuffer outputIds = allocate((long) resultCount * Long.BYTES, "output ids");
        ByteBuffer outputScores = allocate((long) resultCount * Float.BYTES, "output scores");
        NativeBindings.search(current.handle(), queries, queryCount, current.dimensions(), k,
                metricCode(parameters.metric()), outputIds, outputScores);

        ArrayList<List<SearchResult>> batches = new ArrayList<>(queryCount);
        for (int queryIndex = 0; queryIndex < queryCount; queryIndex++) {
            ArrayList<SearchResult> results = new ArrayList<>(k);
            for (int rank = 0; rank < k; rank++) {
                int offset = queryIndex * k + rank;
                results.add(new SearchResult(
                        outputIds.asLongBuffer().get(offset),
                        outputScores.asFloatBuffer().get(offset)));
            }
            batches.add(List.copyOf(results));
        }
        return List.copyOf(batches);
    }

    private State requireBuilt() {
        ensureOpen();
        if (state == null) {
            throw new IllegalStateException("index has not been built");
        }
        return state;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("index has been closed");
        }
    }

    private static void ensureCuvsAvailable() {
        if (!NativeBindings.isCuvsCompiled()) {
            throw new IllegalStateException(
                    "cuVS support was not compiled into the VectorForge native library. Build with -Pcuvs.");
        }
        if (NativeBindings.getCudaDeviceCount() <= 0) {
            throw new IllegalStateException("No CUDA device is available for the VectorForge cuVS backend.");
        }
    }

    private static int metricCode(DistanceMetric metric) {
        Objects.requireNonNull(metric, "metric must not be null");
        return switch (metric) {
            case EUCLIDEAN -> 0;
            case COSINE -> 1;
            case DOT_PRODUCT -> 2;
        };
    }

    private static PackedBuild packBuild(float[][] vectors, long[] ids) {
        Objects.requireNonNull(vectors, "vectors must not be null");
        Objects.requireNonNull(ids, "ids must not be null");
        if (vectors.length == 0 || vectors.length != ids.length) {
            throw new IllegalArgumentException("vectors must be non-empty and match ids length");
        }
        Set<Long> uniqueIds = new HashSet<>();
        for (long id : ids) {
            if (!uniqueIds.add(id)) {
                throw new IllegalArgumentException("duplicate id detected: " + id);
            }
        }
        Objects.requireNonNull(vectors[0], "vector at index 0 must not be null");
        int dimensions = vectors[0].length;
        if (dimensions == 0) {
            throw new IllegalArgumentException("vector dimensions must be positive");
        }
        ByteBuffer vectorBuffer = packFloats(vectors, dimensions);
        ByteBuffer idBuffer = allocate((long) ids.length * Long.BYTES, "vector ids");
        idBuffer.asLongBuffer().put(ids);
        return new PackedBuild(vectorBuffer, idBuffer, vectors.length, dimensions,
                (long) vectors.length * dimensions * Float.BYTES);
    }

    private static ByteBuffer packFloats(float[][] values, int dimensions) {
        ByteBuffer buffer = allocate((long) values.length * dimensions * Float.BYTES, "vector values");
        for (int index = 0; index < values.length; index++) {
            float[] value = values[index];
            if (value == null) {
                throw new IllegalArgumentException("vector at index " + index + " must not be null");
            }
            if (value.length != dimensions) {
                throw new IllegalArgumentException("vector at index " + index + " has dimension "
                        + value.length + " but expected " + dimensions);
            }
            validateFinite(value, "vector at index " + index);
            buffer.asFloatBuffer().position(index * dimensions).put(value);
        }
        return buffer;
    }

    private static void validateK(int k, int vectorCount) {
        if (k <= 0 || k > vectorCount) {
            throw new IllegalArgumentException("k must be positive and <= vector count");
        }
    }

    private static void validateFinite(float[] values, String description) {
        for (int i = 0; i < values.length; i++) {
            if (!Float.isFinite(values[i])) {
                throw new IllegalArgumentException(description + " contains a non-finite value at dimension " + i);
            }
        }
    }

    private static ByteBuffer allocate(long bytes, String description) {
        if (bytes <= 0 || bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(description + " size is outside Java direct buffer limits");
        }
        return ByteBuffer.allocateDirect((int) bytes).order(ByteOrder.nativeOrder());
    }

    private record State(long handle, int vectorCount, int dimensions, long vectorBytes) {
    }

    private record PackedBuild(
            ByteBuffer vectors,
            ByteBuffer ids,
            int vectorCount,
            int dimensions,
            long vectorBytes
    ) {
    }
}

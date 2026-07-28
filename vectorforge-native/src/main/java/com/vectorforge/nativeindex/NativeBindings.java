package com.vectorforge.nativeindex;

import java.nio.ByteBuffer;
import java.util.Objects;

public final class NativeBindings {

    private NativeBindings() {
    }

    public static long createIndex(ByteBuffer vectorsBuffer, ByteBuffer idsBuffer, int vectorCount, int dimensions) {
        NativeLibraryLoader.load();
        Objects.requireNonNull(vectorsBuffer, "vectorsBuffer must not be null");
        Objects.requireNonNull(idsBuffer, "idsBuffer must not be null");
        return nativeCreateIndex(vectorsBuffer, idsBuffer, vectorCount, dimensions);
    }

    public static long createCudaIndex(ByteBuffer vectorsBuffer, ByteBuffer idsBuffer, int vectorCount, int dimensions) {
        NativeLibraryLoader.load();
        Objects.requireNonNull(vectorsBuffer, "vectorsBuffer must not be null");
        Objects.requireNonNull(idsBuffer, "idsBuffer must not be null");
        return nativeCreateCudaIndex(vectorsBuffer, idsBuffer, vectorCount, dimensions);
    }

    public static long createCuvsIndex(ByteBuffer vectorsBuffer, ByteBuffer idsBuffer, int vectorCount, int dimensions) {
        NativeLibraryLoader.load();
        Objects.requireNonNull(vectorsBuffer, "vectorsBuffer must not be null");
        Objects.requireNonNull(idsBuffer, "idsBuffer must not be null");
        return nativeCreateCuvsIndex(vectorsBuffer, idsBuffer, vectorCount, dimensions);
    }

    public static void search(
            long handle,
            ByteBuffer queriesBuffer,
            int queryCount,
            int dimensions,
            int k,
            int metricCode,
            ByteBuffer outputIdsBuffer,
            ByteBuffer outputScoresBuffer
    ) {
        NativeLibraryLoader.load();
        Objects.requireNonNull(queriesBuffer, "queriesBuffer must not be null");
        Objects.requireNonNull(outputIdsBuffer, "outputIdsBuffer must not be null");
        Objects.requireNonNull(outputScoresBuffer, "outputScoresBuffer must not be null");
        nativeSearch(handle, queriesBuffer, queryCount, dimensions, k, metricCode, outputIdsBuffer, outputScoresBuffer);
    }

    public static void searchCuda(
            long handle,
            ByteBuffer queriesBuffer,
            int queryCount,
            int dimensions,
            int k,
            int metricCode,
            ByteBuffer outputIdsBuffer,
            ByteBuffer outputScoresBuffer,
            ByteBuffer timingBuffer
    ) {
        NativeLibraryLoader.load();
        Objects.requireNonNull(queriesBuffer, "queriesBuffer must not be null");
        Objects.requireNonNull(outputIdsBuffer, "outputIdsBuffer must not be null");
        Objects.requireNonNull(outputScoresBuffer, "outputScoresBuffer must not be null");
        Objects.requireNonNull(timingBuffer, "timingBuffer must not be null");
        nativeSearchCuda(handle, queriesBuffer, queryCount, dimensions, k, metricCode, outputIdsBuffer, outputScoresBuffer, timingBuffer);
    }

    public static void destroyIndex(long handle) {
        NativeLibraryLoader.load();
        nativeDestroyIndex(handle);
    }

    public static boolean isCudaCompiled() {
        NativeLibraryLoader.load();
        return nativeIsCudaCompiled();
    }

    public static int getCudaDeviceCount() {
        NativeLibraryLoader.load();
        return nativeGetCudaDeviceCount();
    }

    public static boolean isCuvsCompiled() {
        NativeLibraryLoader.load();
        return nativeIsCuvsCompiled();
    }

    public static String getCuvsVersion() {
        NativeLibraryLoader.load();
        return nativeGetCuvsVersion();
    }

    private static native long nativeCreateIndex(ByteBuffer vectorsBuffer, ByteBuffer idsBuffer, int vectorCount, int dimensions);

    private static native long nativeCreateCudaIndex(ByteBuffer vectorsBuffer, ByteBuffer idsBuffer, int vectorCount, int dimensions);

    private static native long nativeCreateCuvsIndex(ByteBuffer vectorsBuffer, ByteBuffer idsBuffer, int vectorCount, int dimensions);

    private static native void nativeSearch(
            long handle,
            ByteBuffer queriesBuffer,
            int queryCount,
            int dimensions,
            int k,
            int metricCode,
            ByteBuffer outputIdsBuffer,
            ByteBuffer outputScoresBuffer
    );

    private static native void nativeSearchCuda(
            long handle,
            ByteBuffer queriesBuffer,
            int queryCount,
            int dimensions,
            int k,
            int metricCode,
            ByteBuffer outputIdsBuffer,
            ByteBuffer outputScoresBuffer,
            ByteBuffer timingBuffer
    );

    private static native void nativeDestroyIndex(long handle);

    private static native boolean nativeIsCudaCompiled();

    private static native int nativeGetCudaDeviceCount();

    private static native boolean nativeIsCuvsCompiled();

    private static native String nativeGetCuvsVersion();
}

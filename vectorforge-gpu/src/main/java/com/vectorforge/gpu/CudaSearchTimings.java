package com.vectorforge.gpu;

/**
 * Timing breakdown captured for the most recent CUDA search call.
 *
 * @param hostToDeviceMillis query-transfer time from host memory into device memory
 * @param kernelMillis kernel execution time measured after explicit synchronization
 * @param deviceToHostMillis score-transfer time from device memory back to host memory
 * @param totalMillis end-to-end search latency including buffer growth and host-side top-k selection
 * @param queryCount number of queries in the measured batch
 * @param vectorCount number of indexed vectors scanned by the kernel
 * @param k requested result count per query
 */
public record CudaSearchTimings(
        double hostToDeviceMillis,
        double kernelMillis,
        double deviceToHostMillis,
        double totalMillis,
        int queryCount,
        int vectorCount,
        int k
) {
}

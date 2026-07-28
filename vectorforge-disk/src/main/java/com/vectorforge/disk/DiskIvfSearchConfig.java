package com.vectorforge.disk;

/**
 * Query-time controls. Candidate bytes bound decoded IDs and vectors sent to the reranker.
 */
public record DiskIvfSearchConfig(
        int partitionsToProbe,
        long cacheBytes,
        long maxBackendBatchBytes
) {
    public DiskIvfSearchConfig {
        if (partitionsToProbe <= 0) {
            throw new IllegalArgumentException("partitionsToProbe must be positive");
        }
        if (cacheBytes < 0) {
            throw new IllegalArgumentException("cacheBytes must be non-negative");
        }
        if (maxBackendBatchBytes <= 0) {
            throw new IllegalArgumentException("maxBackendBatchBytes must be positive");
        }
    }

    public static DiskIvfSearchConfig defaults() {
        return new DiskIvfSearchConfig(4, 32L * 1024 * 1024, 16L * 1024 * 1024);
    }
}

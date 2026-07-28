package com.vectorforge.disk;

public record DiskIvfMetrics(
        String generation,
        int centroidCount,
        int partitionCount,
        long onDiskBytes,
        int cacheEntries,
        long cacheResidentBytes,
        long cacheHits,
        long cacheMisses,
        long cacheEvictions
) {
}

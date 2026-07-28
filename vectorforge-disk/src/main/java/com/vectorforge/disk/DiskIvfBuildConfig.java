package com.vectorforge.disk;

import com.vectorforge.api.DistanceMetric;

import java.util.Objects;

/**
 * Immutable construction controls for the research prototype.
 */
public record DiskIvfBuildConfig(
        int centroidCount,
        int trainingIterations,
        long seed,
        DistanceMetric metric
) {
    public DiskIvfBuildConfig {
        if (centroidCount <= 0) {
            throw new IllegalArgumentException("centroidCount must be positive");
        }
        if (trainingIterations <= 0) {
            throw new IllegalArgumentException("trainingIterations must be positive");
        }
        Objects.requireNonNull(metric, "metric must not be null");
        if (metric != DistanceMetric.EUCLIDEAN) {
            throw new IllegalArgumentException("prototype currently supports EUCLIDEAN only");
        }
    }

    public static DiskIvfBuildConfig defaults() {
        return new DiskIvfBuildConfig(16, 8, 0x5EEDL, DistanceMetric.EUCLIDEAN);
    }
}

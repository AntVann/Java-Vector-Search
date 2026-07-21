package com.vectorforge.api;

import java.util.Objects;

/**
 * Backend-independent index metadata.
 *
 * @param backend backend name
 * @param built whether an index has been built
 * @param closed whether the index has been closed
 * @param vectorCount number of indexed vectors
 * @param dimensions vector dimensionality
 * @param vectorBytes bytes consumed by stored vector values
 * @param gpuResident whether the backend keeps the index resident on the GPU
 */
public record IndexMetrics(
        String backend,
        boolean built,
        boolean closed,
        long vectorCount,
        int dimensions,
        long vectorBytes,
        boolean gpuResident
) {

    public IndexMetrics {
        Objects.requireNonNull(backend, "backend must not be null");
        if (vectorCount < 0) {
            throw new IllegalArgumentException("vectorCount must be non-negative");
        }
        if (dimensions < 0) {
            throw new IllegalArgumentException("dimensions must be non-negative");
        }
        if (vectorBytes < 0) {
            throw new IllegalArgumentException("vectorBytes must be non-negative");
        }
        if (!built && (vectorCount != 0 || dimensions != 0 || vectorBytes != 0)) {
            throw new IllegalArgumentException("unbuilt metrics must report zero sizes");
        }
    }
}


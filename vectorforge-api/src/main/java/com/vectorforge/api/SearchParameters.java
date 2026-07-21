package com.vectorforge.api;

import java.util.Objects;

/**
 * Search-time controls shared by all backends.
 *
 * @param metric the distance or similarity metric to use
 */
public record SearchParameters(DistanceMetric metric) {

    public SearchParameters {
        Objects.requireNonNull(metric, "metric must not be null");
    }
}


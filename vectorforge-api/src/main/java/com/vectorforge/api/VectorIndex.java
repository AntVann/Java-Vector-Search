package com.vectorforge.api;

import java.util.List;

/**
 * Backend-independent vector index contract.
 */
public interface VectorIndex extends AutoCloseable {

    /**
     * Builds or replaces the index contents.
     *
     * @param vectors input vectors, one row per vector
     * @param ids unique vector identifiers aligned with {@code vectors}
     */
    void build(float[][] vectors, long[] ids);

    /**
     * Searches the index for the nearest neighbors to the provided query.
     *
     * @param query query vector
     * @param k number of results to return
     * @param parameters search controls such as metric selection
     * @return ordered nearest-neighbor results
     */
    List<SearchResult> search(float[] query, int k, SearchParameters parameters);

    /**
     * Returns the current index metadata.
     *
     * @return metrics describing the current index state
     */
    IndexMetrics metrics();

    @Override
    void close();
}


package com.vectorforge.api;

import java.util.List;
import java.util.Objects;

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
     * Searches multiple queries. Backends may override this method to use an
     * optimized batch implementation.
     *
     * @param queries query vectors
     * @param k number of results to return per query
     * @param parameters search controls such as metric selection
     * @return ordered nearest-neighbor results for each query
     * @throws NullPointerException if queries, a query row, or parameters is null
     * @throws IllegalArgumentException if the batch is empty or another search argument is invalid
     */
    default List<List<SearchResult>> searchBatch(
            float[][] queries,
            int k,
            SearchParameters parameters
    ) {
        Objects.requireNonNull(queries, "queries must not be null");
        Objects.requireNonNull(parameters, "parameters must not be null");
        if (queries.length == 0) {
            throw new IllegalArgumentException("queries must not be empty");
        }
        return java.util.Arrays.stream(queries)
                .map(query -> search(query, k, parameters))
                .toList();
    }

    /**
     * Returns the current index metadata.
     *
     * @return metrics describing the current index state
     */
    IndexMetrics metrics();

    @Override
    void close();
}


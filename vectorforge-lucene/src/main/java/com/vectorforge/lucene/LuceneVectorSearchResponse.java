package com.vectorforge.lucene;

import java.util.List;

/**
 * Search results with raw backend and complete adapter latency measured separately.
 */
public record LuceneVectorSearchResponse(
        List<LuceneVectorHit> hits,
        long rawBackendNanos,
        long endToEndNanos
) {
    public LuceneVectorSearchResponse {
        hits = List.copyOf(hits);
    }
}

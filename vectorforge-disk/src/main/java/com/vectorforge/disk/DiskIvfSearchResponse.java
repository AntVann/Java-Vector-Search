package com.vectorforge.disk;

import com.vectorforge.api.SearchResult;

import java.util.List;

public record DiskIvfSearchResponse(
        List<SearchResult> results,
        DiskIvfSearchTimings timings,
        int partitionsProbed,
        long candidatesLoaded,
        long candidateBytes,
        long cacheHits,
        long cacheMisses
) {
    public DiskIvfSearchResponse {
        results = List.copyOf(results);
    }
}

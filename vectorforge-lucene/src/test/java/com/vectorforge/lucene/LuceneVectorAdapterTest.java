package com.vectorforge.lucene;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.cpu.CpuBruteForceIndex;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LuceneVectorAdapterTest {

    @Test
    void mapsVectorIdsBackToExternalDocumentsAndPreservesResultOrdering() throws Exception {
        try (LuceneVectorAdapter adapter = adapter()) {
            long firstId = adapter.add(doc("first", 0.0f, 0.0f, "keep"));
            long secondId = adapter.add(doc("second", 1.0f, 0.0f, "keep"));
            long thirdId = adapter.add(doc("third", 2.0f, 0.0f, "keep"));
            adapter.refreshAndRebuild();

            List<LuceneVectorHit> hits = adapter.search(new float[]{0.1f, 0.0f}, 3, null).hits();

            assertEquals(List.of(firstId, secondId, thirdId),
                    hits.stream().map(LuceneVectorHit::vectorId).toList());
            assertEquals(List.of("first", "second", "third"),
                    hits.stream().map(LuceneVectorHit::externalId).toList());
        }
    }

    @Test
    void excludesDeletedDocumentsAfterRebuild() throws Exception {
        try (LuceneVectorAdapter adapter = adapter()) {
            adapter.add(doc("deleted", 0.0f, 0.0f, "keep"));
            adapter.add(doc("live", 1.0f, 0.0f, "keep"));
            adapter.refreshAndRebuild();
            adapter.delete("deleted");

            assertEquals("deleted", adapter.search(new float[]{0.0f, 0.0f}, 1, null)
                    .hits().getFirst().externalId(), "old snapshot remains visible before refresh");

            assertEquals(1, adapter.refreshAndRebuild());
            assertEquals(List.of("live"), adapter.search(new float[]{0.0f, 0.0f}, 2, null)
                    .hits().stream().map(LuceneVectorHit::externalId).toList());
        }
    }

    @Test
    void returnsEmptyResultsForEmptyIndexAndUnmatchedFilter() throws Exception {
        try (LuceneVectorAdapter adapter = adapter()) {
            adapter.refreshAndRebuild();
            assertTrue(adapter.search(new float[]{0.0f, 0.0f}, 10, null).hits().isEmpty());
            assertTrue(adapter.searchLucene(new float[]{0.0f, 0.0f}, 10, null).isEmpty());

            adapter.add(doc("one", 0.0f, 0.0f, "visible"));
            adapter.refreshAndRebuild();
            assertTrue(adapter.search(new float[]{0.0f, 0.0f}, 10, "missing").hits().isEmpty());
        }
    }

    @Test
    void filtersVectorForgeAfterSearchAndLuceneDuringSearch() throws Exception {
        try (LuceneVectorAdapter adapter = adapter()) {
            adapter.add(doc("excluded-nearest", 0.0f, 0.0f, "private"));
            adapter.add(doc("included", 1.0f, 0.0f, "public"));
            adapter.add(doc("also-included", 2.0f, 0.0f, "public"));
            adapter.refreshAndRebuild();

            List<String> vectorForgeIds = adapter.search(new float[]{0.0f, 0.0f}, 2, "public")
                    .hits().stream().map(LuceneVectorHit::externalId).toList();
            List<String> luceneIds = adapter.searchLucene(new float[]{0.0f, 0.0f}, 2, "public")
                    .stream().map(LuceneVectorHit::externalId).toList();

            assertEquals(List.of("included", "also-included"), vectorForgeIds);
            assertEquals(vectorForgeIds, luceneIds);
        }
    }

    @Test
    void updateGetsNewVectorIdAndBecomesVisibleOnlyAfterRefresh() throws Exception {
        try (LuceneVectorAdapter adapter = adapter()) {
            long oldId = adapter.add(doc("same", 0.0f, 0.0f, "old"));
            adapter.refreshAndRebuild();

            long newId = adapter.update(doc("same", 5.0f, 0.0f, "new"));
            assertNotEquals(oldId, newId);
            assertEquals(oldId, adapter.search(new float[]{0.0f, 0.0f}, 1, null).hits().getFirst().vectorId());

            assertEquals(1, adapter.refreshAndRebuild());
            LuceneVectorHit hit = adapter.search(new float[]{5.0f, 0.0f}, 1, null).hits().getFirst();
            assertEquals(newId, hit.vectorId());
            assertEquals("new", hit.metadata());
        }
    }

    @Test
    void comparesWithLuceneBuiltInVectorSearch() throws Exception {
        try (LuceneVectorAdapter adapter = adapter()) {
            adapter.add(doc("near", 0.0f, 0.0f, "group"));
            adapter.add(doc("middle", 1.0f, 0.0f, "group"));
            adapter.add(doc("far", 3.0f, 0.0f, "group"));
            adapter.refreshAndRebuild();

            List<String> vectorForge = adapter.search(new float[]{0.2f, 0.0f}, 3, null)
                    .hits().stream().map(LuceneVectorHit::externalId).toList();
            List<String> lucene = adapter.searchLucene(new float[]{0.2f, 0.0f}, 3, null)
                    .stream().map(LuceneVectorHit::externalId).toList();

            assertEquals(vectorForge, lucene);
        }
    }

    private static LuceneVectorAdapter adapter() throws Exception {
        return new LuceneVectorAdapter(2, DistanceMetric.EUCLIDEAN, new CpuBruteForceIndex());
    }

    private static LuceneVectorDocument doc(String id, float x, float y, String metadata) {
        return new LuceneVectorDocument(id, "text for " + id, new float[]{x, y}, metadata);
    }
}

package com.vectorforge.lucene;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.IndexMetrics;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;
import com.vectorforge.cpu.CpuBruteForceIndex;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
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

    @Test
    void failedRebuildKeepsThePublishedSnapshotsTogether() throws Exception {
        AtomicInteger creations = new AtomicInteger();
        try (LuceneVectorAdapter adapter = new LuceneVectorAdapter(2, DistanceMetric.EUCLIDEAN, () ->
                creations.getAndIncrement() == 1 ? new MutatingFailingIndex() : new CpuBruteForceIndex())) {
            adapter.add(doc("original", 0.0f, 0.0f, "old"));
            adapter.refreshAndRebuild();
            adapter.update(doc("original", 5.0f, 0.0f, "new"));

            assertThrows(IllegalStateException.class, adapter::refreshAndRebuild);
            LuceneVectorHit vectorForge = adapter.search(new float[]{0.0f, 0.0f}, 1, null).hits().getFirst();
            LuceneVectorHit lucene = adapter.searchLucene(new float[]{0.0f, 0.0f}, 1, null).getFirst();
            assertEquals("old", vectorForge.metadata());
            assertEquals(vectorForge.vectorId(), lucene.vectorId());
        }
    }

    @Test
    void deleteAllPublishesEmptyAndReleasesTheOldBackend() throws Exception {
        TrackingIndex.closed = 0;
        try (LuceneVectorAdapter adapter = new LuceneVectorAdapter(
                2, DistanceMetric.EUCLIDEAN, TrackingIndex::new)) {
            adapter.add(doc("one", 0.0f, 0.0f, "group"));
            adapter.refreshAndRebuild();
            adapter.delete("one");

            assertEquals(0, adapter.refreshAndRebuild());
            assertTrue(adapter.search(new float[]{0.0f, 0.0f}, 1, null).hits().isEmpty());
            assertTrue(adapter.searchLucene(new float[]{0.0f, 0.0f}, 1, null).isEmpty());
            assertEquals(1, TrackingIndex.closed);
        }
    }

    @Test
    void rejectsDuplicateExternalIds() throws Exception {
        try (LuceneVectorAdapter adapter = adapter()) {
            adapter.add(doc("duplicate", 0.0f, 0.0f, "group"));
            assertThrows(IllegalArgumentException.class,
                    () -> adapter.add(doc("duplicate", 1.0f, 0.0f, "group")));
        }
    }

    @Test
    void comparesCosineAndDotProductMappingsWithLucene() throws Exception {
        assertMetricOrder(
                DistanceMetric.COSINE,
                new float[][]{{1.0f, 0.0f}, {0.8f, 0.2f}, {0.0f, 1.0f}},
                new float[]{1.0f, 0.0f}
        );
        float diagonal = (float) (1.0 / Math.sqrt(2.0));
        assertMetricOrder(
                DistanceMetric.DOT_PRODUCT,
                new float[][]{{1.0f, 0.0f}, {diagonal, diagonal}, {0.0f, 1.0f}},
                new float[]{1.0f, 0.0f}
        );
    }

    @Test
    void rejectsNonUnitDotProductVectorsAndQueries() throws Exception {
        try (LuceneVectorAdapter adapter = new LuceneVectorAdapter(
                2, DistanceMetric.DOT_PRODUCT, CpuBruteForceIndex::new)) {
            assertThrows(IllegalArgumentException.class,
                    () -> adapter.add(doc("bad", 2.0f, 0.0f, "group")));
            adapter.add(doc("good", 1.0f, 0.0f, "group"));
            adapter.refreshAndRebuild();
            assertThrows(IllegalArgumentException.class,
                    () -> adapter.search(new float[]{0.5f, 0.0f}, 1, null));
            assertThrows(IllegalArgumentException.class,
                    () -> adapter.searchLucene(new float[]{0.5f, 0.0f}, 1, null));
        }
    }

    private static void assertMetricOrder(DistanceMetric metric, float[][] vectors, float[] query)
            throws Exception {
        try (LuceneVectorAdapter adapter = new LuceneVectorAdapter(2, metric, CpuBruteForceIndex::new)) {
            for (int i = 0; i < vectors.length; i++) {
                adapter.add(new LuceneVectorDocument("doc-" + i, "text " + i, vectors[i], "group"));
            }
            adapter.refreshAndRebuild();
            List<String> vectorForge = adapter.search(query, vectors.length, null).hits().stream()
                    .map(LuceneVectorHit::externalId).toList();
            List<String> lucene = adapter.searchLucene(query, vectors.length, null).stream()
                    .map(LuceneVectorHit::externalId).toList();
            assertEquals(vectorForge, lucene);
        }
    }

    private static LuceneVectorAdapter adapter() throws Exception {
        return new LuceneVectorAdapter(2, DistanceMetric.EUCLIDEAN, CpuBruteForceIndex::new);
    }

    private static LuceneVectorDocument doc(String id, float x, float y, String metadata) {
        return new LuceneVectorDocument(id, "text for " + id, new float[]{x, y}, metadata);
    }

    private static final class MutatingFailingIndex implements VectorIndex {
        @Override
        public void build(float[][] vectors, long[] ids) {
            throw new IllegalStateException("deliberate rebuild failure after mutation");
        }

        @Override
        public List<SearchResult> search(float[] query, int k, SearchParameters parameters) {
            throw new AssertionError("failed candidate index must never be published");
        }

        @Override
        public IndexMetrics metrics() {
            return new IndexMetrics("failing", false, false, 0, 0, 0, false);
        }

        @Override
        public void close() {
        }
    }

    private static final class TrackingIndex implements VectorIndex {
        private static int closed;
        private final CpuBruteForceIndex delegate = new CpuBruteForceIndex();

        @Override
        public void build(float[][] vectors, long[] ids) {
            delegate.build(vectors, ids);
        }

        @Override
        public List<SearchResult> search(float[] query, int k, SearchParameters parameters) {
            return delegate.search(query, k, parameters);
        }

        @Override
        public IndexMetrics metrics() {
            return delegate.metrics();
        }

        @Override
        public void close() {
            closed++;
            delegate.close();
        }
    }
}

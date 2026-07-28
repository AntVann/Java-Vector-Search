package com.vectorforge.lucene;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.cpu.CpuBruteForceIndex;

import java.util.List;
import java.util.Locale;

/**
 * Reproducible CPU-only demonstration of Lucene document storage plus VectorForge search.
 */
public final class LuceneVectorDemo {

    private LuceneVectorDemo() {
    }

    public static void main(String[] args) throws Exception {
        List<LuceneVectorDocument> documents = List.of(
                new LuceneVectorDocument("doc-1", "red bicycle in the city",
                        new float[]{0.95f, 0.05f, 0.00f}, "transport"),
                new LuceneVectorDocument("doc-2", "blue electric train",
                        new float[]{0.80f, 0.20f, 0.05f}, "transport"),
                new LuceneVectorDocument("doc-3", "fresh apple and pear",
                        new float[]{0.05f, 0.90f, 0.10f}, "food"),
                new LuceneVectorDocument("doc-4", "bread from the bakery",
                        new float[]{0.10f, 0.75f, 0.20f}, "food")
        );
        float[] query = {1.0f, 0.0f, 0.0f};

        try (LuceneVectorAdapter adapter = new LuceneVectorAdapter(
                3, DistanceMetric.EUCLIDEAN, CpuBruteForceIndex::new)) {
            for (LuceneVectorDocument document : documents) {
                long vectorId = adapter.add(document);
                System.out.printf("indexed vectorId=%d externalId=%s%n", vectorId, document.externalId());
            }
            System.out.printf("refresh rebuilt %d live documents%n", adapter.refreshAndRebuild());

            LuceneVectorSearchResponse vectorForge = adapter.search(query, 3, "transport");
            List<LuceneVectorHit> lucene = adapter.searchLucene(query, 3, "transport");

            System.out.println("\nVectorForge (metadata filtered after vector search):");
            printHits(vectorForge.hits());
            System.out.printf(Locale.ROOT, "raw backend: %.3f ms; end-to-end: %.3f ms%n",
                    vectorForge.rawBackendNanos() / 1_000_000.0,
                    vectorForge.endToEndNanos() / 1_000_000.0);

            System.out.println("\nLucene built-in k-NN (metadata filtered during vector search):");
            printHits(lucene);
        }
    }

    private static void printHits(List<LuceneVectorHit> hits) {
        for (int rank = 0; rank < hits.size(); rank++) {
            LuceneVectorHit hit = hits.get(rank);
            System.out.printf(Locale.ROOT, "%d. %s vectorId=%d metadata=%s score=%.6f text=%s%n",
                    rank + 1, hit.externalId(), hit.vectorId(), hit.metadata(), hit.score(), hit.text());
        }
    }
}

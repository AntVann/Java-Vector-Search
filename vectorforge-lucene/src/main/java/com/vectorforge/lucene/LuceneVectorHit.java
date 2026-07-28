package com.vectorforge.lucene;

/**
 * A VectorForge or Lucene vector hit resolved back to its document fields.
 */
public record LuceneVectorHit(
        long vectorId,
        String externalId,
        String text,
        String metadata,
        float score
) {
}

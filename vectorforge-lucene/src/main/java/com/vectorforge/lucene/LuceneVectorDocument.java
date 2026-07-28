package com.vectorforge.lucene;

import java.util.Objects;

/**
 * Input document shared by Lucene and VectorForge.
 */
public record LuceneVectorDocument(
        String externalId,
        String text,
        float[] vector,
        String metadata
) {
    public LuceneVectorDocument {
        if (externalId == null || externalId.isBlank()) {
            throw new IllegalArgumentException("externalId must not be blank");
        }
        Objects.requireNonNull(text, "text must not be null");
        Objects.requireNonNull(vector, "vector must not be null");
        if (vector.length == 0) {
            throw new IllegalArgumentException("vector must not be empty");
        }
        Objects.requireNonNull(metadata, "metadata must not be null");
        vector = vector.clone();
    }

    @Override
    public float[] vector() {
        return vector.clone();
    }
}

package com.vectorforge.api;

/**
 * A single nearest-neighbor result.
 *
 * @param id unique vector identifier
 * @param score distance or similarity score
 */
public record SearchResult(long id, float score) {
}


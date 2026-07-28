package com.vectorforge.disk;

/**
 * Non-overlapping query phase timings. A value of -1 means the backend cannot expose that phase.
 */
public record DiskIvfSearchTimings(
        long centroidSearchNanos,
        long diskReadNanos,
        long candidateLoadNanos,
        long backendTransferNanos,
        long backendSearchNanos,
        long deviceToHostNanos,
        long endToEndNanos
) {
}

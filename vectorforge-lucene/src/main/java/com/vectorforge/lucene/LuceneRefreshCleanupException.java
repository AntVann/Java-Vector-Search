package com.vectorforge.lucene;

import java.io.IOException;

/**
 * Reports failure to release a retired snapshot after a replacement snapshot was published.
 *
 * <p>The refresh is committed when this exception is thrown. Callers must not treat it as a
 * rollback signal or retry the same logical mutation.</p>
 */
public final class LuceneRefreshCleanupException extends IOException {

    private final int liveDocumentCount;

    public LuceneRefreshCleanupException(int liveDocumentCount, IOException cause) {
        super("refresh published " + liveDocumentCount
                + " live documents, but retired snapshot cleanup failed", cause);
        this.liveDocumentCount = liveDocumentCount;
    }

    public int liveDocumentCount() {
        return liveDocumentCount;
    }

    public boolean published() {
        return true;
    }
}

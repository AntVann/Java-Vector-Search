package com.vectorforge.lucene;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MultiBits;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Bits;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Standalone adapter that keeps Lucene documents and a VectorForge index in sync at explicit
 * refresh boundaries. It uses only public Lucene APIs.
 */
public final class LuceneVectorAdapter implements AutoCloseable {

    public static final String EXTERNAL_ID_FIELD = "external_id";
    public static final String TEXT_FIELD = "text";
    public static final String VECTOR_FIELD = "dense_vector";
    public static final String METADATA_FIELD = "metadata";
    public static final String VECTOR_ID_FIELD = "vectorforge_id";
    private static final String STORED_VECTOR_FIELD = "vectorforge_vector";

    private final int dimensions;
    private final DistanceMetric metric;
    private final Supplier<? extends VectorIndex> vectorIndexFactory;
    private final Directory directory;
    private final Analyzer analyzer;
    private final IndexWriter writer;
    private final Map<Long, SnapshotDocument> documentsByVectorId = new LinkedHashMap<>();
    private final Set<String> externalIds = new HashSet<>();
    private long nextVectorId = 1L;
    private VectorIndex vectorIndex;
    private DirectoryReader reader;
    private IndexSearcher searcher;
    private boolean vectorIndexBuilt;
    private boolean closed;

    public LuceneVectorAdapter(
            int dimensions,
            DistanceMetric metric,
            Supplier<? extends VectorIndex> vectorIndexFactory
    ) throws IOException {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions must be positive");
        }
        this.dimensions = dimensions;
        this.metric = Objects.requireNonNull(metric, "metric must not be null");
        this.vectorIndexFactory = Objects.requireNonNull(
                vectorIndexFactory, "vectorIndexFactory must not be null");
        this.directory = new ByteBuffersDirectory();
        this.analyzer = new StandardAnalyzer();
        this.writer = new IndexWriter(directory, new IndexWriterConfig(analyzer));
    }

    /**
     * Adds a document and returns its stable VectorForge ID. The document becomes searchable
     * after {@link #refreshAndRebuild()}.
     */
    public synchronized long add(LuceneVectorDocument input) throws IOException {
        ensureOpen();
        Objects.requireNonNull(input, "input must not be null");
        validateVector(input.vector());
        if (externalIds.contains(input.externalId())) {
            throw new IllegalArgumentException("externalId already exists: " + input.externalId());
        }
        long vectorId = nextVectorId++;
        writer.addDocument(toDocument(input, vectorId));
        externalIds.add(input.externalId());
        return vectorId;
    }

    /**
     * Replaces any document with the same external ID and assigns a new VectorForge ID.
     */
    public synchronized long update(LuceneVectorDocument input) throws IOException {
        ensureOpen();
        Objects.requireNonNull(input, "input must not be null");
        validateVector(input.vector());
        long vectorId = nextVectorId++;
        Document replacement = toDocument(input, vectorId);
        writer.updateDocument(new Term(EXTERNAL_ID_FIELD, input.externalId()), replacement);
        externalIds.add(input.externalId());
        return vectorId;
    }

    public synchronized void delete(String externalId) throws IOException {
        ensureOpen();
        if (externalId == null || externalId.isBlank()) {
            throw new IllegalArgumentException("externalId must not be blank");
        }
        writer.deleteDocuments(new Term(EXTERNAL_ID_FIELD, externalId));
        externalIds.remove(externalId);
    }

    /**
     * Opens a new near-real-time Lucene reader and rebuilds VectorForge from Lucene's live
     * documents. Deletes and updates are therefore applied atomically at this boundary.
     * Candidate preparation failures leave the prior snapshot active. A
     * {@link LuceneRefreshCleanupException} means the replacement snapshot is active and only
     * cleanup of retired resources failed.
     */
    public synchronized int refreshAndRebuild() throws IOException {
        ensureOpen();
        DirectoryReader newReader = DirectoryReader.open(writer);
        IndexSearcher newSearcher = new IndexSearcher(newReader);
        Map<Long, SnapshotDocument> snapshot = new LinkedHashMap<>();
        List<float[]> vectors = new ArrayList<>();
        List<Long> ids = new ArrayList<>();
        VectorIndex newVectorIndex = null;
        try {
            Bits liveDocs = MultiBits.getLiveDocs(newReader);
            for (int docId = 0; docId < newReader.maxDoc(); docId++) {
                if (liveDocs != null && !liveDocs.get(docId)) {
                    continue;
                }
                Document document = newSearcher.storedFields().document(docId);
                long vectorId = document.getField(VECTOR_ID_FIELD).numericValue().longValue();
                float[] vector = decodeVector(document.getBinaryValue(STORED_VECTOR_FIELD));
                SnapshotDocument value = new SnapshotDocument(
                        vectorId,
                        document.get(EXTERNAL_ID_FIELD),
                        document.get(TEXT_FIELD),
                        document.get(METADATA_FIELD)
                );
                snapshot.put(vectorId, value);
                vectors.add(vector);
                ids.add(vectorId);
            }
            newVectorIndex = Objects.requireNonNull(
                    vectorIndexFactory.get(), "vectorIndexFactory returned null");
            if (!vectors.isEmpty()) {
                newVectorIndex.build(
                        vectors.toArray(float[][]::new),
                        ids.stream().mapToLong(Long::longValue).toArray()
                );
            }
        } catch (RuntimeException | IOException error) {
            try {
                newReader.close();
            } catch (IOException closeError) {
                error.addSuppressed(closeError);
            }
            if (newVectorIndex != null) {
                try {
                    newVectorIndex.close();
                } catch (RuntimeException closeError) {
                    error.addSuppressed(closeError);
                }
            }
            throw error;
        }

        DirectoryReader oldReader = reader;
        VectorIndex oldVectorIndex = vectorIndex;
        reader = newReader;
        searcher = newSearcher;
        vectorIndex = newVectorIndex;
        vectorIndexBuilt = !vectors.isEmpty();
        documentsByVectorId.clear();
        documentsByVectorId.putAll(snapshot);
        externalIds.clear();
        snapshot.values().forEach(document -> externalIds.add(document.externalId()));
        IOException cleanupFailure = null;
        if (oldReader != null) {
            try {
                oldReader.close();
            } catch (IOException error) {
                cleanupFailure = error;
            }
        }
        if (oldVectorIndex != null) {
            try {
                oldVectorIndex.close();
            } catch (RuntimeException error) {
                if (cleanupFailure == null) {
                    cleanupFailure = new IOException("failed to close previous VectorForge index", error);
                } else {
                    cleanupFailure.addSuppressed(error);
                }
            }
        }
        if (cleanupFailure != null) {
            throw new LuceneRefreshCleanupException(snapshot.size(), cleanupFailure);
        }
        return snapshot.size();
    }

    /**
     * Searches VectorForge and resolves IDs to Lucene fields. Metadata filtering is applied
     * after vector search. All live candidates are requested so filtering does not reduce k
     * incorrectly; this is deliberately simple, but expensive for large indexes.
     */
    public synchronized LuceneVectorSearchResponse search(float[] query, int k, String metadataFilter) {
        long endToEndStart = System.nanoTime();
        ensureOpen();
        validateSearch(query, k);
        if (!vectorIndexBuilt || documentsByVectorId.isEmpty()) {
            return new LuceneVectorSearchResponse(List.of(), 0L, System.nanoTime() - endToEndStart);
        }

        long rawStart = System.nanoTime();
        List<SearchResult> candidates = vectorIndex.search(
                query,
                documentsByVectorId.size(),
                new SearchParameters(metric)
        );
        long rawNanos = System.nanoTime() - rawStart;

        List<LuceneVectorHit> hits = new ArrayList<>(Math.min(k, candidates.size()));
        for (SearchResult candidate : candidates) {
            SnapshotDocument document = documentsByVectorId.get(candidate.id());
            if (document == null) {
                continue;
            }
            if (metadataFilter != null && !metadataFilter.equals(document.metadata())) {
                continue;
            }
            hits.add(document.toHit(candidate.score()));
            if (hits.size() == k) {
                break;
            }
        }
        return new LuceneVectorSearchResponse(hits, rawNanos, System.nanoTime() - endToEndStart);
    }

    /**
     * Runs Lucene's built-in k-NN query. Its optional metadata TermQuery is applied during
     * Lucene vector search.
     */
    public synchronized List<LuceneVectorHit> searchLucene(float[] query, int k, String metadataFilter)
            throws IOException {
        ensureOpen();
        validateSearch(query, k);
        if (searcher == null || documentsByVectorId.isEmpty()) {
            return List.of();
        }
        Query filter = metadataFilter == null ? null : new TermQuery(new Term(METADATA_FIELD, metadataFilter));
        TopDocs topDocs = searcher.search(new KnnFloatVectorQuery(VECTOR_FIELD, query, k, filter), k);
        List<LuceneVectorHit> hits = new ArrayList<>(topDocs.scoreDocs.length);
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            Document document = searcher.storedFields().document(scoreDoc.doc);
            hits.add(new LuceneVectorHit(
                    document.getField(VECTOR_ID_FIELD).numericValue().longValue(),
                    document.get(EXTERNAL_ID_FIELD),
                    document.get(TEXT_FIELD),
                    document.get(METADATA_FIELD),
                    scoreDoc.score
            ));
        }
        return List.copyOf(hits);
    }

    public synchronized int liveDocumentCount() {
        ensureOpen();
        return documentsByVectorId.size();
    }

    private Document toDocument(LuceneVectorDocument input, long vectorId) {
        float[] vector = input.vector();
        Document document = new Document();
        document.add(new StringField(EXTERNAL_ID_FIELD, input.externalId(), Field.Store.YES));
        document.add(new TextField(TEXT_FIELD, input.text(), Field.Store.YES));
        document.add(new StringField(METADATA_FIELD, input.metadata(), Field.Store.YES));
        document.add(new StoredField(VECTOR_ID_FIELD, vectorId));
        document.add(new StoredField(STORED_VECTOR_FIELD, encodeVector(vector)));
        document.add(new KnnFloatVectorField(VECTOR_FIELD, vector, luceneSimilarity(metric)));
        return document;
    }

    private void validateSearch(float[] query, int k) {
        Objects.requireNonNull(query, "query must not be null");
        validateVector(query);
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
    }

    private void validateVector(float[] vector) {
        if (vector.length != dimensions) {
            throw new IllegalArgumentException(
                    "vector dimension mismatch: expected " + dimensions + " but was " + vector.length);
        }
        for (float value : vector) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("vectors must contain only finite values");
            }
        }
        if (metric == DistanceMetric.DOT_PRODUCT) {
            double squaredMagnitude = 0.0;
            for (float value : vector) {
                squaredMagnitude += (double) value * value;
            }
            if (Math.abs(squaredMagnitude - 1.0) > 1.0e-5) {
                throw new IllegalArgumentException("dot-product vectors must have unit length");
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("adapter is closed");
        }
    }

    private static VectorSimilarityFunction luceneSimilarity(DistanceMetric metric) {
        return switch (metric) {
            case EUCLIDEAN -> VectorSimilarityFunction.EUCLIDEAN;
            case COSINE -> VectorSimilarityFunction.COSINE;
            case DOT_PRODUCT -> VectorSimilarityFunction.DOT_PRODUCT;
        };
    }

    private static BytesRef encodeVector(float[] vector) {
        ByteBuffer buffer = ByteBuffer.allocate(Math.multiplyExact(vector.length, Float.BYTES))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (float value : vector) {
            buffer.putFloat(value);
        }
        return new BytesRef(buffer.array());
    }

    private float[] decodeVector(BytesRef bytes) {
        if (bytes == null || bytes.length != Math.multiplyExact(dimensions, Float.BYTES)) {
            throw new IllegalStateException("stored vector is missing or has the wrong size");
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes.bytes, bytes.offset, bytes.length).order(ByteOrder.LITTLE_ENDIAN);
        float[] vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            vector[i] = buffer.getFloat();
        }
        return vector;
    }

    @Override
    public synchronized void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        IOException failure = null;
        try {
            if (reader != null) {
                reader.close();
            }
        } catch (IOException error) {
            failure = error;
        }
        try {
            writer.close();
        } catch (IOException error) {
            if (failure == null) failure = error;
            else failure.addSuppressed(error);
        }
        try {
            analyzer.close();
        } catch (RuntimeException error) {
            if (failure == null) failure = new IOException("failed to close analyzer", error);
            else failure.addSuppressed(error);
        }
        try {
            directory.close();
        } catch (IOException error) {
            if (failure == null) failure = error;
            else failure.addSuppressed(error);
        }
        try {
            if (vectorIndex != null) {
                vectorIndex.close();
            }
        } catch (RuntimeException error) {
            if (failure == null) failure = new IOException("failed to close VectorForge index", error);
            else failure.addSuppressed(error);
        }
        if (failure != null) {
            throw failure;
        }
    }

    private record SnapshotDocument(long vectorId, String externalId, String text, String metadata) {
        private LuceneVectorHit toHit(float score) {
            return new LuceneVectorHit(vectorId, externalId, text, metadata, score);
        }
    }
}

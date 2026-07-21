package com.vectorforge.cpu;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.IndexMetrics;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Exact brute-force CPU baseline used as the correctness reference for future backends.
 */
public final class CpuBruteForceIndex implements VectorIndex {

    private static final String BACKEND_NAME = "cpu-bruteforce";

    private volatile IndexData data;
    private volatile boolean closed;

    @Override
    public synchronized void build(float[][] vectors, long[] ids) {
        ensureOpen();
        IndexData newData = validateAndBuildIndex(vectors, ids);
        data = newData;
    }

    @Override
    public List<SearchResult> search(float[] query, int k, SearchParameters parameters) {
        Objects.requireNonNull(parameters, "parameters must not be null");
        IndexData snapshot = requireBuilt();
        validateQuery(query, snapshot.dimensions);
        validateK(k, snapshot.vectorCount);

        DistanceMetric metric = parameters.metric();
        float queryNorm = metric == DistanceMetric.COSINE ? norm(query) : 0.0f;

        long[] heapIds = new long[k];
        float[] heapScores = new float[k];
        int heapSize = 0;

        for (int vectorIndex = 0; vectorIndex < snapshot.vectorCount; vectorIndex++) {
            int offset = vectorIndex * snapshot.dimensions;
            float score = computeScore(snapshot, query, offset, metric, queryNorm, vectorIndex);
            long id = snapshot.ids[vectorIndex];

            if (heapSize < k) {
                heapIds[heapSize] = id;
                heapScores[heapSize] = score;
                siftUp(heapIds, heapScores, heapSize, metric);
                heapSize++;
                continue;
            }

            if (isBetter(score, id, heapScores[0], heapIds[0], metric)) {
                heapIds[0] = id;
                heapScores[0] = score;
                siftDown(heapIds, heapScores, 0, heapSize, metric);
            }
        }

        sortBestFirst(heapIds, heapScores, heapSize, metric);
        ArrayList<SearchResult> results = new ArrayList<>(heapSize);
        for (int i = 0; i < heapSize; i++) {
            results.add(new SearchResult(heapIds[i], heapScores[i]));
        }
        return List.copyOf(results);
    }

    /**
     * Searches multiple queries using the same index snapshot.
     *
     * @param queries query vectors
     * @param k number of results per query
     * @param parameters search controls
     * @return ordered search results for each query
     */
    public List<List<SearchResult>> searchBatch(float[][] queries, int k, SearchParameters parameters) {
        Objects.requireNonNull(queries, "queries must not be null");
        ArrayList<List<SearchResult>> batchedResults = new ArrayList<>(queries.length);
        for (float[] query : queries) {
            batchedResults.add(search(query, k, parameters));
        }
        return List.copyOf(batchedResults);
    }

    @Override
    public IndexMetrics metrics() {
        IndexData snapshot = data;
        if (snapshot == null) {
            return new IndexMetrics(BACKEND_NAME, false, closed, 0, 0, 0, false);
        }
        return new IndexMetrics(
                BACKEND_NAME,
                true,
                closed,
                snapshot.vectorCount,
                snapshot.dimensions,
                snapshot.vectorBytes,
                false
        );
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        data = null;
    }

    private IndexData requireBuilt() {
        ensureOpen();
        IndexData snapshot = data;
        if (snapshot == null) {
            throw new IllegalStateException("index has not been built");
        }
        return snapshot;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("index has been closed");
        }
    }

    private static IndexData validateAndBuildIndex(float[][] vectors, long[] ids) {
        Objects.requireNonNull(vectors, "vectors must not be null");
        Objects.requireNonNull(ids, "ids must not be null");
        if (vectors.length == 0) {
            throw new IllegalArgumentException("vectors must not be empty");
        }
        if (vectors.length != ids.length) {
            throw new IllegalArgumentException("vectors length must match ids length");
        }

        int dimensions = -1;
        int vectorCount = vectors.length;
        float[] flattened;
        float[] norms;
        long[] copiedIds = Arrays.copyOf(ids, ids.length);
        Set<Long> seenIds = new HashSet<>(Math.max(16, vectorCount * 2));

        for (int i = 0; i < copiedIds.length; i++) {
            if (!seenIds.add(copiedIds[i])) {
                throw new IllegalArgumentException("duplicate id detected: " + copiedIds[i]);
            }
        }

        for (int row = 0; row < vectors.length; row++) {
            float[] vector = vectors[row];
            if (vector == null) {
                throw new IllegalArgumentException("vector at index " + row + " must not be null");
            }
            if (dimensions == -1) {
                if (vector.length == 0) {
                    throw new IllegalArgumentException("vector dimensions must be positive");
                }
                dimensions = vector.length;
            } else if (vector.length != dimensions) {
                throw new IllegalArgumentException("vector at index " + row + " has dimension " + vector.length
                        + " but expected " + dimensions);
            }
        }

        flattened = new float[vectorCount * dimensions];
        norms = new float[vectorCount];

        for (int row = 0; row < vectorCount; row++) {
            float[] vector = vectors[row];
            System.arraycopy(vector, 0, flattened, row * dimensions, dimensions);
            norms[row] = norm(vector);
        }

        long vectorBytes = (long) vectorCount * dimensions * Float.BYTES;
        return new IndexData(flattened, norms, copiedIds, vectorCount, dimensions, vectorBytes);
    }

    private static void validateQuery(float[] query, int expectedDimensions) {
        Objects.requireNonNull(query, "query must not be null");
        if (query.length != expectedDimensions) {
            throw new IllegalArgumentException("query dimension " + query.length
                    + " does not match index dimension " + expectedDimensions);
        }
    }

    private static void validateK(int k, int vectorCount) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        if (k > vectorCount) {
            throw new IllegalArgumentException("k must be <= vector count");
        }
    }

    private static float computeScore(
            IndexData data,
            float[] query,
            int vectorOffset,
            DistanceMetric metric,
            float queryNorm,
            int vectorIndex
    ) {
        return switch (metric) {
            case EUCLIDEAN -> squaredEuclideanDistance(data.vectors, vectorOffset, query, data.dimensions);
            case DOT_PRODUCT -> dotProduct(data.vectors, vectorOffset, query, data.dimensions);
            case COSINE -> cosineSimilarity(data.vectors, vectorOffset, query, data.dimensions, data.norms[vectorIndex], queryNorm);
        };
    }

    private static float squaredEuclideanDistance(float[] vectors, int offset, float[] query, int dimensions) {
        float sum = 0.0f;
        for (int i = 0; i < dimensions; i++) {
            float delta = vectors[offset + i] - query[i];
            sum += delta * delta;
        }
        return sum;
    }

    private static float dotProduct(float[] vectors, int offset, float[] query, int dimensions) {
        float sum = 0.0f;
        for (int i = 0; i < dimensions; i++) {
            sum += vectors[offset + i] * query[i];
        }
        return sum;
    }

    private static float cosineSimilarity(
            float[] vectors,
            int offset,
            float[] query,
            int dimensions,
            float vectorNorm,
            float queryNorm
    ) {
        if (vectorNorm == 0.0f || queryNorm == 0.0f) {
            return 0.0f;
        }
        return dotProduct(vectors, offset, query, dimensions) / (vectorNorm * queryNorm);
    }

    private static float norm(float[] vector) {
        float sum = 0.0f;
        for (float value : vector) {
            sum += value * value;
        }
        return (float) Math.sqrt(sum);
    }

    private static boolean isBetter(float candidateScore, long candidateId, float currentScore, long currentId, DistanceMetric metric) {
        if (candidateScore == currentScore) {
            return candidateId < currentId;
        }
        return metric == DistanceMetric.EUCLIDEAN
                ? candidateScore < currentScore
                : candidateScore > currentScore;
    }

    private static boolean isWorse(float leftScore, long leftId, float rightScore, long rightId, DistanceMetric metric) {
        if (leftScore == rightScore) {
            return leftId > rightId;
        }
        return metric == DistanceMetric.EUCLIDEAN
                ? leftScore > rightScore
                : leftScore < rightScore;
    }

    private static void siftUp(long[] ids, float[] scores, int index, DistanceMetric metric) {
        int current = index;
        while (current > 0) {
            int parent = (current - 1) >>> 1;
            if (!isWorse(scores[current], ids[current], scores[parent], ids[parent], metric)) {
                return;
            }
            swap(ids, scores, current, parent);
            current = parent;
        }
    }

    private static void siftDown(long[] ids, float[] scores, int index, int size, DistanceMetric metric) {
        int current = index;
        while (true) {
            int left = (current << 1) + 1;
            if (left >= size) {
                return;
            }
            int right = left + 1;
            int worstChild = left;
            if (right < size && isWorse(scores[right], ids[right], scores[left], ids[left], metric)) {
                worstChild = right;
            }
            if (!isWorse(scores[worstChild], ids[worstChild], scores[current], ids[current], metric)) {
                return;
            }
            swap(ids, scores, current, worstChild);
            current = worstChild;
        }
    }

    private static void sortBestFirst(long[] ids, float[] scores, int size, DistanceMetric metric) {
        for (int i = 1; i < size; i++) {
            long currentId = ids[i];
            float currentScore = scores[i];
            int j = i - 1;
            while (j >= 0 && isBetter(currentScore, currentId, scores[j], ids[j], metric)) {
                ids[j + 1] = ids[j];
                scores[j + 1] = scores[j];
                j--;
            }
            ids[j + 1] = currentId;
            scores[j + 1] = currentScore;
        }
    }

    private static void swap(long[] ids, float[] scores, int left, int right) {
        long id = ids[left];
        ids[left] = ids[right];
        ids[right] = id;

        float score = scores[left];
        scores[left] = scores[right];
        scores[right] = score;
    }

    private record IndexData(
            float[] vectors,
            float[] norms,
            long[] ids,
            int vectorCount,
            int dimensions,
            long vectorBytes
    ) {
    }
}


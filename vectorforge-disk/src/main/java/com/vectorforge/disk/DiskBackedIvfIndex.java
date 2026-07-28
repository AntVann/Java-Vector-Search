package com.vectorforge.disk;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.IndexMetrics;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.function.Supplier;
import java.util.zip.CRC32C;

/**
 * Experimental immutable disk-backed IVF index. The format and lifecycle are intentionally small
 * and auditable; this is not an online database engine.
 */
public final class DiskBackedIvfIndex implements VectorIndex {

    private static final byte[] MANIFEST_MAGIC = "VFIVFIDX".getBytes(StandardCharsets.US_ASCII);
    private static final byte[] CENTROIDS_MAGIC = "VFIVFCN1".getBytes(StandardCharsets.US_ASCII);
    private static final byte[] PARTITION_MAGIC = "VFIVFPT1".getBytes(StandardCharsets.US_ASCII);
    private static final int FORMAT_VERSION = 1;
    private static final int MANIFEST_FIXED_BYTES = 76;
    private static final int MANIFEST_ENTRY_BYTES = 28;
    private static final int CENTROID_HEADER_BYTES = 40;
    private static final int PARTITION_HEADER_BYTES = 44;
    private static final String CURRENT = "CURRENT";
    private static final String READY = "READY";
    private static final String GENERATIONS = "generations";

    private final Path root;
    private final DiskIvfBuildConfig buildConfig;
    private final DiskIvfSearchConfig searchConfig;
    private final Supplier<? extends VectorIndex> rerankerFactory;
    private final PartitionCache cache;
    private Generation generation;
    private boolean closed;

    private DiskBackedIvfIndex(
            Path root,
            DiskIvfBuildConfig buildConfig,
            DiskIvfSearchConfig searchConfig,
            Supplier<? extends VectorIndex> rerankerFactory,
            Generation generation
    ) {
        this.root = root;
        this.buildConfig = buildConfig;
        this.searchConfig = searchConfig;
        this.rerankerFactory = rerankerFactory;
        this.generation = generation;
        this.cache = new PartitionCache(searchConfig.cacheBytes());
    }

    public static DiskBackedIvfIndex create(
            Path root,
            float[][] vectors,
            long[] ids,
            DiskIvfBuildConfig buildConfig,
            DiskIvfSearchConfig searchConfig,
            Supplier<? extends VectorIndex> rerankerFactory
    ) throws IOException {
        Objects.requireNonNull(root, "root must not be null");
        Objects.requireNonNull(buildConfig, "buildConfig must not be null");
        Objects.requireNonNull(searchConfig, "searchConfig must not be null");
        Objects.requireNonNull(rerankerFactory, "rerankerFactory must not be null");
        writeGeneration(root, vectors, ids, buildConfig);
        return open(root, searchConfig, rerankerFactory);
    }

    public static DiskBackedIvfIndex open(
            Path root,
            DiskIvfSearchConfig searchConfig,
            Supplier<? extends VectorIndex> rerankerFactory
    ) throws IOException {
        Objects.requireNonNull(root, "root must not be null");
        Objects.requireNonNull(searchConfig, "searchConfig must not be null");
        Objects.requireNonNull(rerankerFactory, "rerankerFactory must not be null");
        Generation generation = readGeneration(root);
        DiskIvfBuildConfig buildConfig = new DiskIvfBuildConfig(
                generation.manifest().centroidCount(),
                generation.manifest().trainingIterations(),
                generation.manifest().seed(),
                generation.manifest().metric()
        );
        return new DiskBackedIvfIndex(
                root, buildConfig, searchConfig, rerankerFactory, generation);
    }

    /**
     * Convenience rebuild for in-memory input. Publication creates a new immutable generation.
     */
    @Override
    public synchronized void build(float[][] vectors, long[] ids) {
        ensureOpen();
        try {
            writeGeneration(root, vectors, ids, buildConfig);
            Generation replacement = readGeneration(root);
            generation = replacement;
            cache.clear();
        } catch (IOException error) {
            throw new IllegalStateException("failed to build disk IVF index", error);
        }
    }

    @Override
    public synchronized List<SearchResult> search(float[] query, int k, SearchParameters parameters) {
        return searchDetailed(query, k, parameters).results();
    }

    public synchronized DiskIvfSearchResponse searchDetailed(
            float[] query,
            int k,
            SearchParameters parameters
    ) {
        ensureOpen();
        Objects.requireNonNull(parameters, "parameters must not be null");
        Manifest manifest = generation.manifest();
        validateQuery(query, k, parameters, manifest);
        long endStart = System.nanoTime();

        long centroidStart = System.nanoTime();
        int probes = Math.min(searchConfig.partitionsToProbe(), manifest.partitionCount());
        int[] selected = selectCentroids(query, generation.centroids(), manifest, probes);
        long centroidNanos = System.nanoTime() - centroidStart;

        long recordBytes = Math.addExact(Long.BYTES,
                Math.multiplyExact((long) manifest.dimensions(), Float.BYTES));
        long estimatedBackendVectorBytes = Math.addExact(recordBytes, 32);
        long maxBatchVectors = searchConfig.maxBackendBatchBytes() / estimatedBackendVectorBytes;
        if (maxBatchVectors == 0) {
            throw new IllegalArgumentException(
                    "maxBackendBatchBytes cannot hold one candidate vector");
        }
        int batchCapacity = Math.toIntExact(Math.min(maxBatchVectors, Integer.MAX_VALUE));
        long candidateCount = 0;
        long diskNanos = 0;
        long loadNanos = 0;
        long transferNanos = 0;
        long backendNanos = 0;
        long hitsBefore = cache.hits;
        long missesBefore = cache.misses;
        List<SearchResult> chunkWinners = new ArrayList<>();

        for (int partitionId : selected) {
            PartitionEntry entry = manifest.partitions().get(partitionId);
            PartitionData cached = cache.get(partitionId);
            try {
                if (cached == null && cacheWeight(entry.vectorCount(), manifest.dimensions())
                        <= searchConfig.cacheBytes()) {
                    LoadResult loaded = readPartitionRecords(
                            generation.directory().resolve(partitionFile(partitionId)),
                            generation,
                            0,
                            Math.toIntExact(entry.vectorCount())
                    );
                    cached = loaded.data();
                    cache.put(
                            partitionId,
                            cached,
                            cacheWeight(entry.vectorCount(), manifest.dimensions())
                    );
                    diskNanos += loaded.diskReadNanos();
                    loadNanos += loaded.candidateLoadNanos();
                }
                long offset = 0;
                while (offset < entry.vectorCount()) {
                    int count = Math.toIntExact(Math.min(
                            batchCapacity, entry.vectorCount() - offset));
                    PartitionData batch;
                    if (cached != null) {
                        long copyStart = System.nanoTime();
                        batch = cached.slice(Math.toIntExact(offset), count);
                        loadNanos += System.nanoTime() - copyStart;
                    } else {
                        LoadResult loaded = readPartitionRecords(
                                generation.directory().resolve(partitionFile(partitionId)),
                                generation,
                                offset,
                                count
                        );
                        batch = loaded.data();
                        diskNanos += loaded.diskReadNanos();
                        loadNanos += loaded.candidateLoadNanos();
                    }
                    ChunkResult chunk = rerankChunk(batch, query, k, parameters, manifest.dimensions());
                    loadNanos += chunk.candidateLoadNanos();
                    transferNanos += chunk.backendTransferNanos();
                    backendNanos += chunk.backendSearchNanos();
                    chunkWinners.addAll(chunk.results());
                    chunkWinners.sort(Comparator
                            .comparingDouble(SearchResult::score)
                            .thenComparingLong(SearchResult::id));
                    if (chunkWinners.size() > k) {
                        chunkWinners.subList(k, chunkWinners.size()).clear();
                    }
                    candidateCount += count;
                    offset += count;
                }
            } catch (IOException error) {
                throw new IllegalStateException("failed to load partition " + partitionId, error);
            }
        }

        List<SearchResult> results = List.copyOf(chunkWinners);
        long endNanos = System.nanoTime() - endStart;
        return new DiskIvfSearchResponse(
                results,
                new DiskIvfSearchTimings(
                        centroidNanos,
                        diskNanos,
                        loadNanos,
                        candidateCount == 0 ? -1 : transferNanos,
                        backendNanos,
                        -1,
                        endNanos
                ),
                selected.length,
                candidateCount,
                Math.multiplyExact(candidateCount, recordBytes),
                cache.hits - hitsBefore,
                cache.misses - missesBefore
        );
    }

    @Override
    public synchronized IndexMetrics metrics() {
        Manifest manifest = generation.manifest();
        return new IndexMetrics(
                "disk-ivf-experimental",
                true,
                closed,
                manifest.vectorCount(),
                manifest.dimensions(),
                Math.multiplyExact(
                        Math.multiplyExact(manifest.vectorCount(), manifest.dimensions()),
                        Float.BYTES
                ),
                false
        );
    }

    public synchronized DiskIvfMetrics diskMetrics() {
        return new DiskIvfMetrics(
                generation.id().toString(),
                generation.manifest().centroidCount(),
                generation.manifest().partitionCount(),
                generation.manifest().onDiskBytes(),
                cache.entries.size(),
                cache.residentBytes,
                cache.hits,
                cache.misses,
                cache.evictions
        );
    }

    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        cache.clear();
    }

    private static LoadResult readPartitionRecords(
            Path path,
            Generation generation,
            long recordOffset,
            int count
    ) throws IOException {
        int dimensions = generation.manifest().dimensions();
        int recordBytes = Math.toIntExact(Math.addExact(
                Long.BYTES, Math.multiplyExact((long) dimensions, Float.BYTES)));
        long bytes = Math.multiplyExact((long) count, recordBytes);
        ByteBuffer buffer = ByteBuffer.allocate(Math.toIntExact(bytes)).order(ByteOrder.LITTLE_ENDIAN);
        long diskStart = System.nanoTime();
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            readFully(channel, buffer, Math.addExact(
                    PARTITION_HEADER_BYTES, Math.multiplyExact(recordOffset, recordBytes)));
        }
        long diskNanos = System.nanoTime() - diskStart;
        long decodeStart = System.nanoTime();
        buffer.flip();
        long[] ids = new long[count];
        float[] vectors = new float[Math.multiplyExact(count, dimensions)];
        for (int row = 0; row < count; row++) {
            ids[row] = buffer.getLong();
            for (int column = 0; column < dimensions; column++) {
                vectors[row * dimensions + column] = buffer.getFloat();
            }
        }
        return new LoadResult(
                new PartitionData(ids, vectors),
                diskNanos,
                System.nanoTime() - decodeStart
        );
    }

    private ChunkResult rerankChunk(
            PartitionData batch,
            float[] query,
            int k,
            SearchParameters parameters,
            int dimensions
    ) {
        long loadStart = System.nanoTime();
        float[][] rows = new float[batch.ids().length][dimensions];
        for (int row = 0; row < rows.length; row++) {
            System.arraycopy(batch.vectors(), row * dimensions, rows[row], 0, dimensions);
        }
        long[] ids = batch.ids().clone();
        long candidateLoadNanos = System.nanoTime() - loadStart;
        VectorIndex reranker = Objects.requireNonNull(
                rerankerFactory.get(), "rerankerFactory returned null");
        try (reranker) {
            long transferStart = System.nanoTime();
            reranker.build(rows, ids);
            long transferNanos = System.nanoTime() - transferStart;
            long searchStart = System.nanoTime();
            List<SearchResult> results =
                    reranker.search(query, Math.min(k, rows.length), parameters);
            return new ChunkResult(
                    results,
                    candidateLoadNanos,
                    transferNanos,
                    System.nanoTime() - searchStart
            );
        }
    }

    private static void writeGeneration(
            Path root,
            float[][] vectors,
            long[] ids,
            DiskIvfBuildConfig config
    ) throws IOException {
        ValidatedInput input = validateBuild(vectors, ids, config);
        Files.createDirectories(root.resolve(GENERATIONS));
        UUID id = UUID.randomUUID();
        String name = id.toString();
        Path staging = root.resolve(GENERATIONS).resolve("." + name + ".tmp");
        Path published = root.resolve(GENERATIONS).resolve(name);
        Files.createDirectories(staging.resolve("partitions"));

        float[][] centroids = trainCentroids(input.vectors(), config);
        int[] assignments = assign(input.vectors(), centroids);
        List<List<Integer>> postings = new ArrayList<>(centroids.length);
        for (int i = 0; i < centroids.length; i++) postings.add(new ArrayList<>());
        for (int i = 0; i < assignments.length; i++) postings.get(assignments[i]).add(i);
        for (List<Integer> posting : postings) {
            posting.sort(Comparator.comparingLong(index -> input.ids()[index]));
        }

        List<PartitionEntry> entries = new ArrayList<>();
        long onDiskBytes = 0;
        try {
            Path centroidPath = staging.resolve("centroids.vfc");
            byte[] centroidBytes = encodeCentroids(id, centroids);
            writeForced(centroidPath, centroidBytes);
            onDiskBytes = Math.addExact(onDiskBytes, centroidBytes.length);

            for (int partitionId = 0; partitionId < postings.size(); partitionId++) {
                Path path = staging.resolve(partitionFile(partitionId));
                PartitionEntry entry = writePartition(
                        path, id, partitionId, input.dimensions(),
                        postings.get(partitionId), input);
                entries.add(entry);
                onDiskBytes = Math.addExact(onDiskBytes, entry.fileBytes());
            }

            byte[] manifestBytes = encodeManifest(
                    id, input, config, entries, centroidBytes.length,
                    crcOf(centroidBytes, 0, centroidBytes.length - Integer.BYTES),
                    onDiskBytes
            );
            writeForced(staging.resolve("manifest.vfi"), manifestBytes);
            writeForced(staging.resolve(READY), "ready\n".getBytes(StandardCharsets.US_ASCII));
            moveAtomically(staging, published);

            Path currentTmp = root.resolve(CURRENT + "." + name + ".tmp");
            writeForced(currentTmp, (name + "\n").getBytes(StandardCharsets.US_ASCII));
            moveAtomically(currentTmp, root.resolve(CURRENT));
        } catch (IOException | RuntimeException error) {
            throw error;
        }
    }

    private static Generation readGeneration(Path root) throws IOException {
        Path current = root.resolve(CURRENT);
        if (!Files.isRegularFile(current)) {
            throw new DiskIvfFormatException(current, "missing CURRENT generation pointer");
        }
        String name = Files.readString(current, StandardCharsets.US_ASCII).trim();
        UUID id;
        try {
            id = UUID.fromString(name);
        } catch (IllegalArgumentException error) {
            throw new DiskIvfFormatException(current, "invalid generation UUID");
        }
        Path directory = root.resolve(GENERATIONS).resolve(name);
        if (!Files.isRegularFile(directory.resolve(READY))) {
            throw new DiskIvfFormatException(directory.resolve(READY), "generation is not ready");
        }
        Manifest manifest = decodeManifest(directory.resolve("manifest.vfi"), id);
        float[][] centroids = decodeCentroids(directory.resolve("centroids.vfc"), id, manifest);
        for (PartitionEntry entry : manifest.partitions()) {
            validatePartitionFile(directory.resolve(partitionFile(entry.id())), id, manifest, entry);
        }
        return new Generation(id, directory, manifest, centroids);
    }

    private static ValidatedInput validateBuild(
            float[][] vectors, long[] ids, DiskIvfBuildConfig config) {
        Objects.requireNonNull(vectors, "vectors must not be null");
        Objects.requireNonNull(ids, "ids must not be null");
        if (vectors.length == 0 || vectors.length != ids.length) {
            throw new IllegalArgumentException("vectors and ids must be non-empty and aligned");
        }
        if (config.centroidCount() > vectors.length) {
            throw new IllegalArgumentException("centroidCount must be <= vector count");
        }
        int dimensions = -1;
        float[][] copy = new float[vectors.length][];
        Set<Long> seen = new HashSet<>();
        long[] copiedIds = ids.clone();
        for (int row = 0; row < vectors.length; row++) {
            if (!seen.add(ids[row])) {
                throw new IllegalArgumentException("duplicate id: " + ids[row]);
            }
            float[] vector = Objects.requireNonNull(vectors[row], "vector must not be null");
            if (dimensions < 0) dimensions = vector.length;
            if (dimensions <= 0 || vector.length != dimensions) {
                throw new IllegalArgumentException("vectors must have one positive dimension");
            }
            copy[row] = vector.clone();
            for (float value : copy[row]) {
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("vectors must contain finite values");
                }
            }
        }
        return new ValidatedInput(copy, copiedIds, dimensions);
    }

    private static float[][] trainCentroids(float[][] vectors, DiskIvfBuildConfig config) {
        Random random = new Random(config.seed());
        int dimensions = vectors[0].length;
        float[][] centroids = new float[config.centroidCount()][dimensions];
        List<Integer> choices = new ArrayList<>();
        for (int i = 0; i < vectors.length; i++) choices.add(i);
        java.util.Collections.shuffle(choices, random);
        for (int i = 0; i < centroids.length; i++) {
            centroids[i] = vectors[choices.get(i)].clone();
        }
        for (int iteration = 0; iteration < config.trainingIterations(); iteration++) {
            int[] assignments = assign(vectors, centroids);
            double[][] sums = new double[centroids.length][dimensions];
            int[] counts = new int[centroids.length];
            for (int row = 0; row < vectors.length; row++) {
                int cluster = assignments[row];
                counts[cluster]++;
                for (int column = 0; column < dimensions; column++) {
                    sums[cluster][column] += vectors[row][column];
                }
            }
            for (int cluster = 0; cluster < centroids.length; cluster++) {
                if (counts[cluster] == 0) continue;
                for (int column = 0; column < dimensions; column++) {
                    centroids[cluster][column] = (float) (sums[cluster][column] / counts[cluster]);
                }
            }
        }
        return centroids;
    }

    private static int[] assign(float[][] vectors, float[][] centroids) {
        int[] assignments = new int[vectors.length];
        for (int row = 0; row < vectors.length; row++) {
            assignments[row] = closest(vectors[row], centroids);
        }
        return assignments;
    }

    private static int closest(float[] vector, float[][] centroids) {
        int best = 0;
        float bestDistance = squaredDistance(vector, centroids[0]);
        for (int i = 1; i < centroids.length; i++) {
            float distance = squaredDistance(vector, centroids[i]);
            if (distance < bestDistance) {
                best = i;
                bestDistance = distance;
            }
        }
        return best;
    }

    private static int[] selectCentroids(
            float[] query, float[][] centroids, Manifest manifest, int probes) {
        Integer[] order = new Integer[centroids.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator
                .comparingDouble((Integer id) -> squaredDistance(query, centroids[id]))
                .thenComparingInt(Integer::intValue));
        int[] selected = new int[probes];
        for (int i = 0; i < probes; i++) selected[i] = order[i];
        return selected;
    }

    private static float squaredDistance(float[] left, float[] right) {
        float total = 0;
        for (int i = 0; i < left.length; i++) {
            float delta = left[i] - right[i];
            total += delta * delta;
        }
        return total;
    }

    private static void validateQuery(
            float[] query, int k, SearchParameters parameters, Manifest manifest) {
        Objects.requireNonNull(query, "query must not be null");
        if (query.length != manifest.dimensions()) {
            throw new IllegalArgumentException("query dimension mismatch");
        }
        for (float value : query) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException("query must be finite");
        }
        if (k <= 0 || k > manifest.vectorCount()) {
            throw new IllegalArgumentException("k must be positive and <= vector count");
        }
        if (parameters.metric() != manifest.metric()) {
            throw new IllegalArgumentException("search metric must match index build metric");
        }
    }

    private static byte[] encodeCentroids(UUID id, float[][] centroids) {
        int dimensions = centroids[0].length;
        int size = Math.toIntExact(Math.addExact(
                CENTROID_HEADER_BYTES + Integer.BYTES,
                Math.multiplyExact(
                        Math.multiplyExact((long) centroids.length, dimensions), Float.BYTES)));
        ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(CENTROIDS_MAGIC).putInt(FORMAT_VERSION);
        putUuid(buffer, id);
        buffer.putInt(dimensions).putInt(centroids.length).putInt(1);
        for (float[] centroid : centroids) for (float value : centroid) buffer.putFloat(value);
        buffer.putInt(crcOf(buffer.array(), 0, size - Integer.BYTES));
        return buffer.array();
    }

    private static PartitionEntry writePartition(
            Path path,
            UUID id,
            int partitionId,
            int dimensions,
            List<Integer> rows,
            ValidatedInput input
    ) throws IOException {
        int recordBytes = Math.toIntExact(Math.addExact(
                Long.BYTES, Math.multiplyExact((long) dimensions, Float.BYTES)));
        long size = Math.addExact(
                PARTITION_HEADER_BYTES + Integer.BYTES,
                Math.multiplyExact((long) rows.size(), recordBytes));
        CRC32C crc = new CRC32C();
        ByteBuffer header = ByteBuffer.allocate(PARTITION_HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        header.put(PARTITION_MAGIC).putInt(FORMAT_VERSION);
        putUuid(header, id);
        header.putInt(partitionId).putInt(dimensions).putLong(rows.size());
        crc.update(header.array(), 0, header.capacity());
        header.flip();
        Files.createDirectories(path.getParent());
        try (FileChannel channel = FileChannel.open(
                path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            writeFully(channel, header);
            ByteBuffer record = ByteBuffer.allocate(recordBytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int row : rows) {
                record.clear();
                record.putLong(input.ids()[row]);
                for (float value : input.vectors()[row]) record.putFloat(value);
                crc.update(record.array(), 0, record.capacity());
                record.flip();
                writeFully(channel, record);
            }
            ByteBuffer checksum = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            checksum.putInt((int) crc.getValue()).flip();
            writeFully(channel, checksum);
            channel.force(true);
        }
        return new PartitionEntry(partitionId, rows.size(), size, (int) crc.getValue());
    }

    private static byte[] encodeManifest(
            UUID id,
            ValidatedInput input,
            DiskIvfBuildConfig config,
            List<PartitionEntry> entries,
            long centroidFileBytes,
            int centroidCrc,
            long onDiskBytes
    ) {
        int size = Math.addExact(
                Math.addExact(MANIFEST_FIXED_BYTES,
                        Math.multiplyExact(entries.size(), MANIFEST_ENTRY_BYTES)),
                Integer.BYTES);
        ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put(MANIFEST_MAGIC).putInt(FORMAT_VERSION);
        putUuid(buffer, id);
        buffer.putInt(input.dimensions()).putInt(config.centroidCount()).putLong(input.vectors().length);
        buffer.putInt(1).putLong(config.seed()).putInt(config.trainingIterations());
        buffer.putInt(entries.size()).putLong(centroidFileBytes).putInt(centroidCrc);
        for (PartitionEntry entry : entries) {
            buffer.putInt(entry.id()).putLong(entry.vectorCount()).putLong(entry.fileBytes())
                    .putInt(entry.crc()).putInt(0);
        }
        buffer.putInt(crcOf(buffer.array(), 0, size - Integer.BYTES));
        return buffer.array();
    }

    private static Manifest decodeManifest(Path path, UUID expectedId) throws IOException {
        byte[] bytes = readCheckedFile(path, MANIFEST_FIXED_BYTES + Integer.BYTES);
        validateTrailingCrc(path, bytes);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        expectMagic(path, buffer, MANIFEST_MAGIC);
        expectVersion(path, buffer);
        expectUuid(path, buffer, expectedId);
        int dimensions = positive(path, "dimensions", buffer.getInt());
        int centroidCount = positive(path, "centroidCount", buffer.getInt());
        long vectorCount = positive(path, "vectorCount", buffer.getLong());
        int metricCode = buffer.getInt();
        if (metricCode != 1) throw new DiskIvfFormatException(path, "unsupported metric code");
        long seed = buffer.getLong();
        int iterations = positive(path, "trainingIterations", buffer.getInt());
        int partitionCount = positive(path, "partitionCount", buffer.getInt());
        long centroidBytes = positive(path, "centroidFileBytes", buffer.getLong());
        int centroidCrc = buffer.getInt();
        long expectedLength;
        try {
            expectedLength = Math.addExact(
                    Math.addExact(MANIFEST_FIXED_BYTES,
                            Math.multiplyExact((long) partitionCount, MANIFEST_ENTRY_BYTES)),
                    Integer.BYTES);
        } catch (ArithmeticException error) {
            throw new DiskIvfFormatException(path, "manifest directory size overflow");
        }
        if (bytes.length != expectedLength) {
            throw new DiskIvfFormatException(path, "manifest length mismatch");
        }
        List<PartitionEntry> entries = new ArrayList<>();
        long onDiskBytes;
        try {
            onDiskBytes = Math.addExact(centroidBytes, bytes.length);
        } catch (ArithmeticException error) {
            throw new DiskIvfFormatException(path, "manifest file totals overflow");
        }
        long partitionVectors = 0;
        for (int i = 0; i < partitionCount; i++) {
            int id = buffer.getInt();
            long count = buffer.getLong();
            long fileBytes = buffer.getLong();
            int crc = buffer.getInt();
            buffer.getInt();
            if (id != i || count < 0 || fileBytes < PARTITION_HEADER_BYTES + Integer.BYTES) {
                throw new DiskIvfFormatException(path, "invalid partition directory entry");
            }
            entries.add(new PartitionEntry(id, count, fileBytes, crc));
            try {
                onDiskBytes = Math.addExact(onDiskBytes, fileBytes);
                partitionVectors = Math.addExact(partitionVectors, count);
            } catch (ArithmeticException error) {
                throw new DiskIvfFormatException(path, "partition totals overflow");
            }
        }
        if (centroidCount != partitionCount) {
            throw new DiskIvfFormatException(path, "centroid and partition counts differ");
        }
        if (partitionVectors != vectorCount) {
            throw new DiskIvfFormatException(path, "partition vector total differs from manifest");
        }
        return new Manifest(
                dimensions, centroidCount, vectorCount, DistanceMetric.EUCLIDEAN,
                seed, iterations, partitionCount, centroidBytes, centroidCrc,
                List.copyOf(entries), onDiskBytes);
    }

    private static float[][] decodeCentroids(Path path, UUID id, Manifest manifest) throws IOException {
        if (!Files.isRegularFile(path)) throw new DiskIvfFormatException(path, "missing file");
        long actualLength = Files.size(path);
        long expectedLength;
        try {
            expectedLength = Math.addExact(
                    CENTROID_HEADER_BYTES + Integer.BYTES,
                    Math.multiplyExact(
                            Math.multiplyExact(
                                    (long) manifest.centroidCount(), manifest.dimensions()),
                            Float.BYTES));
        } catch (ArithmeticException error) {
            throw new DiskIvfFormatException(path, "centroid payload size overflow");
        }
        if (actualLength != expectedLength || actualLength != manifest.centroidFileBytes()) {
            throw new DiskIvfFormatException(path, "centroid file length mismatch");
        }
        if (actualLength > Integer.MAX_VALUE) {
            throw new DiskIvfFormatException(path, "centroid file exceeds prototype allocation limit");
        }
        byte[] bytes = readCheckedFile(path, CENTROID_HEADER_BYTES + Integer.BYTES);
        validateTrailingCrc(path, bytes);
        if (crcOf(bytes, 0, bytes.length - Integer.BYTES) != manifest.centroidCrc()) {
            throw new DiskIvfFormatException(path, "centroid manifest checksum mismatch");
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        expectMagic(path, buffer, CENTROIDS_MAGIC);
        expectVersion(path, buffer);
        expectUuid(path, buffer, id);
        if (buffer.getInt() != manifest.dimensions()
                || buffer.getInt() != manifest.centroidCount()
                || buffer.getInt() != 1) {
            throw new DiskIvfFormatException(path, "centroid header mismatch");
        }
        float[][] centroids = new float[manifest.centroidCount()][manifest.dimensions()];
        for (float[] centroid : centroids) {
            for (int i = 0; i < centroid.length; i++) {
                centroid[i] = buffer.getFloat();
                if (!Float.isFinite(centroid[i])) {
                    throw new DiskIvfFormatException(path, "non-finite centroid");
                }
            }
        }
        return centroids;
    }

    private static void validatePartitionFile(
            Path path, UUID id, Manifest manifest, PartitionEntry entry) throws IOException {
        if (!Files.isRegularFile(path)) throw new DiskIvfFormatException(path, "missing file");
        long actualLength = Files.size(path);
        if (actualLength != entry.fileBytes()) {
            throw new DiskIvfFormatException(path, "partition file length mismatch");
        }
        long expectedBytes;
        try {
            long recordBytes = Math.addExact(
                    Long.BYTES, Math.multiplyExact((long) manifest.dimensions(), Float.BYTES));
            expectedBytes = Math.addExact(
                    PARTITION_HEADER_BYTES + Integer.BYTES,
                    Math.multiplyExact(entry.vectorCount(), recordBytes));
        } catch (ArithmeticException error) {
            throw new DiskIvfFormatException(path, "partition payload size overflow");
        }
        if (expectedBytes != actualLength) {
            throw new DiskIvfFormatException(path, "partition payload length mismatch");
        }
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(PARTITION_HEADER_BYTES)
                    .order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, header, 0);
            header.flip();
            expectMagic(path, header, PARTITION_MAGIC);
            expectVersion(path, header);
            expectUuid(path, header, id);
            if (header.getInt() != entry.id()
                    || header.getInt() != manifest.dimensions()
                    || header.getLong() != entry.vectorCount()) {
                throw new DiskIvfFormatException(path, "partition header mismatch");
            }
            validateChannelCrc(path, channel, actualLength, entry.crc());
        }
    }

    private static void validateChannelCrc(
            Path path, FileChannel channel, long fileBytes, int manifestCrc) throws IOException {
        CRC32C crc = new CRC32C();
        ByteBuffer chunk = ByteBuffer.allocate(64 * 1024);
        long position = 0;
        long remaining = fileBytes - Integer.BYTES;
        while (remaining > 0) {
            chunk.clear();
            chunk.limit((int) Math.min(chunk.capacity(), remaining));
            readFully(channel, chunk, position);
            chunk.flip();
            crc.update(chunk);
            position += chunk.limit();
            remaining -= chunk.limit();
        }
        ByteBuffer storedBuffer = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, storedBuffer, fileBytes - Integer.BYTES);
        storedBuffer.flip();
        int stored = storedBuffer.getInt();
        int actual = (int) crc.getValue();
        if (stored != actual || manifestCrc != actual) {
            throw new DiskIvfFormatException(path, "CRC32C mismatch");
        }
    }

    private static byte[] readCheckedFile(Path path, int minimumBytes) throws IOException {
        if (!Files.isRegularFile(path)) throw new DiskIvfFormatException(path, "missing file");
        byte[] bytes = Files.readAllBytes(path);
        if (bytes.length < minimumBytes) throw new DiskIvfFormatException(path, "truncated file");
        return bytes;
    }

    private static void validateTrailingCrc(Path path, byte[] bytes) throws DiskIvfFormatException {
        int stored = ByteBuffer.wrap(bytes, bytes.length - Integer.BYTES, Integer.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN).getInt();
        int actual = crcOf(bytes, 0, bytes.length - Integer.BYTES);
        if (stored != actual) throw new DiskIvfFormatException(path, "CRC32C mismatch");
    }

    private static void expectMagic(Path path, ByteBuffer buffer, byte[] expected)
            throws DiskIvfFormatException {
        byte[] actual = new byte[expected.length];
        buffer.get(actual);
        if (!Arrays.equals(actual, expected)) throw new DiskIvfFormatException(path, "bad magic");
    }

    private static void expectVersion(Path path, ByteBuffer buffer) throws DiskIvfFormatException {
        if (buffer.getInt() != FORMAT_VERSION) {
            throw new DiskIvfFormatException(path, "unsupported format version");
        }
    }

    private static void expectUuid(Path path, ByteBuffer buffer, UUID expected)
            throws DiskIvfFormatException {
        UUID actual = new UUID(buffer.getLong(), buffer.getLong());
        if (!actual.equals(expected)) throw new DiskIvfFormatException(path, "generation UUID mismatch");
    }

    private static void putUuid(ByteBuffer buffer, UUID id) {
        buffer.putLong(id.getMostSignificantBits()).putLong(id.getLeastSignificantBits());
    }

    private static int positive(Path path, String name, int value) throws DiskIvfFormatException {
        if (value <= 0) throw new DiskIvfFormatException(path, name + " must be positive");
        return value;
    }

    private static long positive(Path path, String name, long value) throws DiskIvfFormatException {
        if (value <= 0) throw new DiskIvfFormatException(path, name + " must be positive");
        return value;
    }

    private static int crcOf(byte[] bytes, int offset, int length) {
        CRC32C crc = new CRC32C();
        crc.update(bytes, offset, length);
        return (int) crc.getValue();
    }

    private static void writeForced(Path path, byte[] bytes) throws IOException {
        Files.createDirectories(path.getParent());
        try (FileChannel channel = FileChannel.open(
                path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            ByteBuffer buffer = ByteBuffer.wrap(bytes);
            while (buffer.hasRemaining()) channel.write(buffer);
            channel.force(true);
        }
    }

    private static void moveAtomically(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException error) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static void readFully(FileChannel channel, ByteBuffer buffer, long position) throws IOException {
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer, position);
            if (read < 0) throw new IOException("unexpected end of partition");
            position += read;
        }
    }

    private static void writeFully(FileChannel channel, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) {
            channel.write(buffer);
        }
    }

    private static String partitionFile(int partitionId) {
        return "partitions/part-" + String.format("%05d", partitionId) + ".vfp";
    }

    private static long cacheWeight(long vectorCount, int dimensions) {
        long primitivePayload = Math.multiplyExact(vectorCount,
                Math.addExact(Long.BYTES, Math.multiplyExact((long) dimensions, Float.BYTES)));
        return Math.addExact(primitivePayload, 64);
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("index is closed");
    }

    private record ValidatedInput(float[][] vectors, long[] ids, int dimensions) {
    }

    private record PartitionEntry(int id, long vectorCount, long fileBytes, int crc) {
    }

    private record Manifest(
            int dimensions,
            int centroidCount,
            long vectorCount,
            DistanceMetric metric,
            long seed,
            int trainingIterations,
            int partitionCount,
            long centroidFileBytes,
            int centroidCrc,
            List<PartitionEntry> partitions,
            long onDiskBytes
    ) {
    }

    private record Generation(
            UUID id, Path directory, Manifest manifest, float[][] centroids) {
    }

    private record PartitionData(long[] ids, float[] vectors) {
        private PartitionData slice(int offset, int count) {
            if (offset == 0 && count == ids.length) return this;
            int dimensions = vectors.length / ids.length;
            return new PartitionData(
                    Arrays.copyOfRange(ids, offset, offset + count),
                    Arrays.copyOfRange(
                            vectors,
                            Math.multiplyExact(offset, dimensions),
                            Math.multiplyExact(offset + count, dimensions))
            );
        }
    }

    private record LoadResult(
            PartitionData data, long diskReadNanos, long candidateLoadNanos) {
    }

    private record ChunkResult(
            List<SearchResult> results,
            long candidateLoadNanos,
            long backendTransferNanos,
            long backendSearchNanos
    ) {
    }

    private static final class PartitionCache {
        private final long budget;
        private final LinkedHashMap<Integer, CacheEntry> entries =
                new LinkedHashMap<>(16, 0.75f, true);
        private long residentBytes;
        private long hits;
        private long misses;
        private long evictions;

        private PartitionCache(long budget) {
            this.budget = budget;
        }

        private PartitionData get(int id) {
            CacheEntry entry = entries.get(id);
            if (entry == null) {
                misses++;
                return null;
            }
            hits++;
            return entry.data();
        }

        private void put(int id, PartitionData data, long bytes) {
            CacheEntry replaced = entries.remove(id);
            if (replaced != null) {
                residentBytes -= replaced.bytes();
            }
            while (!entries.isEmpty() && residentBytes + bytes > budget) {
                Map.Entry<Integer, CacheEntry> eldest = entries.entrySet().iterator().next();
                entries.remove(eldest.getKey());
                residentBytes -= eldest.getValue().bytes();
                evictions++;
            }
            if (bytes <= budget) {
                entries.put(id, new CacheEntry(data, bytes));
                residentBytes += bytes;
            }
        }

        private void clear() {
            entries.clear();
            residentBytes = 0;
            hits = 0;
            misses = 0;
            evictions = 0;
        }
    }

    private record CacheEntry(PartitionData data, long bytes) {
    }
}

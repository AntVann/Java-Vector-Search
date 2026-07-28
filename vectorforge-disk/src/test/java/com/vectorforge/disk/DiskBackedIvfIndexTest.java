package com.vectorforge.disk;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.IndexMetrics;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;
import com.vectorforge.cpu.CpuBruteForceIndex;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.zip.CRC32C;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DiskBackedIvfIndexTest {

    private static final SearchParameters EUCLIDEAN =
            new SearchParameters(DistanceMetric.EUCLIDEAN);

    @TempDir
    Path temporaryDirectory;

    @Test
    void fullProbeMatchesExactCpuOrderingAndScores() throws Exception {
        Dataset data = randomDataset(17, 256, 16);
        try (CpuBruteForceIndex exact = new CpuBruteForceIndex();
             DiskBackedIvfIndex disk = create(
                     temporaryDirectory, data, 8, 8, 4_000_000, 4_000_000)) {
            exact.build(data.vectors(), data.ids());
            List<SearchResult> expected = exact.search(data.vectors()[37], 10, EUCLIDEAN);
            List<SearchResult> actual = disk.search(data.vectors()[37], 10, EUCLIDEAN);
            assertEquals(expected.size(), actual.size());
            for (int i = 0; i < expected.size(); i++) {
                assertEquals(expected.get(i).id(), actual.get(i).id());
                assertEquals(expected.get(i).score(), actual.get(i).score(), 1.0e-6);
            }
        }
    }

    @Test
    void reopenPreservesGenerationAndResults() throws Exception {
        Dataset data = randomDataset(19, 80, 4);
        List<SearchResult> before;
        String generation;
        try (DiskBackedIvfIndex index = create(
                temporaryDirectory, data, 4, 4, 1_000_000, 1_000_000)) {
            before = index.search(data.vectors()[3], 5, EUCLIDEAN);
            generation = index.diskMetrics().generation();
        }
        try (DiskBackedIvfIndex reopened = open(temporaryDirectory, 4)) {
            assertEquals(generation, reopened.diskMetrics().generation());
            assertEquals(before, reopened.search(data.vectors()[3], 5, EUCLIDEAN));
            assertEquals(data.vectors().length, reopened.metrics().vectorCount());
        }
    }

    @Test
    void repeatedPartitionAccessUsesByteBoundedCache() throws Exception {
        Dataset data = separatedDataset();
        try (DiskBackedIvfIndex index =
                     create(temporaryDirectory, data, 2, 1, 1_000_000, 1_000_000)) {
            DiskIvfSearchResponse first = index.searchDetailed(new float[]{0, 0}, 2, EUCLIDEAN);
            DiskIvfSearchResponse second = index.searchDetailed(new float[]{0, 0}, 2, EUCLIDEAN);
            assertTrue(first.cacheMisses() >= 1);
            assertTrue(second.cacheHits() >= 1);
            assertEquals(0, second.timings().diskReadNanos());
            assertTrue(index.diskMetrics().cacheResidentBytes() <= 1_000_000);
        }
    }

    @Test
    void candidateBudgetBoundsRerankerInput() throws Exception {
        Dataset data = randomDataset(23, 100, 8);
        long recordBytes = Long.BYTES + 8L * Float.BYTES;
        RecordingIndex.maxBuildSize = 0;
        try (DiskBackedIvfIndex index = DiskBackedIvfIndex.create(
                temporaryDirectory, data.vectors(), data.ids(),
                new DiskIvfBuildConfig(4, 6, 29, DistanceMetric.EUCLIDEAN),
                new DiskIvfSearchConfig(4, 0, recordBytes * 7),
                RecordingIndex::new)) {
            DiskIvfSearchResponse response =
                    index.searchDetailed(data.vectors()[0], 10, EUCLIDEAN);
            assertEquals(100, response.candidatesLoaded());
            assertTrue(RecordingIndex.maxBuildSize <= 3);
            assertEquals(10, response.results().size());
            try (CpuBruteForceIndex exact = new CpuBruteForceIndex()) {
                exact.build(data.vectors(), data.ids());
                assertEquals(exact.search(data.vectors()[0], 10, EUCLIDEAN), response.results());
            }
        }
    }

    @Test
    void emptyPartitionsAreValidAfterReopen() throws Exception {
        float[][] vectors = new float[12][2];
        long[] ids = new long[12];
        for (int i = 0; i < vectors.length; i++) {
            vectors[i] = new float[]{1, 1};
            ids[i] = i + 1;
        }
        Dataset data = new Dataset(vectors, ids);
        try (DiskBackedIvfIndex ignored =
                     create(temporaryDirectory, data, 4, 4, 1_000_000, 1_000_000)) {
            assertTrue(Files.size(generationDirectory().resolve("partitions/part-00003.vfp")) > 0);
        }
        try (DiskBackedIvfIndex reopened = open(temporaryDirectory, 4)) {
            assertEquals(5, reopened.search(new float[]{1, 1}, 5, EUCLIDEAN).size());
        }
    }

    @Test
    void rejectsCorruptMetadata() throws Exception {
        buildSmall();
        Path manifest = generationDirectory().resolve("manifest.vfi");
        byte[] bytes = Files.readAllBytes(manifest);
        bytes[0] ^= 0x40;
        Files.write(manifest, bytes, StandardOpenOption.TRUNCATE_EXISTING);
        assertThrows(DiskIvfFormatException.class, () -> open(temporaryDirectory, 2));
    }

    @Test
    void rejectsMissingPartitionFile() throws Exception {
        buildSmall();
        Files.delete(generationDirectory().resolve("partitions/part-00000.vfp"));
        DiskIvfFormatException error = assertThrows(
                DiskIvfFormatException.class, () -> open(temporaryDirectory, 2));
        assertTrue(error.getMessage().contains("missing file"));
    }

    @Test
    void rejectsPartialPartitionWrite() throws Exception {
        buildSmall();
        Path partition = generationDirectory().resolve("partitions/part-00000.vfp");
        byte[] bytes = Files.readAllBytes(partition);
        Files.write(partition, java.util.Arrays.copyOf(bytes, bytes.length - 3),
                StandardOpenOption.TRUNCATE_EXISTING);
        assertThrows(DiskIvfFormatException.class, () -> open(temporaryDirectory, 2));
    }

    @Test
    void incompleteUnpublishedGenerationIsIgnored() throws Exception {
        buildSmall();
        String current = Files.readString(temporaryDirectory.resolve("CURRENT")).trim();
        Path orphan = temporaryDirectory.resolve("generations/.partial.tmp");
        Files.createDirectories(orphan);
        Files.writeString(orphan.resolve("manifest.vfi"), "partial");
        try (DiskBackedIvfIndex reopened = open(temporaryDirectory, 2)) {
            assertEquals(current, reopened.diskMetrics().generation());
            assertFalse(reopened.search(new float[]{0, 0}, 1, EUCLIDEAN).isEmpty());
        }
    }

    @Test
    void stalePointerTempDoesNotBlockRebuild() throws Exception {
        buildSmall();
        Files.writeString(temporaryDirectory.resolve("CURRENT.tmp"), "stale");
        Dataset replacement = randomDataset(31, 20, 2);
        try (DiskBackedIvfIndex index = open(temporaryDirectory, 2)) {
            index.build(replacement.vectors(), replacement.ids());
            assertEquals(20, index.metrics().vectorCount());
        }
    }

    @Test
    void rejectsChecksummedManifestWithMismatchedCentroidAndPartitionCounts() throws Exception {
        buildSmall();
        Path manifest = generationDirectory().resolve("manifest.vfi");
        byte[] bytes = Files.readAllBytes(manifest);
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(32, 3);
        CRC32C crc = new CRC32C();
        crc.update(bytes, 0, bytes.length - Integer.BYTES);
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                .putInt(bytes.length - Integer.BYTES, (int) crc.getValue());
        Files.write(manifest, bytes, StandardOpenOption.TRUNCATE_EXISTING);
        assertThrows(DiskIvfFormatException.class, () -> open(temporaryDirectory, 2));
    }

    @Test
    void approximateRecallIsDeterministicAndFullProbeIsExact() throws Exception {
        Dataset data = clusteredDataset(29, 800, 12);
        List<float[]> queries = List.of(
                data.vectors()[5], data.vectors()[105], data.vectors()[405], data.vectors()[705]);
        double probeOne = averageRecall(data, queries, 1, temporaryDirectory.resolve("one"));
        double probeFour = averageRecall(data, queries, 4, temporaryDirectory.resolve("four"));
        double probeEight = averageRecall(data, queries, 8, temporaryDirectory.resolve("eight"));
        assertTrue(probeOne >= 0 && probeOne <= 1);
        assertTrue(probeFour >= probeOne);
        assertEquals(1.0, probeEight, 0.0);
        assertEquals(probeOne,
                averageRecall(data, queries, 1, temporaryDirectory.resolve("one-repeat")), 0.0);
    }

    private double averageRecall(Dataset data, List<float[]> queries, int probes, Path path)
            throws Exception {
        try (CpuBruteForceIndex exact = new CpuBruteForceIndex();
             DiskBackedIvfIndex disk = create(path, data, 8, probes, 2_000_000, 2_000_000)) {
            exact.build(data.vectors(), data.ids());
            double total = 0;
            for (float[] query : queries) {
                Set<Long> truth = resultIds(exact.search(query, 10, EUCLIDEAN));
                Set<Long> found = resultIds(disk.search(query, 10, EUCLIDEAN));
                found.retainAll(truth);
                total += found.size() / 10.0;
            }
            return total / queries.size();
        }
    }

    private static Set<Long> resultIds(List<SearchResult> results) {
        Set<Long> ids = new HashSet<>();
        results.forEach(result -> ids.add(result.id()));
        return ids;
    }

    private void buildSmall() throws Exception {
        try (DiskBackedIvfIndex ignored =
                     create(temporaryDirectory, separatedDataset(), 2, 2, 1_000_000, 1_000_000)) {
            // setup
        }
    }

    private static DiskBackedIvfIndex create(
            Path path, Dataset data, int centroids, int probes, long cache, long candidates)
            throws IOException {
        return DiskBackedIvfIndex.create(
                path, data.vectors(), data.ids(),
                new DiskIvfBuildConfig(centroids, 6, 29, DistanceMetric.EUCLIDEAN),
                new DiskIvfSearchConfig(probes, cache, candidates),
                CpuBruteForceIndex::new
        );
    }

    private static DiskBackedIvfIndex open(Path path, int probes) throws IOException {
        return DiskBackedIvfIndex.open(
                path, new DiskIvfSearchConfig(probes, 1_000_000, 1_000_000),
                CpuBruteForceIndex::new);
    }

    private Path generationDirectory() throws IOException {
        String generation = Files.readString(temporaryDirectory.resolve("CURRENT")).trim();
        return temporaryDirectory.resolve("generations").resolve(generation);
    }

    private static Dataset separatedDataset() {
        return new Dataset(
                new float[][]{{0, 0}, {0.1f, 0}, {0, 0.1f}, {10, 10}, {10.1f, 10}, {10, 10.1f}},
                new long[]{1, 2, 3, 4, 5, 6});
    }

    private static Dataset randomDataset(long seed, int count, int dimensions) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimensions];
        long[] ids = new long[count];
        for (int row = 0; row < count; row++) {
            ids[row] = row + 1;
            for (int column = 0; column < dimensions; column++) {
                vectors[row][column] = random.nextFloat() * 2 - 1;
            }
        }
        return new Dataset(vectors, ids);
    }

    private static Dataset clusteredDataset(long seed, int count, int dimensions) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimensions];
        long[] ids = new long[count];
        for (int row = 0; row < count; row++) {
            int cluster = row % 8;
            ids[row] = row + 1;
            for (int column = 0; column < dimensions; column++) {
                vectors[row][column] = cluster * 5.0f + (float) random.nextGaussian() * 0.2f;
            }
        }
        return new Dataset(vectors, ids);
    }

    private record Dataset(float[][] vectors, long[] ids) {
    }

    private static final class RecordingIndex implements VectorIndex {
        private static int maxBuildSize;
        private final CpuBruteForceIndex delegate = new CpuBruteForceIndex();

        @Override
        public void build(float[][] vectors, long[] ids) {
            maxBuildSize = Math.max(maxBuildSize, vectors.length);
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
            delegate.close();
        }
    }
}

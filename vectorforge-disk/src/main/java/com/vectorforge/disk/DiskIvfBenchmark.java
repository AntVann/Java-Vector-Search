package com.vectorforge.disk;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.cpu.CpuBruteForceIndex;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Small fixed-seed end-to-end research smoke. It emits JSON Lines and makes no scale claim.
 */
public final class DiskIvfBenchmark {

    private static final SearchParameters EUCLIDEAN =
            new SearchParameters(DistanceMetric.EUCLIDEAN);

    private DiskIvfBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Path root = args.length == 0
                ? Path.of("vectorforge-disk", "target", "disk-ivf-smoke-index")
                : Path.of(args[0]);
        int vectorCount = 10_000;
        int dimensions = 32;
        int queryCount = 20;
        int k = 10;
        long seed = 8675309L;
        Dataset data = dataset(seed, vectorCount, dimensions);
        float[][] queries = new float[queryCount][];
        for (int i = 0; i < queryCount; i++) queries[i] = data.vectors()[i * 97].clone();

        long buildStart = System.nanoTime();
        try (DiskBackedIvfIndex ignored = DiskBackedIvfIndex.create(
                root, data.vectors(), data.ids(),
                new DiskIvfBuildConfig(16, 8, seed, DistanceMetric.EUCLIDEAN),
                new DiskIvfSearchConfig(16, 2L * 1024 * 1024, 4L * 1024 * 1024),
                CpuBruteForceIndex::new)) {
            // Build and close before timed reopen/search.
        }
        double buildMillis = millis(System.nanoTime() - buildStart);

        List<List<SearchResult>> truth = new java.util.ArrayList<>(queryCount);
        try (CpuBruteForceIndex exact = new CpuBruteForceIndex()) {
            exact.build(data.vectors(), data.ids());
            for (int i = 0; i < queryCount; i++) {
                truth.add(exact.search(queries[i], k, EUCLIDEAN));
            }
        }

        System.out.printf(Locale.ROOT,
                "{\"record_type\":\"metadata\",\"seed\":%d,\"vectors\":%d,"
                        + "\"dimensions\":%d,\"queries\":%d,\"k\":%d,"
                        + "\"centroids\":16,\"build_ms\":%.6f,"
                        + "\"cache_mode\":\"warm-after-3-query-warmup\"}%n",
                seed, vectorCount, dimensions, queryCount, k, buildMillis);
        for (int probes : new int[]{1, 4, 16}) {
            run(root, queries, truth, k, probes);
        }
    }

    private static void run(
            Path root, float[][] queries, List<List<SearchResult>> truth, int k, int probes)
            throws Exception {
        try (DiskBackedIvfIndex index = DiskBackedIvfIndex.open(
                root,
                new DiskIvfSearchConfig(probes, 2L * 1024 * 1024, 4L * 1024 * 1024),
                CpuBruteForceIndex::new)) {
            for (int i = 0; i < 3; i++) {
                index.search(queries[i], k, EUCLIDEAN);
            }
            long centroid = 0;
            long disk = 0;
            long load = 0;
            long transfer = 0;
            long search = 0;
            long end = 0;
            long candidates = 0;
            long cacheHits = 0;
            long cacheMisses = 0;
            double recall = 0;
            for (int i = 0; i < queries.length; i++) {
                DiskIvfSearchResponse response = index.searchDetailed(queries[i], k, EUCLIDEAN);
                DiskIvfSearchTimings timing = response.timings();
                centroid += timing.centroidSearchNanos();
                disk += timing.diskReadNanos();
                load += timing.candidateLoadNanos();
                transfer += timing.backendTransferNanos();
                search += timing.backendSearchNanos();
                end += timing.endToEndNanos();
                candidates += response.candidatesLoaded();
                cacheHits += response.cacheHits();
                cacheMisses += response.cacheMisses();
                recall += recall(truth.get(i), response.results(), k);
            }
            double divisor = queries.length;
            System.out.printf(Locale.ROOT,
                    "{\"record_type\":\"result\",\"nprobe\":%d,\"recall_at_k\":%.6f,"
                            + "\"centroid_ms\":%.6f,\"disk_read_ms\":%.6f,"
                            + "\"candidate_load_ms\":%.6f,\"backend_transfer_ms\":%.6f,"
                            + "\"backend_search_ms\":%.6f,\"end_to_end_ms\":%.6f,"
                            + "\"avg_candidates\":%.2f,\"cache_hits\":%d,\"cache_misses\":%d}%n",
                    probes, recall / divisor,
                    millis(centroid) / divisor,
                    millis(disk) / divisor,
                    millis(load) / divisor,
                    millis(transfer) / divisor,
                    millis(search) / divisor,
                    millis(end) / divisor,
                    candidates / divisor,
                    cacheHits,
                    cacheMisses);
        }
    }

    private static double recall(List<SearchResult> expected, List<SearchResult> actual, int k) {
        Set<Long> ids = new HashSet<>();
        actual.forEach(result -> ids.add(result.id()));
        long matches = expected.stream().filter(result -> ids.contains(result.id())).count();
        return matches / (double) k;
    }

    private static Dataset dataset(long seed, int count, int dimensions) {
        Random random = new Random(seed);
        float[][] vectors = new float[count][dimensions];
        long[] ids = new long[count];
        for (int row = 0; row < count; row++) {
            int cluster = row % 16;
            ids[row] = row + 1;
            for (int column = 0; column < dimensions; column++) {
                vectors[row][column] =
                        cluster * 2.0f + (float) random.nextGaussian() * 0.35f;
            }
        }
        return new Dataset(vectors, ids);
    }

    private static double millis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private record Dataset(float[][] vectors, long[] ids) {
    }
}

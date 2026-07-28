package com.vectorforge.benchmarks;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.api.SearchResult;
import com.vectorforge.api.VectorIndex;
import com.vectorforge.cpu.CpuBruteForceIndex;
import com.vectorforge.gpu.CudaBruteForceIndex;
import com.vectorforge.gpu.CudaSearchTimings;
import com.vectorforge.gpu.CuvsVectorIndex;
import com.vectorforge.nativeindex.NativeBindings;
import com.vectorforge.nativeindex.NativeBruteForceIndex;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.UUID;

/**
 * End-to-end, machine-readable backend profiler. This is deliberately separate from JMH.
 */
public final class EndToEndBenchmarkRunner {

    private EndToEndBenchmarkRunner() {
    }

    public static void main(String[] args) throws IOException {
        Config config = Config.parse(args);
        boolean[] failed = {false};
        try (PrintWriter output = config.output() == null
                ? new PrintWriter(System.out, true, StandardCharsets.UTF_8)
                : new PrintWriter(Files.newBufferedWriter(config.output(), StandardCharsets.UTF_8))) {
            emitMetadata(output, config);
            for (int vectors : config.vectorCounts()) {
                for (int dimensions : config.dimensions()) {
                    long datasetBytes = Math.multiplyExact(Math.multiplyExact((long) vectors, dimensions), Float.BYTES);
                    for (int batch : config.batchSizes()) {
                        for (int k : config.kValues()) {
                            for (DistanceMetric metric : config.metrics()) {
                                Scenario scenario = new Scenario(vectors, dimensions, batch, k, metric, datasetBytes);
                                failed[0] |= runScenario(output, scenario, config);
                            }
                        }
                    }
                }
            }
            output.flush();
            if (output.checkError()) {
                throw new IOException("failed to write benchmark output");
            }
        }
        if (failed[0]) {
            throw new IllegalStateException("one or more detected backends failed; see error records");
        }
    }

    private static boolean runScenario(PrintWriter output, Scenario scenario, Config config) {
        boolean failed = false;
        if (scenario.k() > scenario.vectors()) {
            for (String backend : config.backends()) {
                emitSkip(output, backend, scenario, "k_exceeds_vector_count");
            }
            return false;
        }
        if (scenario.datasetBytes() > config.maxDatasetBytes()) {
            for (String backend : config.backends()) {
                emitSkip(output, backend, scenario, "dataset_exceeds_max_dataset_bytes");
            }
            return false;
        }
        String feasibility = feasibilityFailure(scenario);
        if (feasibility != null) {
            for (String backend : config.backends()) {
                emitSkip(output, backend, scenario, feasibility);
            }
            return false;
        }

        long seed = config.seed() ^ scenario.vectors() ^ ((long) scenario.dimensions() << 16)
                ^ ((long) scenario.batch() << 32) ^ scenario.k();
        float[][] vectors = randomVectors(scenario.vectors(), scenario.dimensions(), seed);
        float[][] queries = randomVectors(scenario.batch(), scenario.dimensions(), seed ^ 0x5DEECE66DL);
        long[] ids = sequentialIds(scenario.vectors());
        SearchParameters parameters = new SearchParameters(scenario.metric());

        List<List<SearchResult>> baseline;
        try (CpuBruteForceIndex cpu = new CpuBruteForceIndex()) {
            cpu.build(vectors, ids);
            baseline = cpu.searchBatch(queries, scenario.k(), parameters);
        }

        for (String backend : config.backends()) {
            if (!supportsMetric(backend, scenario.metric())) {
                emitSkip(output, backend, scenario, "metric_not_supported");
                continue;
            }
            if (!available(backend)) {
                emitSkip(output, backend, scenario, "backend_unavailable");
                continue;
            }
            try {
                if (backend.equals(config.forceErrorBackend())) {
                    throw new IllegalStateException("forced validation error");
                }
                measure(output, backend, scenario, config, vectors, ids, queries, parameters, baseline);
            } catch (RuntimeException | LinkageError error) {
                emitError(output, backend, scenario,
                        "execution_failed:" + error.getClass().getSimpleName() + ":" + safeMessage(error));
                failed = true;
            }
        }
        return failed;
    }

    private static void measure(
            PrintWriter output,
            String backend,
            Scenario scenario,
            Config config,
            float[][] vectors,
            long[] ids,
            float[][] queries,
            SearchParameters parameters,
            List<List<SearchResult>> baseline
    ) {
        long heapBefore = heapUsed();
        long rssBefore = residentBytes();
        long gpuBefore = gpuMemoryUsedBytes();
        try (VectorIndex index = createBackend(backend)) {
            long buildStarted = System.nanoTime();
            index.build(vectors, ids);
            double buildMillis = elapsedMillis(buildStarted);
            long heapAfterBuild = heapUsed();
            long rssAfterBuild = residentBytes();
            long gpuAfterBuild = gpuMemoryUsedBytes();

            List<List<SearchResult>> first = searchBatch(index, queries, scenario.k(), parameters);
            double recall = recallAtK(baseline, first, scenario.k());
            for (int i = 0; i < config.warmup(); i++) {
                searchBatch(index, queries, scenario.k(), parameters);
            }

            double[] batchMillis = new double[config.iterations()];
            double h2d = 0.0;
            double kernel = 0.0;
            double d2h = 0.0;
            int cudaTimingSamples = 0;
            for (int i = 0; i < batchMillis.length; i++) {
                long started = System.nanoTime();
                searchBatch(index, queries, scenario.k(), parameters);
                batchMillis[i] = elapsedMillis(started);
                if (index instanceof CudaBruteForceIndex cuda) {
                    CudaSearchTimings timings = cuda.lastSearchTimings();
                    if (timings != null) {
                        h2d += timings.hostToDeviceMillis();
                        kernel += timings.kernelMillis();
                        d2h += timings.deviceToHostMillis();
                        cudaTimingSamples++;
                    }
                }
            }

            double[] sorted = Arrays.copyOf(batchMillis, batchMillis.length);
            Arrays.sort(sorted);
            double totalSeconds = Arrays.stream(batchMillis).sum() / 1000.0;
            double qps = (scenario.batch() * (double) batchMillis.length) / totalSeconds;
            emitResult(output, backend, scenario, config, buildMillis, recall, sorted, batchMillis, qps,
                    heapAfterBuild - heapBefore,
                    subtractIfKnown(rssAfterBuild, rssBefore),
                    subtractIfKnown(gpuAfterBuild, gpuBefore),
                    cudaTimingSamples == 0 ? null : h2d / cudaTimingSamples,
                    cudaTimingSamples == 0 ? null : kernel / cudaTimingSamples,
                    cudaTimingSamples == 0 ? null : d2h / cudaTimingSamples);
        }
    }

    private static void emitMetadata(PrintWriter output, Config config) {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("record_type", "metadata");
        values.put("schema_version", 1);
        values.put("run_id", UUID.randomUUID().toString());
        values.put("timestamp_utc", Instant.now().toString());
        values.put("mode", config.mode());
        values.put("seed", config.seed());
        values.put("warmup_iterations", config.warmup());
        values.put("measurement_iterations", config.iterations());
        values.put("max_dataset_bytes", config.maxDatasetBytes());
        values.put("vector_counts", config.vectorCounts());
        values.put("dimensions", config.dimensions());
        values.put("batch_sizes", config.batchSizes());
        values.put("k_values", config.kValues());
        values.put("metrics", config.metrics());
        values.put("backends", config.backends());
        values.put("os_name", System.getProperty("os.name"));
        values.put("os_version", System.getProperty("os.version"));
        values.put("os_arch", System.getProperty("os.arch"));
        values.put("jvm", System.getProperty("java.vm.name"));
        values.put("java_version", System.getProperty("java.version"));
        values.put("jvm_input_args", ManagementFactory.getRuntimeMXBean().getInputArguments());
        values.put("invocation", System.getProperty("sun.java.command"));
        values.put("processors", Runtime.getRuntime().availableProcessors());
        values.put("cpu", cpuDescription());
        values.put("max_heap_bytes", Runtime.getRuntime().maxMemory());
        values.put("compiler", env("CXX"));
        values.put("compiler_version", compilerVersion());
        values.put("cuda_toolkit_root", env("CUDAToolkit_ROOT"));
        values.put("cuda_version", cudaVersion());
        values.put("conda_prefix", env("CONDA_PREFIX"));
        values.put("cuda_device_count", safeCudaDeviceCount());
        values.put("cuvs_version", safeCuvsVersion());
        values.put("gpu", gpuDescription());
        values.put("git_sha", command("git", "rev-parse", "HEAD"));
        values.put("git_dirty", gitDirty());
        values.put("maven_version", firstLine(command("mvn", "--version")));
        values.put("cmake_version", firstLine(command("cmake", "--version")));
        values.put("measurement_note",
                "End-to-end Java batch wall clock; build separate; heap/RSS/GPU deltas are approximate.");
        values.put("percentile_note", config.iterations() < 20
                ? "p95/p99 are order statistics from a tiny sample and are not stable tail estimates."
                : "p95/p99 are order statistics from the configured sample count.");
        output.println(json(values));
    }

    private static void emitResult(
            PrintWriter output,
            String backend,
            Scenario scenario,
            Config config,
            double buildMillis,
            double recall,
            double[] sorted,
            double[] raw,
            double qps,
            long heapDelta,
            Long rssDelta,
            Long gpuDelta,
            Double h2d,
            Double kernel,
            Double d2h
    ) {
        Map<String, Object> values = common("result", backend, scenario);
        values.put("warmup_iterations", config.warmup());
        values.put("measurement_iterations", config.iterations());
        values.put("build_ms", buildMillis);
        values.put("end_to_end_batch_avg_ms", Arrays.stream(raw).average().orElseThrow());
        values.put("end_to_end_batch_p50_ms", percentile(sorted, 0.50));
        values.put("end_to_end_batch_p95_ms", percentile(sorted, 0.95));
        values.put("end_to_end_batch_p99_ms", percentile(sorted, 0.99));
        values.put("qps", qps);
        values.put("recall_at_k", recall);
        values.put("heap_delta_bytes", heapDelta);
        values.put("process_rss_delta_bytes", rssDelta);
        values.put("gpu_memory_delta_bytes", gpuDelta);
        values.put("cuda_h2d_avg_ms", h2d);
        values.put("cuda_kernel_avg_ms", kernel);
        values.put("cuda_d2h_avg_ms", d2h);
        values.put("raw_batch_ms", raw);
        output.println(json(values));
    }

    private static void emitSkip(PrintWriter output, String backend, Scenario scenario, String reason) {
        Map<String, Object> values = common("skip", backend, scenario);
        values.put("reason", reason);
        output.println(json(values));
    }

    private static void emitError(PrintWriter output, String backend, Scenario scenario, String reason) {
        Map<String, Object> values = common("error", backend, scenario);
        values.put("reason", reason);
        output.println(json(values));
    }

    private static Map<String, Object> common(String type, String backend, Scenario scenario) {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("record_type", type);
        values.put("backend", backend);
        values.put("vectors", scenario.vectors());
        values.put("dimensions", scenario.dimensions());
        values.put("batch_size", scenario.batch());
        values.put("k", scenario.k());
        values.put("metric", scenario.metric().name());
        values.put("dataset_bytes", scenario.datasetBytes());
        return values;
    }

    private static VectorIndex createBackend(String backend) {
        return switch (backend) {
            case "cpu" -> new CpuBruteForceIndex();
            case "native" -> new NativeBruteForceIndex();
            case "cuda" -> new CudaBruteForceIndex();
            case "cuvs" -> new CuvsVectorIndex();
            default -> throw new IllegalArgumentException("unsupported backend: " + backend);
        };
    }

    private static boolean available(String backend) {
        return switch (backend) {
            case "cpu" -> true;
            case "native" -> nativeAvailable();
            case "cuda" -> CudaBruteForceIndex.isCudaAvailable();
            case "cuvs" -> CuvsVectorIndex.isCuvsAvailable();
            default -> false;
        };
    }

    private static boolean supportsMetric(String backend, DistanceMetric metric) {
        return !"cuda".equals(backend) || metric == DistanceMetric.DOT_PRODUCT;
    }

    private static List<List<SearchResult>> searchBatch(
            VectorIndex index, float[][] queries, int k, SearchParameters parameters) {
        if (index instanceof CpuBruteForceIndex cpu) {
            return cpu.searchBatch(queries, k, parameters);
        }
        if (index instanceof NativeBruteForceIndex nativeIndex) {
            return nativeIndex.searchBatch(queries, k, parameters);
        }
        if (index instanceof CudaBruteForceIndex cuda) {
            return cuda.searchBatch(queries, k, parameters);
        }
        if (index instanceof CuvsVectorIndex cuvs) {
            return cuvs.searchBatch(queries, k, parameters);
        }
        throw new IllegalArgumentException("backend does not expose batch search");
    }

    private static double recallAtK(
            List<List<SearchResult>> expected, List<List<SearchResult>> actual, int k) {
        long matches = 0;
        for (int query = 0; query < expected.size(); query++) {
            var actualIds = actual.get(query).stream().map(SearchResult::id).collect(java.util.stream.Collectors.toSet());
            for (SearchResult result : expected.get(query)) {
                if (actualIds.contains(result.id())) {
                    matches++;
                }
            }
        }
        return matches / (double) (expected.size() * k);
    }

    private static float[][] randomVectors(int count, int dimensions, long seed) {
        Random random = new Random(seed);
        float[][] values = new float[count][dimensions];
        for (float[] value : values) {
            for (int i = 0; i < dimensions; i++) {
                value[i] = (random.nextFloat() * 2.0F) - 1.0F;
            }
        }
        return values;
    }

    private static long[] sequentialIds(int count) {
        long[] ids = new long[count];
        for (int i = 0; i < count; i++) {
            ids[i] = i + 1L;
        }
        return ids;
    }

    private static long heapUsed() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static String feasibilityFailure(Scenario scenario) {
        try {
            long queryBytes = Math.multiplyExact(
                    Math.multiplyExact((long) scenario.batch(), scenario.dimensions()), Float.BYTES);
            long resultCount = Math.multiplyExact((long) scenario.batch(), scenario.k());
            long resultBytes = Math.multiplyExact(resultCount, 64L);
            long idsBytes = Math.multiplyExact((long) scenario.vectors(), Long.BYTES);
            long estimatedHeap = Math.addExact(Math.addExact(scenario.datasetBytes(), queryBytes),
                    Math.addExact(idsBytes, resultBytes));
            if (estimatedHeap > Runtime.getRuntime().maxMemory() * 7L / 10L) {
                return "estimated_java_heap_exceeds_70_percent";
            }
            return null;
        } catch (ArithmeticException error) {
            return "scenario_size_overflow";
        }
    }

    private static long residentBytes() {
        try {
            for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
                if (line.startsWith("VmRSS:")) {
                    return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024L;
                }
            }
        } catch (IOException | RuntimeException ignored) {
            // Not available on this platform.
        }
        return -1L;
    }

    private static long gpuMemoryUsedBytes() {
        String value = command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits");
        if (value == null) {
            return -1L;
        }
        try {
            return Long.parseLong(value.lines().findFirst().orElseThrow().trim()) * 1024L * 1024L;
        } catch (RuntimeException ignored) {
            return -1L;
        }
    }

    private static String gpuDescription() {
        return command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader");
    }

    private static String cpuDescription() {
        try {
            return Files.readAllLines(Path.of("/proc/cpuinfo")).stream()
                    .filter(line -> line.startsWith("model name"))
                    .map(line -> line.substring(line.indexOf(':') + 1).trim())
                    .findFirst()
                    .orElse(null);
        } catch (IOException | RuntimeException ignored) {
            return null;
        }
    }

    private static String compilerVersion() {
        String compiler = env("CXX");
        return compiler == null ? null : command(compiler, "--version");
    }

    private static String cudaVersion() {
        String root = env("CUDAToolkit_ROOT");
        return root == null ? command("nvcc", "--version") : command(root + "/bin/nvcc", "--version");
    }

    private static boolean nativeAvailable() {
        try (NativeBruteForceIndex ignored = new NativeBruteForceIndex()) {
            return true;
        } catch (RuntimeException | LinkageError ignored) {
            return false;
        }
    }

    private static Boolean gitDirty() {
        String status = command("git", "status", "--porcelain");
        return status == null ? null : !status.isBlank();
    }

    private static String firstLine(String value) {
        return value == null ? null : value.lines().findFirst().orElse(null);
    }

    private static String command(String... command) {
        try {
            Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
            String text = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
            return process.waitFor() == 0 ? text : null;
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
            return null;
        } catch (IOException ignored) {
            return null;
        }
    }

    private static int safeCudaDeviceCount() {
        try {
            return NativeBindings.getCudaDeviceCount();
        } catch (RuntimeException | LinkageError ignored) {
            return -1;
        }
    }

    private static String safeCuvsVersion() {
        try {
            return NativeBindings.isCuvsCompiled() ? NativeBindings.getCuvsVersion() : null;
        } catch (RuntimeException | LinkageError ignored) {
            return null;
        }
    }

    private static String env(String name) {
        return System.getenv(name);
    }

    private static Long subtractIfKnown(long after, long before) {
        return after < 0 || before < 0 ? null : after - before;
    }

    private static double percentile(double[] sorted, double value) {
        int index = Math.max(0, Math.min(sorted.length - 1, (int) Math.ceil(value * sorted.length) - 1));
        return sorted[index];
    }

    private static double elapsedMillis(long started) {
        return (System.nanoTime() - started) / 1_000_000.0;
    }

    private static String safeMessage(Throwable error) {
        return error.getMessage() == null ? "" : error.getMessage().replace('\n', ' ');
    }

    private static String json(Map<String, Object> values) {
        StringBuilder output = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Object> entry : values.entrySet()) {
            if (!first) {
                output.append(',');
            }
            first = false;
            output.append('"').append(escape(entry.getKey())).append("\":").append(jsonValue(entry.getValue()));
        }
        return output.append('}').toString();
    }

    private static String jsonValue(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof Number || value instanceof Boolean) {
            return value.toString();
        }
        if (value instanceof double[] values) {
            StringBuilder output = new StringBuilder("[");
            for (int i = 0; i < values.length; i++) {
                if (i > 0) {
                    output.append(',');
                }
                output.append(String.format(Locale.US, "%.6f", values[i]));
            }
            return output.append(']').toString();
        }
        if (value instanceof int[] values) {
            return Arrays.toString(values).replace(" ", "");
        }
        if (value instanceof Iterable<?> values) {
            StringBuilder output = new StringBuilder("[");
            boolean first = true;
            for (Object item : values) {
                if (!first) output.append(',');
                first = false;
                output.append(jsonValue(item));
            }
            return output.append(']').toString();
        }
        return '"' + escape(value.toString()) + '"';
    }

    private static String escape(String value) {
        StringBuilder escaped = new StringBuilder();
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"' -> escaped.append("\\\"");
                case '\\' -> escaped.append("\\\\");
                case '\b' -> escaped.append("\\b");
                case '\f' -> escaped.append("\\f");
                case '\n' -> escaped.append("\\n");
                case '\r' -> escaped.append("\\r");
                case '\t' -> escaped.append("\\t");
                default -> {
                    if (c < 0x20) escaped.append(String.format("\\u%04x", (int) c));
                    else escaped.append(c);
                }
            }
        }
        return escaped.toString();
    }

    private record Scenario(
            int vectors, int dimensions, int batch, int k, DistanceMetric metric, long datasetBytes) {
    }

    private record Config(
            String mode,
            int[] vectorCounts,
            int[] dimensions,
            int[] batchSizes,
            int[] kValues,
            List<DistanceMetric> metrics,
            List<String> backends,
            int warmup,
            int iterations,
            long seed,
            long maxDatasetBytes,
            Path output,
            String forceErrorBackend
    ) {
        private static Config parse(String[] args) {
            Map<String, String> options = new LinkedHashMap<>();
            Set<String> known = Set.of("--mode", "--vectors", "--dimensions", "--batches", "--k",
                    "--metrics", "--backends", "--warmup", "--iterations", "--seed",
                    "--max-dataset-bytes", "--output", "--force-error-backend");
            for (int i = 0; i < args.length; i += 2) {
                if (!args[i].startsWith("--") || i + 1 >= args.length) {
                    throw new IllegalArgumentException("options require --name value pairs");
                }
                if (!known.contains(args[i])) throw new IllegalArgumentException("unknown option: " + args[i]);
                if (options.put(args[i], args[i + 1]) != null) {
                    throw new IllegalArgumentException("duplicate option: " + args[i]);
                }
            }
            String mode = options.getOrDefault("--mode", "small");
            Preset preset = Preset.forName(mode);
            Config config = new Config(
                    mode,
                    ints(options.get("--vectors"), preset.vectors()),
                    ints(options.get("--dimensions"), preset.dimensions()),
                    ints(options.get("--batches"), preset.batches()),
                    ints(options.get("--k"), preset.k()),
                    metrics(options.get("--metrics")),
                    strings(options.getOrDefault("--backends", "cpu,native,cuda,cuvs")),
                    integer(options, "--warmup", preset.warmup()),
                    integer(options, "--iterations", preset.iterations()),
                    Long.parseLong(options.getOrDefault("--seed", "113")),
                    Long.parseLong(options.getOrDefault("--max-dataset-bytes",
                            Long.toString(preset.maxDatasetBytes()))),
                    options.containsKey("--output") ? Path.of(options.get("--output")) : null,
                    options.get("--force-error-backend")
            );
            config.validate();
            return config;
        }

        private void validate() {
            requirePositive(vectorCounts, "vectors");
            requirePositive(dimensions, "dimensions");
            requirePositive(batchSizes, "batches");
            requirePositive(kValues, "k");
            if (warmup < 0) throw new IllegalArgumentException("warmup must be non-negative");
            if (iterations <= 0) throw new IllegalArgumentException("iterations must be positive");
            if (maxDatasetBytes <= 0) throw new IllegalArgumentException("max-dataset-bytes must be positive");
            if (metrics.isEmpty() || backends.isEmpty()) throw new IllegalArgumentException("metrics/backends must not be empty");
            Set<String> allowed = Set.of("cpu", "native", "cuda", "cuvs");
            if (!allowed.containsAll(backends)) throw new IllegalArgumentException("unknown backend in " + backends);
            if (forceErrorBackend != null && !allowed.contains(forceErrorBackend)) {
                throw new IllegalArgumentException("unknown force-error-backend: " + forceErrorBackend);
            }
            if (Arrays.stream(kValues).max().orElseThrow() > Arrays.stream(vectorCounts).min().orElseThrow()) {
                throw new IllegalArgumentException("every k must be <= every vector count");
            }
        }

        private static void requirePositive(int[] values, String name) {
            if (values.length == 0 || Arrays.stream(values).anyMatch(value -> value <= 0)) {
                throw new IllegalArgumentException(name + " must be a nonempty list of positive integers");
            }
        }

        private static int integer(Map<String, String> options, String key, int fallback) {
            return Integer.parseInt(options.getOrDefault(key, Integer.toString(fallback)));
        }

        private static int[] ints(String value, int[] fallback) {
            if (value == null) {
                return fallback;
            }
            return strings(value).stream().mapToInt(Integer::parseInt).toArray();
        }

        private static List<String> strings(String value) {
            List<String> parsed = Arrays.stream(value.split(",", -1)).map(String::trim).toList();
            if (parsed.isEmpty() || parsed.stream().anyMatch(String::isEmpty)
                    || new java.util.LinkedHashSet<>(parsed).size() != parsed.size()) {
                throw new IllegalArgumentException("lists must be nonempty and contain unique nonempty values");
            }
            return parsed;
        }

        private static List<DistanceMetric> metrics(String value) {
            if (value == null) {
                return List.of(DistanceMetric.EUCLIDEAN, DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT);
            }
            return strings(value).stream().map(item -> DistanceMetric.valueOf(item.toUpperCase(Locale.ROOT))).toList();
        }
    }

    private record Preset(
            int[] vectors, int[] dimensions, int[] batches, int[] k,
            int warmup, int iterations, long maxDatasetBytes) {
        private static Preset forName(String mode) {
            return switch (mode) {
                case "smoke" -> new Preset(new int[]{10_000}, new int[]{128}, new int[]{1},
                        new int[]{10}, 1, 3, 512L * 1024L * 1024L);
                case "small" -> new Preset(new int[]{10_000}, new int[]{128, 384},
                        new int[]{1, 8, 32}, new int[]{1, 10}, 5, 10, 1024L * 1024L * 1024L);
                case "large" -> new Preset(new int[]{100_000, 1_000_000}, new int[]{384, 768},
                        new int[]{32, 128}, new int[]{10, 100}, 3, 5, 4L * 1024L * 1024L * 1024L);
                case "matrix" -> new Preset(new int[]{10_000, 100_000, 1_000_000},
                        new int[]{128, 384, 768}, new int[]{1, 8, 32, 128},
                        new int[]{1, 10, 100}, 3, 5, 4L * 1024L * 1024L * 1024L);
                default -> throw new IllegalArgumentException("unsupported mode: " + mode);
            };
        }
    }
}

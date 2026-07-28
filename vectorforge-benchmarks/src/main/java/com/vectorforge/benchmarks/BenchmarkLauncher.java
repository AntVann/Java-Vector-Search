package com.vectorforge.benchmarks;

import com.vectorforge.gpu.CudaBruteForceIndex;
import com.vectorforge.gpu.CuvsVectorIndex;
import com.vectorforge.nativeindex.NativeBruteForceIndex;
import org.openjdk.jmh.Main;

import java.util.ArrayList;
import java.util.List;

/**
 * Selects only usable backends when no explicit JMH benchmark expression is supplied.
 */
public final class BenchmarkLauncher {

    private BenchmarkLauncher() {
    }

    public static void main(String[] args) throws Exception {
        if (hasBenchmarkExpression(args)) {
            Main.main(args);
            return;
        }

        List<String> methods = new ArrayList<>();
        methods.add(".*BackendSearchBenchmark.cpuSearch");
        methods.add(".*IndexBuildBenchmark.cpuBuild");
        if (nativeAvailable()) {
            methods.add(".*BackendSearchBenchmark.nativeSearch");
            methods.add(".*IndexBuildBenchmark.nativeBuild");
        }
        if (CudaBruteForceIndex.isCudaAvailable()) {
            methods.add(".*BackendSearchBenchmark.cudaSearch");
            methods.add(".*IndexBuildBenchmark.cudaBuild");
        }
        if (CuvsVectorIndex.isCuvsAvailable()) {
            methods.add(".*BackendSearchBenchmark.cuvsSearch");
            methods.add(".*IndexBuildBenchmark.cuvsBuild");
        }

        String expression = String.join("|", methods);
        String[] forwarded = new String[args.length + 1];
        forwarded[0] = expression;
        System.arraycopy(args, 0, forwarded, 1, args.length);
        System.out.println("Enabled JMH backends: " + String.join(", ", backendNames(methods)));
        Main.main(forwarded);
    }

    private static boolean nativeAvailable() {
        try (NativeBruteForceIndex ignored = new NativeBruteForceIndex()) {
            return true;
        } catch (RuntimeException | LinkageError ignored) {
            return false;
        }
    }

    private static boolean hasBenchmarkExpression(String[] args) {
        return args.length > 0 && !args[0].startsWith("-");
    }

    private static List<String> backendNames(List<String> methods) {
        return methods.stream()
                .map(method -> method.substring(method.lastIndexOf('.') + 1)
                        .replace("Search", "")
                        .replace("Build", ""))
                .distinct()
                .toList();
    }
}

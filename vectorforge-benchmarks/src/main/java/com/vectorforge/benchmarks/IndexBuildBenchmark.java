package com.vectorforge.benchmarks;

import com.vectorforge.api.IndexMetrics;
import com.vectorforge.api.VectorIndex;
import com.vectorforge.cpu.CpuBruteForceIndex;
import com.vectorforge.gpu.CudaBruteForceIndex;
import com.vectorforge.gpu.CuvsVectorIndex;
import com.vectorforge.nativeindex.NativeBruteForceIndex;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/**
 * Measures index construction and destruction. Input allocation is deliberately outside timing.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(1)
public class IndexBuildBenchmark {

    @Benchmark
    public void cpuBuild(BuildState state, Blackhole blackhole) {
        state.buildAndClose(CpuBruteForceIndex::new, blackhole);
    }

    @Benchmark
    public void nativeBuild(BuildState state, Blackhole blackhole) {
        state.buildAndClose(NativeBruteForceIndex::new, blackhole);
    }

    @Benchmark
    public void cudaBuild(BuildState state, Blackhole blackhole) {
        state.buildAndClose(CudaBruteForceIndex::new, blackhole);
    }

    @Benchmark
    public void cuvsBuild(BuildState state, Blackhole blackhole) {
        state.buildAndClose(CuvsVectorIndex::new, blackhole);
    }

    @State(Scope.Benchmark)
    public static class BuildState {

        @Param({"1000", "10000"})
        public int vectorCount;

        @Param({"64", "128"})
        public int dimensions;

        private float[][] vectors;
        private long[] ids;

        @Setup(Level.Trial)
        public void setUp() {
            Random random = new Random(0xB17D5EEDL);
            vectors = new float[vectorCount][dimensions];
            ids = new long[vectorCount];
            for (int row = 0; row < vectorCount; row++) {
                ids[row] = row + 1L;
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    vectors[row][dimension] = random.nextFloat() * 2.0f - 1.0f;
                }
            }
        }

        void buildAndClose(Supplier<? extends VectorIndex> factory, Blackhole blackhole) {
            try (VectorIndex index = factory.get()) {
                index.build(vectors, ids);
                IndexMetrics metrics = index.metrics();
                blackhole.consume(metrics);
            }
        }
    }
}

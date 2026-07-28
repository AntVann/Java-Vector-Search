package com.vectorforge.benchmarks;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
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
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class BackendSearchBenchmark {

    @Benchmark
    public void cpuSearch(CpuState state, Blackhole blackhole) {
        blackhole.consume(state.search());
    }

    @Benchmark
    public void nativeSearch(NativeState state, Blackhole blackhole) {
        blackhole.consume(state.search());
    }

    @Benchmark
    public void cudaSearch(CudaState state, Blackhole blackhole) {
        blackhole.consume(state.search());
    }

    @Benchmark
    public void cuvsSearch(CuvsState state, Blackhole blackhole) {
        blackhole.consume(state.search());
    }

    @State(Scope.Benchmark)
    public abstract static class SearchState {

        @Param({"1000", "10000"})
        public int vectorCount;

        @Param({"64", "128"})
        public int dimensions;

        @Param({"1", "16"})
        public int batchSize;

        @Param({"10"})
        public int k;

        protected VectorIndex index;
        protected float[][] queries;
        protected SearchParameters parameters;

        protected final void initialize(VectorIndex newIndex, DistanceMetric metric) {
            if (k > vectorCount) {
                throw new IllegalArgumentException("k must not exceed vectorCount");
            }
            Random random = new Random(0x5EEDL);
            float[][] vectors = randomVectors(random, vectorCount, dimensions);
            long[] ids = new long[vectorCount];
            for (int i = 0; i < vectorCount; i++) {
                ids[i] = i + 1L;
            }
            queries = randomVectors(random, batchSize, dimensions);
            parameters = new SearchParameters(metric);
            index = newIndex;
            index.build(vectors, ids);
        }

        final Object search() {
            if (batchSize == 1) {
                return index.search(queries[0], k, parameters);
            }
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
            throw new IllegalStateException("unsupported benchmark backend");
        }

        @TearDown(Level.Trial)
        public final void close() {
            if (index != null) {
                index.close();
            }
        }

        private static float[][] randomVectors(Random random, int count, int dimensions) {
            float[][] values = new float[count][dimensions];
            for (float[] value : values) {
                for (int dimension = 0; dimension < value.length; dimension++) {
                    value[dimension] = random.nextFloat() * 2.0f - 1.0f;
                }
            }
            return values;
        }
    }

    @State(Scope.Benchmark)
    public static class CpuState extends MetricState {
        @Setup(Level.Trial)
        public void setUp() {
            initialize(new CpuBruteForceIndex(), parsedMetric());
        }
    }

    @State(Scope.Benchmark)
    public static class NativeState extends MetricState {
        @Setup(Level.Trial)
        public void setUp() {
            initialize(new NativeBruteForceIndex(), parsedMetric());
        }
    }

    @State(Scope.Benchmark)
    public static class CuvsState extends MetricState {
        @Setup(Level.Trial)
        public void setUp() {
            initialize(new CuvsVectorIndex(), parsedMetric());
        }
    }

    @State(Scope.Benchmark)
    public abstract static class MetricState extends SearchState {
        @Param({"EUCLIDEAN", "COSINE", "DOT_PRODUCT"})
        public String metric;

        final DistanceMetric parsedMetric() {
            return DistanceMetric.valueOf(metric);
        }
    }

    @State(Scope.Benchmark)
    public static class CudaState extends SearchState {
        @Setup(Level.Trial)
        public void setUp() {
            initialize(new CudaBruteForceIndex(), DistanceMetric.DOT_PRODUCT);
        }
    }
}

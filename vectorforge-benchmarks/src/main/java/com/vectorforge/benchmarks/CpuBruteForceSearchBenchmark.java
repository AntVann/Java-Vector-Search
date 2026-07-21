package com.vectorforge.benchmarks;

import com.vectorforge.api.DistanceMetric;
import com.vectorforge.api.SearchParameters;
import com.vectorforge.cpu.CpuBruteForceIndex;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
public class CpuBruteForceSearchBenchmark {

    @Param({"10000"})
    public int vectorCount;

    @Param({"128"})
    public int dimensions;

    @Param({"10"})
    public int k;

    private CpuBruteForceIndex index;
    private float[] query;

    @Setup(Level.Trial)
    public void setUp() {
        Random random = new Random(42L);
        float[][] vectors = new float[vectorCount][dimensions];
        long[] ids = new long[vectorCount];

        for (int i = 0; i < vectorCount; i++) {
            ids[i] = i + 1L;
            for (int j = 0; j < dimensions; j++) {
                vectors[i][j] = (random.nextFloat() * 2.0f) - 1.0f;
            }
        }

        query = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            query[i] = (random.nextFloat() * 2.0f) - 1.0f;
        }

        index = new CpuBruteForceIndex();
        index.build(vectors, ids);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (index != null) {
            index.close();
        }
    }

    @Benchmark
    public Object searchTopK() {
        return index.search(query, k, new SearchParameters(DistanceMetric.EUCLIDEAN));
    }
}


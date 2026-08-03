package com.vectorforge.nativeindex;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeEnvironmentReportTest {

    @Test
    void diagnosticMessageIncludesAttemptsAndRuntimeRemediation() {
        String message = NativeLibraryLoader.diagnosticMessage(List.of(
                "explicit=/missing/vectorforge_jni",
                "System.loadLibrary(vectorforge_jni)"
        ));

        assertTrue(message.contains("explicit=/missing/vectorforge_jni"));
        assertTrue(message.contains("os="));
        assertTrue(message.contains("java.library.path="));
        assertTrue(message.contains("PATH on Windows"));
        assertTrue(message.contains("LD_LIBRARY_PATH on Linux"));
        assertTrue(message.contains("do not bundle"));
    }

    @Test
    void reportIsUsefulWhetherTheOptionalLibraryIsPresentOrAbsent() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        boolean loaded = NativeEnvironmentReport.print(
                new PrintStream(bytes, true, StandardCharsets.UTF_8));
        String report = bytes.toString(StandardCharsets.UTF_8);

        assertTrue(report.contains("os.name="));
        assertTrue(report.contains("java.version="));
        assertTrue(report.contains("native.loaded=" + loaded));
        if (loaded) {
            assertTrue(report.contains("cuda.compiled="));
            assertTrue(report.contains("cuvs.compiled="));
        } else {
            assertTrue(report.contains("native.error="));
            assertTrue(report.contains("remediation="));
        }
        assertFalse(report.contains("\n\n"));
    }

    @Test
    void loaderFailureRetainsEveryEarlierAttempt() {
        UnsatisfiedLinkError first = new UnsatisfiedLinkError("explicit file failed");
        UnsatisfiedLinkError second = new UnsatisfiedLinkError("explicit directory failed");
        UnsatisfiedLinkError last = new UnsatisfiedLinkError("library path failed");

        NativeInteropException failure = NativeLibraryLoader.loadFailure(
                List.of("first", "second", "last"),
                List.of(first, second, last)
        );

        assertSame(last, failure.getCause());
        assertTrue(java.util.Arrays.equals(
                new Throwable[]{first, second},
                failure.getSuppressed()
        ));
    }
}

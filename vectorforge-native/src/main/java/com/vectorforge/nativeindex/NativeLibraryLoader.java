package com.vectorforge.nativeindex;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

final class NativeLibraryLoader {

    private static final String LIBRARY_BASENAME = "vectorforge_jni";
    private static final AtomicBoolean LOADED = new AtomicBoolean();
    private static final Object LOAD_LOCK = new Object();

    private NativeLibraryLoader() {
    }

    static void load() {
        if (LOADED.get()) {
            return;
        }

        synchronized (LOAD_LOCK) {
            if (LOADED.get()) {
                return;
            }

            List<String> attempts = new ArrayList<>();
            List<UnsatisfiedLinkError> failures = new ArrayList<>();

            String explicitLibrary = System.getProperty("vectorforge.native.library");
            if (explicitLibrary != null && !explicitLibrary.isBlank()) {
                Path path = Path.of(explicitLibrary).toAbsolutePath();
                attempts.add("vectorforge.native.library=" + path + " (exists=" + Files.exists(path) + ")");
                try {
                    System.load(path.toString());
                    LOADED.set(true);
                    return;
                } catch (UnsatisfiedLinkError ex) {
                    failures.add(ex);
                }
            }

            String explicitDirectory = System.getProperty("vectorforge.native.library.dir");
            if (explicitDirectory != null && !explicitDirectory.isBlank()) {
                Path libraryPath = Path.of(explicitDirectory)
                        .resolve(System.mapLibraryName(LIBRARY_BASENAME))
                        .toAbsolutePath();
                attempts.add("vectorforge.native.library.dir=" + libraryPath
                        + " (exists=" + Files.exists(libraryPath) + ")");
                try {
                    System.load(libraryPath.toString());
                    LOADED.set(true);
                    return;
                } catch (UnsatisfiedLinkError ex) {
                    failures.add(ex);
                }
            }

            attempts.add("System.loadLibrary(" + LIBRARY_BASENAME + ")");
            try {
                System.loadLibrary(LIBRARY_BASENAME);
                LOADED.set(true);
            } catch (UnsatisfiedLinkError ex) {
                failures.add(ex);
                throw loadFailure(attempts, failures);
            }
        }
    }

    static NativeInteropException loadFailure(
            List<String> attempts,
            List<UnsatisfiedLinkError> failures
    ) {
        if (failures.isEmpty()) {
            throw new IllegalArgumentException("failures must not be empty");
        }
        UnsatisfiedLinkError cause = failures.getLast();
        NativeInteropException failure = new NativeInteropException(
                diagnosticMessage(attempts),
                cause
        );
        for (int i = 0; i < failures.size() - 1; i++) {
            failure.addSuppressed(failures.get(i));
        }
        return failure;
    }

    static String diagnosticMessage(List<String> attempts) {
        return "Unable to load the VectorForge native library. "
                + "Attempts: " + String.join("; ", attempts) + ". "
                + "Runtime: os=" + System.getProperty("os.name")
                + ", arch=" + System.getProperty("os.arch")
                + ", java.library.path=" + System.getProperty("java.library.path") + ". "
                + "Set vectorforge.native.library or vectorforge.native.library.dir, "
                + "or build with -Pnative/-Pcuda/-Pcuvs. If the JNI file exists, this error "
                + "usually means a dependent library is missing; configure PATH on Windows "
                + "or LD_LIBRARY_PATH on Linux. cuVS builds require the matching cuVS, RAPIDS, "
                + "and CUDA runtime libraries and do not bundle them.";
    }
}

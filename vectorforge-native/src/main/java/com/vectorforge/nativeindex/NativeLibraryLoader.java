package com.vectorforge.nativeindex;

import java.nio.file.Files;
import java.nio.file.Path;
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

            UnsatisfiedLinkError priorFailure = null;

            String explicitLibrary = System.getProperty("vectorforge.native.library");
            if (explicitLibrary != null && !explicitLibrary.isBlank()) {
                try {
                    System.load(explicitLibrary);
                    LOADED.set(true);
                    return;
                } catch (UnsatisfiedLinkError ex) {
                    priorFailure = ex;
                }
            }

            String explicitDirectory = System.getProperty("vectorforge.native.library.dir");
            if (explicitDirectory != null && !explicitDirectory.isBlank()) {
                Path libraryPath = Path.of(explicitDirectory).resolve(System.mapLibraryName(LIBRARY_BASENAME));
                if (Files.exists(libraryPath)) {
                    try {
                        System.load(libraryPath.toAbsolutePath().toString());
                        LOADED.set(true);
                        return;
                    } catch (UnsatisfiedLinkError ex) {
                        priorFailure = ex;
                    }
                }
            }

            try {
                System.loadLibrary(LIBRARY_BASENAME);
                LOADED.set(true);
            } catch (UnsatisfiedLinkError ex) {
                if (priorFailure != null) {
                    ex.addSuppressed(priorFailure);
                }
                throw new NativeInteropException(
                        "Unable to load the VectorForge native library. "
                                + "Set vectorforge.native.library or vectorforge.native.library.dir, "
                                + "or build the project with the native profile.",
                        ex
                );
            }
        }
    }
}


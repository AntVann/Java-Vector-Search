package com.vectorforge.nativeindex;

import java.io.PrintStream;
import java.util.Objects;

/**
 * Prints a small, read-only diagnostic report for native backend discovery.
 */
public final class NativeEnvironmentReport {

    private NativeEnvironmentReport() {
    }

    public static void main(String[] args) {
        if (!print(System.out)) {
            System.exit(2);
        }
    }

    public static boolean print(PrintStream output) {
        Objects.requireNonNull(output, "output must not be null");
        output.println("os.name=" + System.getProperty("os.name"));
        output.println("os.arch=" + System.getProperty("os.arch"));
        output.println("java.version=" + System.getProperty("java.version"));
        output.println("java.library.path=" + System.getProperty("java.library.path"));
        output.println("vectorforge.native.library="
                + valueOrUnset(System.getProperty("vectorforge.native.library")));
        output.println("vectorforge.native.library.dir="
                + valueOrUnset(System.getProperty("vectorforge.native.library.dir")));
        try {
            boolean cudaCompiled = NativeBindings.isCudaCompiled();
            boolean cuvsCompiled = NativeBindings.isCuvsCompiled();
            output.println("native.loaded=true");
            output.println("cuda.compiled=" + cudaCompiled);
            output.println("cuda.device.count="
                    + (cudaCompiled ? NativeBindings.getCudaDeviceCount() : 0));
            output.println("cuvs.compiled=" + cuvsCompiled);
            output.println("cuvs.version="
                    + (cuvsCompiled ? NativeBindings.getCuvsVersion() : "unavailable"));
            return true;
        } catch (NativeInteropException | UnsatisfiedLinkError error) {
            output.println("native.loaded=false");
            output.println("native.error=" + oneLine(error.getMessage()));
            output.println("remediation=Build with -Pnative/-Pcuda/-Pcuvs and configure "
                    + "vectorforge.native.library.dir plus PATH or LD_LIBRARY_PATH for dependencies.");
            return false;
        }
    }

    private static String valueOrUnset(String value) {
        return value == null || value.isBlank() ? "<unset>" : value;
    }

    private static String oneLine(String value) {
        return value == null ? errorName() : value.replace('\r', ' ').replace('\n', ' ');
    }

    private static String errorName() {
        return "native library load failed";
    }
}

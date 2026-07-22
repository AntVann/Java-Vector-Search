package com.vectorforge.nativeindex;

/**
 * Unchecked exception used for native-library loading failures or unexpected JNI-side errors.
 */
public final class NativeInteropException extends RuntimeException {

    public NativeInteropException(String message) {
        super(message);
    }

    public NativeInteropException(String message, Throwable cause) {
        super(message, cause);
    }
}


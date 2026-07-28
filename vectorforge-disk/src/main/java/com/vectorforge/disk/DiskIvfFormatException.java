package com.vectorforge.disk;

import java.io.IOException;
import java.nio.file.Path;

public final class DiskIvfFormatException extends IOException {
    public DiskIvfFormatException(Path path, String message) {
        super(path + ": " + message);
    }
}

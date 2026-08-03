package com.vectorforge.nativeindex;

import java.lang.ref.Cleaner;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongConsumer;

/**
 * Last-resort cleanup for an opaque native handle.
 *
 * <p>Explicit owners should take and destroy their handle themselves so teardown failures can be
 * reported. The cleaner path is necessarily best-effort and suppresses teardown failures.
 */
public final class NativeHandleGuard {

    private static final Cleaner CLEANER = Cleaner.create();

    private final CleanupAction action;
    private final Cleaner.Cleanable cleanable;

    private NativeHandleGuard(Object owner, LongConsumer destroyer) {
        action = new CleanupAction(destroyer);
        cleanable = CLEANER.register(owner, action);
    }

    public static NativeHandleGuard register(Object owner, LongConsumer destroyer) {
        return new NativeHandleGuard(
                Objects.requireNonNull(owner, "owner must not be null"),
                Objects.requireNonNull(destroyer, "destroyer must not be null"));
    }

    public long replace(long handle) {
        if (handle <= 0) {
            throw new IllegalArgumentException("native handle must be positive");
        }
        return action.handle.getAndSet(handle);
    }

    public long take() {
        return action.handle.getAndSet(0L);
    }

    public void clean() {
        cleanable.clean();
    }

    private static final class CleanupAction implements Runnable {
        private final AtomicLong handle = new AtomicLong();
        private final LongConsumer destroyer;

        private CleanupAction(LongConsumer destroyer) {
            this.destroyer = destroyer;
        }

        @Override
        public void run() {
            long current = handle.getAndSet(0L);
            if (current == 0L) {
                return;
            }
            try {
                destroyer.accept(current);
            } catch (Throwable ignored) {
                // Cleaner actions cannot report failures to the abandoned owner.
            }
        }
    }
}

package com.vectorforge.nativeindex;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class NativeHandleGuardTest {

    @Test
    void cleanerDestroysLiveHandleExactlyOnce() {
        List<Long> destroyed = new ArrayList<>();
        Object owner = new Object();
        NativeHandleGuard guard = NativeHandleGuard.register(owner, destroyed::add);

        assertEquals(0L, guard.replace(11L));
        guard.clean();
        guard.clean();

        assertEquals(List.of(11L), destroyed);
    }

    @Test
    void explicitTakePreventsCleanerDestruction() {
        List<Long> destroyed = new ArrayList<>();
        Object owner = new Object();
        NativeHandleGuard guard = NativeHandleGuard.register(owner, destroyed::add);

        assertEquals(0L, guard.replace(11L));
        assertEquals(11L, guard.replace(22L));
        assertEquals(22L, guard.take());
        guard.clean();

        assertEquals(List.of(), destroyed);
    }

    @Test
    void cleanerSuppressesDestroyerFailure() {
        Object owner = new Object();
        NativeHandleGuard guard = NativeHandleGuard.register(owner, ignored -> {
            throw new IllegalStateException("cleanup failed");
        });
        guard.replace(11L);

        guard.clean();
    }

    @Test
    void rejectsInvalidHandle() {
        NativeHandleGuard guard = NativeHandleGuard.register(new Object(), ignored -> {
        });
        assertThrows(IllegalArgumentException.class, () -> guard.replace(0L));
        guard.clean();
    }
}

# Contributing to VectorForge

## Prerequisites

- JDK 21
- Maven 3.9 or newer
- Git
- Optional native build: CMake 3.20 or newer and a C++17 compiler
- Optional GPU builds: the requirements documented in `docs/cuda-backend.md`
  and `docs/cuvs-integration-plan.md`

Do not make CUDA or cuVS a requirement of the default reactor.

## Change Guidelines

- Preserve module boundaries and keep optional dependencies profile-gated.
- Validate dimensions, counts, buffer sizes, IDs, and finite vector values before
  expensive allocation or JNI calls.
- Use try-with-resources for every `VectorIndex` that owns native resources.
- Do not guess external APIs or claim performance without a recorded run.
- Use fixed seeds and keep data generation outside timed benchmark regions.
- Add tests for lifecycle misuse, failure cleanup, deterministic ordering, and
  backend parity when changing search behavior.

## Verification

Run the largest applicable subset:

```text
mvn clean verify
mvn clean verify -Pnative
mvn clean verify -Pcuda
mvn clean verify -Pcuvs
git diff --check
```

The default command must pass without CMake, CUDA, cuVS, or GPU hardware.
Profile-enabled GPU tests must report executed test counts; an all-skipped run
does not validate that profile.

GitHub-hosted CI covers Windows and Linux default/native builds. Actual GPU
execution is manual-only and requires a maintained self-hosted runner with the
`nvidia` label. The cuVS job additionally requires the `cuvs-26-06` label and a
runner-defined `CUVS_ENV_PREFIX`; it does not install or bundle cuVS libraries.

For benchmark changes, package the benchmark module, run the validation script,
retain machine-readable output, and regenerate Markdown summaries from that
output. Record the exact commit, dirty state, seed, JVM, OS, compiler, CUDA/cuVS,
and hardware information.

## Review Checklist

- Public API behavior and exceptions are documented.
- Cleanup is idempotent and safe on partial failure.
- Optional backends fail clearly and do not silently fall back.
- Tests cover the change on CPU and each available optional backend.
- Documentation labels experimental and unavailable capabilities accurately.

## Native diagnostics

After building an optional native profile:

```powershell
$env:JAVA_TOOL_OPTIONS="-Dvectorforge.native.library.dir=vectorforge-native/target/native-lib"
java -cp vectorforge-native/target/classes com.vectorforge.nativeindex.NativeEnvironmentReport
```

```bash
JAVA_TOOL_OPTIONS=-Dvectorforge.native.library.dir=vectorforge-native/target/native-lib \
  java -cp vectorforge-native/target/classes \
  com.vectorforge.nativeindex.NativeEnvironmentReport
```

The report identifies JNI loading, compiled GPU features, CUDA device count,
and cuVS version, or prints attempted locations and loader-path remediation.

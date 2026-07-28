# Limitations

VectorForge is a research and interview project, not a production database or
hosted service. Results describe only the documented code, hardware, inputs, and
commands.

## Correctness and API

- Exact backends reject non-finite vector values, but finite inputs can still
  overflow floating-point accumulation at extreme magnitudes.
- The shared API is build/search oriented. It has no incremental mutation,
  persistence, transactions, authentication, authorization, or network protocol.
- Implementations have different supported metrics. Callers must not assume a
  metric is portable without checking backend documentation.

## Native and GPU

- Native, CUDA, and cuVS resources require deterministic `close()` via
  try-with-resources. There is currently no `Cleaner` fallback for abandoned
  wrappers.
- The custom CUDA backend is exact dot product and performs host-side top-k. It
  is educational, not a tuned ANN implementation.
- cuVS is verified only against version 26.06 on Linux/WSL2. Runtime cuVS,
  RAPIDS, and CUDA shared libraries must be discoverable through the platform
  loader path.
- GPU tests skip when no usable device exists. A CPU-only success is not evidence
  that GPU code executed.

## Lucene and Disk

- Lucene integration rebuilds from live documents at explicit refresh points;
  it does not integrate with Lucene segment internals.
- Disk IVF uses immutable generations and a single-process consistency model.
  Concurrent writers, distributed operation, online compaction, replication,
  and crash-safe database semantics are unsupported.
- Disk-IVF scale is limited to tested datasets and hardware. No broader scale
  claim is made.

## Benchmarks

- Checked results are machine-specific observations, not general performance
  claims.
- Smoke harnesses do not provide JMH isolation, confidence intervals, randomized
  backend order, or control of thermal/power state.
- The resident-backend end-to-end suite and disk-IVF harness are separate.
- Native and GPU memory measurements are best-effort and platform dependent.

## Operational Gaps

- There is no compatibility guarantee before a stable release.
- There is no CI runner with NVIDIA hardware in this repository.
- Libraries intentionally do not install a logging framework; demos and
  benchmark CLIs write structured or human-readable output to standard streams.

#!/usr/bin/env python3
"""Convert VectorForge end-to-end benchmark JSON Lines into Markdown tables."""

import argparse
import json
from pathlib import Path


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = next((record for record in records if record["record_type"] == "metadata"), {})
    results = [record for record in records if record["record_type"] == "result"]
    skips = [record for record in records if record["record_type"] == "skip"]
    errors = [record for record in records if record["record_type"] == "error"]

    lines = [
        "# VectorForge End-to-End Benchmark",
        "",
        "Generated from machine-readable JSON Lines. These measurements are local observations, not generalized performance claims.",
        "",
        "## System",
        "",
        "| Attribute | Value |",
        "| --- | --- |",
    ]
    for key in (
        "timestamp_utc", "mode", "os_name", "os_version", "os_arch", "jvm",
        "java_version", "processors", "cpu", "max_heap_bytes", "compiler",
        "compiler_version", "cuda_toolkit_root", "cuda_version",
        "cuda_device_count", "cuvs_version", "gpu", "git_sha", "git_dirty",
        "maven_version", "cmake_version",
    ):
        lines.append(f"| `{key}` | {fmt(metadata.get(key))} |")

    lines += [
        "",
        "## Results",
        "",
        "| Backend | Vectors | Dims | Batch | k | Metric | Build ms | Batch avg ms | p50 ms | p95 ms | p99 ms | QPS | Recall@k | Heap delta | RSS delta | GPU delta | CUDA H2D / kernel / D2H ms |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in results:
        cuda = " / ".join(fmt(record.get(key)) for key in (
            "cuda_h2d_avg_ms", "cuda_kernel_avg_ms", "cuda_d2h_avg_ms"))
        lines.append(
            f"| {record['backend']} | {record['vectors']} | {record['dimensions']} | "
            f"{record['batch_size']} | {record['k']} | {record['metric']} | "
            f"{fmt(record['build_ms'])} | {fmt(record['end_to_end_batch_avg_ms'])} | "
            f"{fmt(record['end_to_end_batch_p50_ms'])} | {fmt(record['end_to_end_batch_p95_ms'])} | "
            f"{fmt(record['end_to_end_batch_p99_ms'])} | {fmt(record['qps'])} | "
            f"{fmt(record['recall_at_k'], 6)} | {fmt(record['heap_delta_bytes'])} | "
            f"{fmt(record['process_rss_delta_bytes'])} | {fmt(record['gpu_memory_delta_bytes'])} | {cuda} |"
        )

    if skips:
        lines += [
            "",
            "## Skips",
            "",
            "| Backend | Vectors | Dims | Batch | k | Metric | Reason |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for record in skips:
            lines.append(
                f"| {record['backend']} | {record['vectors']} | {record['dimensions']} | "
                f"{record['batch_size']} | {record['k']} | {record['metric']} | {record['reason']} |"
            )

    if errors:
        lines += [
            "",
            "## Errors",
            "",
            "| Backend | Vectors | Dims | Batch | k | Metric | Reason |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for record in errors:
            lines.append(
                f"| {record['backend']} | {record['vectors']} | {record['dimensions']} | "
                f"{record['batch_size']} | {record['k']} | {record['metric']} | {fmt(record['reason'])} |"
            )

    lines += [
        "",
        "## Caveats",
        "",
        "- Build time is separate from measured query batches.",
        "- QPS is derived from completed queries divided by summed end-to-end batch time.",
        "- Heap, process RSS, and `nvidia-smi` deltas are approximate snapshots and may be unavailable.",
        "- CUDA phase timings are emitted only by the custom CUDA backend.",
        f"- {metadata.get('percentile_note', 'Tail percentiles depend on the configured sample count.')}",
        "- Review the JSON Lines artifact for raw samples and explicit skip records.",
        "",
    ]
    rendered = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

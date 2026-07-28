#!/usr/bin/env bash
set -euo pipefail

jar_path="${1:-vectorforge-benchmarks/target/vectorforge-benchmarks.jar}"
native_dir="${2:-vectorforge-native/target/native-lib}"
main_class="com.vectorforge.benchmarks.EndToEndBenchmarkRunner"
python_bin="${PYTHON:-python3}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python 3 executable not found: $python_bin (override with PYTHON=/path/to/python3)" >&2
  exit 1
fi

if java -cp "$jar_path" "$main_class" --unknown value >/dev/null 2>&1; then
  echo "expected unknown-option validation to fail" >&2
  exit 1
fi

if java -cp "$jar_path" "$main_class" --iterations 0 >/dev/null 2>&1; then
  echo "expected zero-iteration validation to fail" >&2
  exit 1
fi

error_output="$(mktemp)"
trap 'rm -f "$error_output"' EXIT
if java -Dvectorforge.native.library.dir="$native_dir" -cp "$jar_path" "$main_class" \
    --mode smoke --metrics DOT_PRODUCT --backends cpu \
    --force-error-backend cpu --output "$error_output" >/dev/null 2>&1; then
  echo "expected forced detected-backend error to produce a nonzero exit" >&2
  exit 1
fi

"$python_bin" - "$error_output" <<'PY'
import json
import sys

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
errors = [record for record in records if record["record_type"] == "error"]
if len(errors) != 1 or errors[0]["backend"] != "cpu":
    raise SystemExit("forced error did not produce one CPU error record")
PY

echo "End-to-end benchmark validation checks passed."

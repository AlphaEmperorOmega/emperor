#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash download_logs.sh [log_entry] [output_zip]

Run this from the project directory. Create a zip archive for downloading data
from ./logs, or from one folder inside ./logs. A selected folder is preserved
as the archive prefix so imports restore it under the same experiment path.

Arguments:
  log_entry    Optional folder inside ./logs to archive. Defaults to all
               contents of ./logs. Accepts either my_experiment or
               logs/my_experiment.
  output_zip   Zip path to create. Defaults to ./<folder>_<timestamp>.zip.

Examples:
  bash download_logs.sh
  bash download_logs.sh my_experiment
  bash download_logs.sh logs/my_experiment
  bash download_logs.sh logs emperor_logs.zip
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -f "experiment.sh" ] || [ ! -f "pyproject.toml" ] || [ ! -d "emperor" ]; then
  echo "Error: run this script from the project root." >&2
  echo >&2
  usage >&2
  exit 1
fi

LOG_ROOT="logs"

if [ ! -d "$LOG_ROOT" ]; then
  echo "Error: ./logs not found. Run this script from the project directory." >&2
  echo >&2
  usage >&2
  exit 1
fi

LOG_ENTRY="${1:-}"
if [ -z "$LOG_ENTRY" ] || [ "$LOG_ENTRY" = "$LOG_ROOT" ] || [ "$LOG_ENTRY" = "$LOG_ROOT/" ]; then
  SOURCE_PATH="$LOG_ROOT"
else
  case "$LOG_ENTRY" in
    /*|..|../*|*/../*|*/..)
      echo "Error: log entry must be a folder inside ./logs: $LOG_ENTRY" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
  if [[ "$LOG_ENTRY" == "$LOG_ROOT/"* ]]; then
    LOG_ENTRY="${LOG_ENTRY#"$LOG_ROOT"/}"
  fi
  SOURCE_PATH="$LOG_ROOT/$LOG_ENTRY"
fi

if [ ! -d "$SOURCE_PATH" ]; then
  echo "Error: log folder not found under ./logs: $SOURCE_PATH" >&2
  echo >&2
  usage >&2
  exit 1
fi

if [ -n "${2:-}" ]; then
  OUTPUT_PATH="$2"
else
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  ARCHIVE_NAME="$(basename "${SOURCE_PATH%/}")"
  ARCHIVE_NAME="${ARCHIVE_NAME//[^[:alnum:]._-]/_}"
  if [ -z "$ARCHIVE_NAME" ]; then
    ARCHIVE_NAME="logs"
  fi
  OUTPUT_PATH="${ARCHIVE_NAME}_${TIMESTAMP}.zip"
fi

python3 - "$SOURCE_PATH" "$OUTPUT_PATH" "$LOG_ROOT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).expanduser()
log_root = Path(sys.argv[3]).resolve()
if not output.is_absolute():
    output = (Path.cwd() / output).resolve()
else:
    output = output.resolve()

if not source.is_dir():
    raise SystemExit(f"Error: log folder not found: {source}")
try:
    source.relative_to(log_root)
except ValueError:
    raise SystemExit(f"Error: log folder must resolve inside {log_root}")

output.parent.mkdir(parents=True, exist_ok=True)
file_count = 0

with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
    for path in sorted(source.rglob("*")):
        resolved_path = path.resolve()
        if resolved_path == output:
            continue
        archive.write(path, resolved_path.relative_to(log_root).as_posix())
        if path.is_file():
            file_count += 1

size_bytes = output.stat().st_size
size_mb = size_bytes / (1024 * 1024)
print(f"Created archive: {output}")
print(f"Included files: {file_count}")
print(f"Archive size: {size_mb:.2f} MiB")
PY

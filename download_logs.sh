#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-${EMPEROR_PYTHON:-}}"
if [ -z "$PYTHON_BIN" ] && [ -x "$REPOSITORY_ROOT/torchenv/bin/python" ]; then
  PYTHON_BIN="$REPOSITORY_ROOT/torchenv/bin/python"
fi
exec "${PYTHON_BIN:-python3}" -m models.project_cli logs:archive "$@"

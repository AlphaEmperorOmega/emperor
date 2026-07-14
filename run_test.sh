#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. New automation should use: mise run test -- ...
REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
cd "$REPOSITORY_ROOT"
exec "$PYTHON_BIN" -m models.project_cli test "$@"

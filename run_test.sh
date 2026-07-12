#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$REPOSITORY_ROOT/tests"
PYTHON_BIN="${PYTHON:-python3}"

# The checkout root must not make production packages importable by accident.
# The tests directory is the only source path added here; Emperor itself must be
# provided by the active environment's editable or regular installation.
export PYTHONSAFEPATH=1
export PYTHONPATH="$TEST_ROOT"

if ! "$PYTHON_BIN" -P -c \
  'from importlib.metadata import distribution; distribution("emperor")' \
  >/dev/null 2>&1; then
  echo "Error: install Emperor in the active environment before running tests." >&2
  echo "Run: python -m pip install --no-deps -e ." >&2
  exit 1
fi

if [ -n "${1:-}" ]; then
  FILE="$1"
  FILE="${FILE%.py}"
  TEST_PATH="$TEST_ROOT/unit/test_${FILE}.py"

  if [ ! -f "$TEST_PATH" ]; then
    echo "Error: $TEST_PATH not found" >&2
    exit 1
  fi

  MODULE="unit.test_${FILE}"

  if [ -n "${2:-}" ]; then
    CLASS="$2"
    if [ -n "${3:-}" ]; then
      FUNC="$3"
      "$PYTHON_BIN" -P -m unittest -f "$MODULE.$CLASS.$FUNC"
    else
      "$PYTHON_BIN" -P -m unittest -f "$MODULE.$CLASS"
    fi
  else
    "$PYTHON_BIN" -P -m unittest -f "$MODULE"
  fi
else
  "$PYTHON_BIN" -P -m unittest discover -f -s "$TEST_ROOT" -t "$TEST_ROOT"
fi

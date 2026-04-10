#!/usr/bin/env bash

DOCS_DIR="docs"

if [ -n "$1" ]; then
  FILE="$1"
  FILE="${FILE%.py}"
  TEST_PATH="$DOCS_DIR/test_${FILE}.py"

  if [ ! -f "$TEST_PATH" ]; then
    echo "Error: $TEST_PATH not found"
  fi

  MODULE="${DOCS_DIR}.test_${FILE}"

  if [ -n "$2" ]; then
    CLASS="$2"
    if [ -n "$3" ]; then
      FUNC="$3"
      python3 -m unittest -f "$MODULE.$CLASS.$FUNC"
    else
      python3 -m unittest -f "$MODULE.$CLASS"
    fi
  else
    python3 -m unittest -f "$MODULE"
  fi
else
  python3 -m unittest discover -f "$DOCS_DIR"
fi

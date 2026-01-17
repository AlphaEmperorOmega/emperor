#!/usr/bin/env bash

DOCS_DIR="docs"

if [ -n "$1" ]; then
    TEST_PATH="$DOCS_DIR/test_$1"
    if [ ! -f "$TEST_PATH" ]; then
        echo "Error: $TEST_PATH not found"
        exit 1
    fi
    python3 -m unittest -f "$TEST_PATH"
else
    python3 -m unittest discover -f "$DOCS_DIR"
fi

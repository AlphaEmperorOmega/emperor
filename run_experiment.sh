#!/usr/bin/env bash

MODELS_DIR="models"

list_experiments() {
  echo "Available experiments:"
  for f in "$MODELS_DIR"/*/; do
    name="$(basename "$f")"
    [ "$name" = "__pycache__" ] && continue
    echo "  $name"
  done
}

if [ -z "$1" ] || [ "$1" = "--list" ]; then
  echo "Usage: $0 <experiment> [options]"
  echo ""
  list_experiments
elif [ ! -d "$MODELS_DIR/$1" ]; then
  echo "Error: $MODELS_DIR/$1/ not found"
  echo ""
  list_experiments
else
  EXPERIMENT="$1"
  shift
  python3 -m "models.$EXPERIMENT" "$@"
fi

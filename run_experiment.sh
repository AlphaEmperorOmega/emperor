#!/usr/bin/env bash

MODELS_DIR="models"

list_experiments() {
  echo "Available experiments:"
  for f in "$MODELS_DIR"/*.py; do
    name="$(basename "$f" .py)"
    [ "$name" = "__init__" ] || [ "$name" = "parser" ] && continue
    echo "  $name"
  done
}

if [ -z "$1" ] || [ "$1" = "--list" ]; then
  echo "Usage: $0 <experiment> [--name <config_name>]"
  echo ""
  list_experiments
elif [ ! -f "$MODELS_DIR/$1.py" ]; then
  echo "Error: $MODELS_DIR/$1.py not found"
  echo ""
  list_experiments
else
  EXPERIMENT="$1"
  shift
  python3 -m "models.$EXPERIMENT" "$@"
fi

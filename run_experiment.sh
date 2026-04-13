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

list_options() {
  python3 -c "from models.$1 import ExperimentOptions; [print(f'  {n}') for n in ExperimentOptions.names()]"
}

if [ -z "$1" ] || [ "$1" = "--list" ]; then
  echo "Usage: $0 <experiment> [options]"
  echo ""
  list_experiments
elif [ ! -d "$MODELS_DIR/$1" ]; then
  echo "Error: $MODELS_DIR/$1/ not found"
  echo ""
  list_experiments
elif [ "$2" = "--list-options" ]; then
  echo "Available options for $1:"
  list_options "$1"
else
  EXPERIMENT="$1"
  shift
  python3 -m "models.$EXPERIMENT" "$@"
fi

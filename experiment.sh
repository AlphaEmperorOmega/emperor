#!/usr/bin/env bash

MODELS_DIR="models"
COMMAND="source experiment.sh"

list_experiments() {
  echo "Available experiments:"
  for f in "$MODELS_DIR"/*/; do
    name="$(basename "$f")"
    [ "$name" = "__pycache__" ] && continue
    echo "  $name"
  done
}

list_flags() {
  echo "Available flags:"
  echo ""
  echo "  --list                  ## Show available experiments and flags"
  echo "                          $COMMAND --list"
  echo ""
  echo "  --list-options          ## Show available options for an experiment"
  echo "                          $COMMAND <experiment> --list-options"
  echo ""
  echo "  --option <option>       ## Run one experiment option"
  echo "                          $COMMAND <experiment> --option <option>"
  echo ""
  echo "  --all-options           ## Run all experiment options sequentially"
  echo "                          $COMMAND <experiment> --all-options"
  echo ""
  echo "  --grid-search           ## Run grid search for the selected option"
  echo "                          $COMMAND <experiment> --option <option> --grid-search"
  echo ""
  echo "  --random-search <n>     ## Run random search with <n> sampled combinations"
  echo "                          $COMMAND <experiment> --option <option> --random-search <n>"
  echo ""
  echo "  --logdir <dir>          ## Store experiment logs in <dir>"
  echo "                          $COMMAND <experiment> --option <option> --logdir <dir>"
}

list_options() {
  python3 -c "from models.$1 import ExperimentOptions; [print(f'  {n}') for n in ExperimentOptions.names()]"
}

if [ -z "$1" ]; then
  echo "Usage: $COMMAND <experiment> [options]"
  echo ""
  list_flags
elif [ "$1" = "--list" ]; then
  echo "Usage: $COMMAND <experiment> [options]"
  echo ""
  list_experiments
  echo ""
  list_flags
elif [ ! -d "$MODELS_DIR/$1" ]; then
  echo "Error: $MODELS_DIR/$1/ not found"
  echo ""
  echo "Run '$COMMAND --list' to see available experiments."
  echo "Run '$COMMAND' to see available flags."
elif [ "$2" = "--list-options" ]; then
  echo "Available options for $1:"
  list_options "$1"
else
  EXPERIMENT="$1"
  shift
  python3 -m "models.$EXPERIMENT" "$@"
fi

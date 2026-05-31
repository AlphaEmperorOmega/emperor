#!/usr/bin/env bash

MODELS_DIR="models"
COMMAND="source experiment.sh"
MODEL_ARG="<model>"
PRESET_ARG="<preset>"
OPTS_ARG="[options]"

list_models() {
  echo "Available models:"
  for f in "$MODELS_DIR"/*/; do
    name="$(basename "$f")"
    [ "$name" = "__pycache__" ] && continue
    echo "  $name"
  done
}

list_flags() {
  echo "Available flags:"
  echo ""
  echo "  --list-models           ## Show available models"
  echo "                          $COMMAND --list-models"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --list-presets          ## Show available presets for a model"
  echo "                          $COMMAND $MODEL_ARG --list-presets"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --list-config           ## Show overridable config flags and defaults"
  echo "                          $COMMAND $MODEL_ARG --list-config"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --preset $PRESET_ARG       ## Run one model preset"
  echo "                          $COMMAND $MODEL_ARG --preset $PRESET_ARG"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --print-model           ## Print model structure instead of running training"
  echo "                          $COMMAND $MODEL_ARG --preset $PRESET_ARG --print-model"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --all-presets           ## Run all model presets sequentially"
  echo "                          $COMMAND $MODEL_ARG --all-presets"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --grid-search           ## Run grid search for the selected option"
  echo "                          $COMMAND $MODEL_ARG --preset $PRESET_ARG --grid-search"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --random-search <n>     ## Run random search with <n> sampled combinations"
  echo "                          $COMMAND $MODEL_ARG --preset $PRESET_ARG --random-search <n>"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --search-keys <keys>    ## Restrict sweep to named config-file search axes"
  echo "                          $COMMAND $MODEL_ARG --preset $PRESET_ARG --grid-search --search-keys HIDDEN_DIM STACK_NUM_LAYERS"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --search-set <k=v,..>   ## Sweep command-line values for one search axis"
  echo "                          $COMMAND $MODEL_ARG --preset $PRESET_ARG --grid-search --search-set hidden_dim=64,128"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --config <flags...>     ## Override model config values without editing config.py"
  echo "                          $COMMAND $MODEL_ARG --preset $PRESET_ARG --config --num-epochs 30 --callback-early-stopping-patience 0"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --logdir <dir>          ## Store model logs in <dir>"
  echo "                          $COMMAND $MODEL_ARG --preset $PRESET_ARG --logdir <dir>"
}

show_no_argument_usage_help() {
  echo "Usage: $COMMAND $MODEL_ARG $OPTS_ARG"
  echo ""
  list_flags
}

show_model_list_usage() {
  echo "Usage: $COMMAND $MODEL_ARG $OPTS_ARG"
  echo ""
  list_models
}

show_unknown_model_error() {
  echo "Error: $MODELS_DIR/$1/ not found"
  echo ""
  echo "Run '$COMMAND --list-models' to see available models."
  echo "Run '$COMMAND' to see available flags."
}

show_model_preset_options() {
  list_presets "$1"
}

show_model_config_options() {
  python3 -m models.config_overrides "$1"
}

show_all_presets_print_model_error() {
  echo "Error: --print-model requires --preset $PRESET_ARG and cannot be used with --all-presets."
  echo ""
  echo "Run '$COMMAND $1 --preset $PRESET_ARG --print-model' to inspect one preset."
}

list_presets() {
  python3 -m models.config_overrides "$1" --presets
}

parse_model_command_args() {
  local -n print_model_ref="$1"
  local -n all_presets_ref="$2"
  local -n args_ref="$3"
  shift 3

  print_model_ref=false
  all_presets_ref=false
  args_ref=()
  for arg in "$@"; do
    if [ "$arg" = "--print-model" ]; then
      print_model_ref=true
    elif [ "$arg" = "--all-presets" ]; then
      all_presets_ref=true
      args_ref+=("$arg")
    else
      args_ref+=("$arg")
    fi
  done
}

run_model_command() {
  local model="$1"
  shift

  local print_model=false
  local all_presets=false
  local args=()
  parse_model_command_args print_model all_presets args "$@"

  if [ "$print_model" = true ] && [ "$all_presets" = true ]; then
    show_all_presets_print_model_error "$model"
  elif [ "$print_model" = true ]; then
    python3 -m viewer.backend.cli --model "$model" "${args[@]}"
  else
    python3 -m "models.$model" "${args[@]}"
  fi
}

if [ -z "$1" ]; then
  show_no_argument_usage_help
elif [ "$1" = "--list-models" ]; then
  show_model_list_usage
elif [ ! -d "$MODELS_DIR/$1" ]; then
  show_unknown_model_error "$1"
elif [ "$2" = "--list-presets" ]; then
  show_model_preset_options "$1"
elif [ "$2" = "--list-config" ]; then
  show_model_config_options "$1"
elif [ "$2" = "--preset" ] && [ -z "$3" ]; then
  show_model_preset_options "$1"
else
  run_model_command "$@"
fi

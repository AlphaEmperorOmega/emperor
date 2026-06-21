#!/usr/bin/env bash

MODELS_DIR="models"
COMMAND="source experiment.sh"
MODEL_TYPE_ARG="<type>"
MODEL_NAME_ARG="<name>"
MODEL_SELECTOR_ARG="--model-type $MODEL_TYPE_ARG --model $MODEL_NAME_ARG"
PRESET_ARG="<preset>"
OPTS_ARG="[options]"

list_model_types() {
  local model_types
  model_types="$(python3 -m models.catalog --list-types)" || return 1

  echo "Available model types:"
  while IFS= read -r model_type; do
    [ -n "$model_type" ] && echo "  --model-type $model_type"
  done <<< "$model_types"
}

list_models() {
  local model_type="$1"
  local models
  if [ -n "$model_type" ]; then
    models="$(python3 -m models.catalog --list --model-type "$model_type")" || return 1
    echo "Available models for --model-type $model_type:"
  else
    models="$(python3 -m models.catalog --list)" || return 1
    echo "Available models:"
  fi

  while IFS= read -r model; do
    [ -n "$model" ] && echo "  $model"
  done <<< "$models"
}

list_datasets() {
  local model_type="$1"
  local model="$2"
  local datasets
  datasets="$(python3 -m models.config_overrides "$model_type/$model" --datasets)" || return 1

  echo "Available datasets for --model-type $model_type --model $model:"
  while IFS= read -r dataset; do
    [ -n "$dataset" ] && echo "  --datasets $dataset"
  done <<< "$datasets"
}

list_monitors() {
  local model_type="$1"
  local model="$2"
  local monitors
  monitors="$(python3 -m models.config_overrides "$model_type/$model" --monitors)" || return 1

  echo "Available monitors for --model-type $model_type --model $model:"
  while IFS= read -r monitor; do
    [ -n "$monitor" ] && echo "  --monitors $monitor"
  done <<< "$monitors"
}

model_module() {
  python3 -m models.catalog --module --model-type "$1" --model "$2"
}

is_known_model() {
  model_module "$1" "$2" > /dev/null
}

is_known_model_type() {
  python3 -m models.catalog --list --model-type "$1" > /dev/null 2>&1
}

list_flags() {
  echo "Available flags:"
  echo ""
  echo "  --list-model-types      ## Show available model categories"
  echo "                          $COMMAND --list-model-types"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --list-models           ## Show available models"
  echo "                          $COMMAND --list-models"
  echo "                          $COMMAND --model-type linears --list-models"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --list-datasets         ## Show available datasets for a model"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --list-datasets"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --list-monitors         ## Show available monitor callbacks for a model"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --list-monitors"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --model-type $MODEL_TYPE_ARG     ## Select a model category"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --list-presets"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --model $MODEL_NAME_ARG          ## Select a model package name"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --list-presets          ## Show available presets for a model"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --list-presets"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --list-config           ## Show overridable config flags and defaults"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --list-config"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --preset $PRESET_ARG       ## Run one model preset"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --presets p1 p2        ## Run selected model presets sequentially"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --presets baseline gating --grid-search"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --print-model           ## Print model structure instead of running training"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG --print-model"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --monitors <names...>   ## Enable monitor callbacks for a training run"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG --monitors linear halting"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --all-presets           ## Run all model presets sequentially"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --all-presets"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --grid-search           ## Run grid search for the selected option"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG --grid-search"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --random-search <n>     ## Run random search with <n> sampled combinations"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG --random-search <n>"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --search-keys <keys>    ## Restrict sweep to named config-file search axes"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG --grid-search --search-keys HIDDEN_DIM STACK_NUM_LAYERS"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --search-set <k=v,..>   ## Sweep command-line values for one search axis"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG --grid-search --search-set hidden_dim=64,128"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --config <flags...>     ## Override model config values without editing config.py"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG --config --num-epochs 30 --callback-early-stopping-patience 0"
  echo "-----------------------------------------------------------------------------------------------"
  echo "  --logdir <dir>          ## Store model logs in <dir>"
  echo "                          $COMMAND $MODEL_SELECTOR_ARG --preset $PRESET_ARG --logdir <dir>"
}

show_no_argument_usage_help() {
  echo "Usage: $COMMAND $MODEL_SELECTOR_ARG $OPTS_ARG"
  echo ""
  list_flags
}

show_model_list_usage() {
  local model_type="$1"
  if [ -n "$model_type" ]; then
    echo "Usage: $COMMAND --model-type $model_type --model $MODEL_NAME_ARG $OPTS_ARG"
  else
    echo "Usage: $COMMAND $MODEL_SELECTOR_ARG $OPTS_ARG"
  fi
  echo ""
  list_models "$model_type"
}

show_model_type_list_usage() {
  echo "Usage: $COMMAND $MODEL_SELECTOR_ARG $OPTS_ARG"
  echo ""
  list_model_types
}

show_dataset_list_usage() {
  echo "Usage: $COMMAND --model-type $1 --model $2 $OPTS_ARG"
  echo ""
  list_datasets "$1" "$2"
}

show_monitor_list_usage() {
  echo "Usage: $COMMAND --model-type $1 --model $2 $OPTS_ARG"
  echo ""
  list_monitors "$1" "$2"
}

show_unknown_model_error() {
  echo "Error: unknown model '--model-type $1 --model $2'"
  echo ""
  echo "Run '$COMMAND --list-models' to see available models."
  echo "Run '$COMMAND' to see available flags."
}

show_unknown_model_type_error() {
  echo "Error: unknown model type '--model-type $1'."
  echo ""
  echo "Run '$COMMAND --list-model-types' to see available model types."
}

show_missing_model_selector_error() {
  echo "Error: model commands require --model-type $MODEL_TYPE_ARG --model $MODEL_NAME_ARG."
  echo ""
  echo "Run '$COMMAND --list-models' to see available models."
  echo "Run '$COMMAND' to see available flags."
}

show_missing_flag_value_error() {
  echo "Error: $1 requires a value."
  echo ""
  echo "Run '$COMMAND' to see available flags."
}

show_positional_model_error() {
  local model_type="$MODEL_TYPE_ARG"
  local model="$MODEL_NAME_ARG"
  if [[ "$1" == */* ]]; then
    model_type="${1%%/*}"
    model="${1##*/}"
  fi
  echo "Positional model ids are no longer supported. Use --model-type $model_type --model $model ..."
}

show_model_preset_options() {
  list_presets "$1" "$2"
}

show_model_config_options() {
  python3 -m models.config_overrides "$1/$2"
}

show_multi_preset_print_model_error() {
  echo "Error: --print-model requires --preset $PRESET_ARG and cannot be used with --all-presets or --presets."
  echo ""
  echo "Run '$COMMAND --model-type $1 --model $2 --preset $PRESET_ARG --print-model' to inspect one preset."
}

show_monitor_print_model_error() {
  echo "Error: --monitors applies to training runs and cannot be used with --print-model."
  echo ""
  echo "Run '$COMMAND --model-type $1 --model $2 --preset $PRESET_ARG --print-model' without --monitors."
}

list_presets() {
  python3 -m models.config_overrides "$1/$2" --presets
}

parse_model_command_args() {
  local -n print_model_ref="$1"
  local -n all_presets_ref="$2"
  local -n selected_presets_ref="$3"
  local -n monitor_selection_ref="$4"
  local -n args_ref="$5"
  shift 5

  print_model_ref=false
  all_presets_ref=false
  selected_presets_ref=false
  monitor_selection_ref=false
  args_ref=()
  for arg in "$@"; do
    if [ "$arg" = "--print-model" ]; then
      print_model_ref=true
    elif [ "$arg" = "--monitors" ]; then
      monitor_selection_ref=true
      args_ref+=("$arg")
    elif [ "$arg" = "--all-presets" ]; then
      all_presets_ref=true
      args_ref+=("$arg")
    elif [ "$arg" = "--presets" ]; then
      selected_presets_ref=true
      args_ref+=("$arg")
    else
      args_ref+=("$arg")
    fi
  done
}

run_model_command() {
  local model_type="$1"
  local model="$2"
  shift 2

  local print_model=false
  local all_presets=false
  local selected_presets=false
  local monitor_selection=false
  local args=()
  parse_model_command_args print_model all_presets selected_presets monitor_selection args "$@"

  if [ "$print_model" = true ] && { [ "$all_presets" = true ] || [ "$selected_presets" = true ]; }; then
    show_multi_preset_print_model_error "$model_type" "$model"
    return 1
  elif [ "$print_model" = true ] && [ "$monitor_selection" = true ]; then
    show_monitor_print_model_error "$model_type" "$model"
    return 1
  elif [ "$print_model" = true ]; then
    python3 -m viewer.backend.cli --model-type "$model_type" --model "$model" "${args[@]}"
  else
    local module
    module="$(model_module "$model_type" "$model")" || return 1
    python3 -m "$module" "${args[@]}"
  fi
}

main() {
  if [ -z "$1" ]; then
    show_no_argument_usage_help
    return 0
  elif [[ "$1" != --* ]]; then
    show_positional_model_error "$1"
    return 1
  fi

  local model_type=""
  local model=""
  local list_model_types_requested=false
  local list_models_requested=false
  local list_datasets_requested=false
  local list_monitors_requested=false
  local args=()

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --list-model-types)
        list_model_types_requested=true
        shift
        ;;
      --list-models)
        list_models_requested=true
        shift
        ;;
      --list-datasets)
        list_datasets_requested=true
        shift
        ;;
      --list-monitors)
        list_monitors_requested=true
        shift
        ;;
      --model-type)
        if [ -z "$2" ]; then
          show_missing_flag_value_error "--model-type"
          return 1
        fi
        model_type="$2"
        shift 2
        ;;
      --model)
        if [ -z "$2" ]; then
          show_missing_flag_value_error "--model"
          return 1
        fi
        model="$2"
        shift 2
        ;;
      *)
        args+=("$1")
        shift
        ;;
    esac
  done

  if [ "$list_model_types_requested" = true ]; then
    show_model_type_list_usage
  elif [ "$list_models_requested" = true ]; then
    if [ -n "$model_type" ] && ! is_known_model_type "$model_type"; then
      show_unknown_model_type_error "$model_type"
      return 1
    fi
    show_model_list_usage "$model_type"
  elif [ "$list_datasets_requested" = true ]; then
    if [ -z "$model_type" ] || [ -z "$model" ]; then
      show_missing_model_selector_error
      return 1
    elif ! is_known_model "$model_type" "$model"; then
      show_unknown_model_error "$model_type" "$model"
      return 1
    fi
    show_dataset_list_usage "$model_type" "$model"
  elif [ "$list_monitors_requested" = true ]; then
    if [ -z "$model_type" ] || [ -z "$model" ]; then
      show_missing_model_selector_error
      return 1
    elif ! is_known_model "$model_type" "$model"; then
      show_unknown_model_error "$model_type" "$model"
      return 1
    fi
    show_monitor_list_usage "$model_type" "$model"
  elif [ -z "$model_type" ] || [ -z "$model" ]; then
    show_missing_model_selector_error
    return 1
  elif ! is_known_model "$model_type" "$model"; then
    show_unknown_model_error "$model_type" "$model"
    return 1
  elif [ "${args[0]}" = "--list-presets" ]; then
    show_model_preset_options "$model_type" "$model"
  elif [ "${args[0]}" = "--list-config" ]; then
    show_model_config_options "$model_type" "$model"
  elif [ "${args[0]}" = "--preset" ] && [ -z "${args[1]}" ]; then
    show_model_preset_options "$model_type" "$model"
  else
    run_model_command "$model_type" "$model" "${args[@]}"
  fi
}

main "$@"
status=$?
return "$status" 2> /dev/null || exit "$status"

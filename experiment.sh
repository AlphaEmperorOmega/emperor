#!/usr/bin/env bash

# Compatibility wrapper. New automation should use: mise run experiment -- ...
_emperor_python() {
  if [ "$(type -t python3)" = "function" ]; then
    python3 "$@"
    return $?
  fi
  if [ -n "${EMPEROR_PYTHON:-}" ]; then
    "$EMPEROR_PYTHON" "$@"
    return $?
  fi
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -x "$script_dir/torchenv/bin/python" ]; then
    "$script_dir/torchenv/bin/python" "$@"
    return $?
  fi
  command python3 "$@"
}

model_module() {
  _emperor_python -m models.catalog --module --model-type "$1" --model "$2"
}

# Retained only for callers that sourced the former script and invoked this
# helper directly. Argument parsing and normal execution live in the project Adapter.
run_model_command() {
  local model_type="$1"
  local model="$2"
  shift 2
  local print_model=false
  local shape_trace=""
  local inspection_flag_count=0
  local all_presets=false
  local selected_presets=false
  local monitor_selection=false
  local args=()
  local argument
  for argument in "$@"; do
    case "$argument" in
      --print-model)
        print_model=true
        inspection_flag_count=$((inspection_flag_count + 1))
        ;;
      --print-model-shapes)
        print_model=true
        shape_trace=outputs
        inspection_flag_count=$((inspection_flag_count + 1))
        ;;
      --print-model-tensor-shapes)
        print_model=true
        shape_trace=variables
        inspection_flag_count=$((inspection_flag_count + 1))
        ;;
      --all-presets) all_presets=true; args+=("$argument") ;;
      --presets) selected_presets=true; args+=("$argument") ;;
      --monitors) monitor_selection=true; args+=("$argument") ;;
      *) args+=("$argument") ;;
    esac
  done
  if [ "$inspection_flag_count" -gt 1 ]; then
    _emperor_python -m models.project_cli --model-type "$model_type" --model "$model" "$@"
    return $?
  fi
  if [ "$print_model" = true ] && { [ "$all_presets" = true ] || [ "$selected_presets" = true ]; }; then
    _emperor_python -m models.project_cli --model-type "$model_type" --model "$model" "$@"
    return $?
  fi
  if [ "$print_model" = true ] && [ "$monitor_selection" = true ]; then
    _emperor_python -m models.project_cli --model-type "$model_type" --model "$model" "$@"
    return $?
  fi
  if [ "$print_model" = true ]; then
    local shape_args=()
    if [ -n "$shape_trace" ]; then
      shape_args=(--shape-trace "$shape_trace")
    fi
    _emperor_python -m models.project_cli inspect --model-type "$model_type" --model "$model" "${args[@]}" "${shape_args[@]}"
  else
    local module
    module="$(model_module "$model_type" "$model")" || return 1
    _emperor_python -m "$module" "${args[@]}"
  fi
}

_emperor_python -m models.project_cli "$@"
status=$?
return "$status" 2>/dev/null || exit "$status"

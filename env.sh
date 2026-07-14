#!/usr/bin/env bash

# Compatibility wrapper. New automation should use the portable mise tasks.
_emperor_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_emperor_action="start"
_emperor_profile="${EMPEROR_SETUP_PROFILE:-cpu}"
case "${1:-}" in
  "") ;;
  --legacy-profile) _emperor_profile="cuda-legacy" ;;
  --workbench-status) _emperor_action="status" ;;
  --workbench-stop) _emperor_action="stop" ;;
  -h|--help)
    echo "Usage: source env.sh [--legacy-profile] [--workbench-stop] [--workbench-status]"
    echo
    echo "  --legacy-profile   Start with the Linux CUDA 12.6 legacy profile."
    echo "  --workbench-stop    Stop Workbench services."
    echo "  --workbench-status  Print validated Workbench service status."
    return 0 2>/dev/null || exit 0
    ;;
  *)
    echo "Unknown env.sh option: $1"
    echo "Run 'source env.sh --help' for supported options."
    return 1 2>/dev/null || exit 1
    ;;
esac

if ! command -v mise >/dev/null 2>&1; then
  echo "Error: mise is required. Install it from https://mise.jdx.dev/." >&2
  return 1 2>/dev/null || exit 1
fi

cd "$_emperor_root" || { return 1 2>/dev/null || exit 1; }
if [ "$_emperor_action" != "start" ]; then
  mise run "workbench:$_emperor_action"
  _emperor_status=$?
  unset _emperor_action _emperor_profile _emperor_root
  return "$_emperor_status" 2>/dev/null || exit "$_emperor_status"
fi

mise run dev --profile "$_emperor_profile" || {
  return $? 2>/dev/null || exit $?
}
# shellcheck disable=SC1091
source "$_emperor_root/torchenv/bin/activate" || {
  return $? 2>/dev/null || exit $?
}
unset _emperor_action _emperor_profile _emperor_root
return 0 2>/dev/null || exit 0

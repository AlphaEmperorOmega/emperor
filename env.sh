#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/torchenv"
FRONTEND_PATH="$PROJECT_ROOT/workbench/frontend"
WORKBENCH_RUNTIME_PATH="$PROJECT_ROOT/workbench/.runtime"
WORKBENCH_BACKEND_PID="$WORKBENCH_RUNTIME_PATH/backend.pid"
WORKBENCH_FRONTEND_PID="$WORKBENCH_RUNTIME_PATH/frontend.pid"
WORKBENCH_BACKEND_LOG="$WORKBENCH_RUNTIME_PATH/backend.log"
WORKBENCH_FRONTEND_LOG="$WORKBENCH_RUNTIME_PATH/frontend.log"
WORKBENCH_DEPENDENCY_MARKER="$VENV_PATH/.emperor-pyproject.cksum"
WORKBENCH_BACKEND_PORT="${WORKBENCH_BACKEND_PORT:-9999}"
WORKBENCH_FRONTEND_PORT="${WORKBENCH_FRONTEND_PORT:-9000}"
WORKBENCH_API_URL="${NEXT_PUBLIC_WORKBENCH_API_URL:-http://127.0.0.1:$WORKBENCH_BACKEND_PORT}"
WORKBENCH_ACTION="start"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --workbench-stop)
      WORKBENCH_ACTION="stop"
      ;;
    --workbench-status)
      WORKBENCH_ACTION="status"
      ;;
    -h|--help)
      echo "Usage: source env.sh [--workbench-stop] [--workbench-status]"
      echo ""
      echo "  --workbench-stop    Stop workbench servers started by env.sh."
      echo "  --workbench-status  Print workbench server PID status."
      return 0
      ;;
    *)
      echo "Unknown env.sh option: $1"
      echo "Run 'source env.sh --help' for supported options."
      return 1
      ;;
  esac
  shift
done

ensure_mise() {
  if command -v mise &> /dev/null; then
    return 0
  fi

  echo "mise not found. Installing mise..."
  curl https://mise.run | sh
  if [ $? -ne 0 ]; then
    echo "Failed to install mise"
    return 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
}

venv_python_version() {
  if [ ! -x "$VENV_PATH/bin/python" ]; then
    echo ""
    return 0
  fi
  "$VENV_PATH/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

mise_python_version() {
  mise exec -- python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

create_venv() {
  echo "Creating virtual environment..."
  mise exec -- python -m venv "$VENV_PATH"
  if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment"
    return 1
  fi
}

install_frontend_dependencies() {
  if [ ! -f "$FRONTEND_PATH/package.json" ]; then
    return 0
  fi

  if [ -d "$FRONTEND_PATH/node_modules" ]; then
    return 0
  fi

  echo "Installing workbench frontend dependencies..."
  (
    cd "$FRONTEND_PATH" || exit 1
    if [ -f package-lock.json ]; then
      mise exec -- npm ci
    else
      mise exec -- npm install
    fi
  )
}

pyproject_dependency_signature() {
  cksum < "$PROJECT_ROOT/pyproject.toml"
}

backend_python_dependencies_available() {
  "$VENV_PATH/bin/python" - torch fastapi uvicorn tensorboard pydantic pydantic_settings httpx lightning.pytorch ruff <<'PY'
import importlib.util
import sys

missing = [
    module
    for module in sys.argv[1:]
    if importlib.util.find_spec(module) is None
]

if missing:
    print(
        "Missing workbench backend Python dependencies: "
        + ", ".join(missing)
    )
    raise SystemExit(1)
PY
}

project_dependencies_current() {
  local current_signature
  local installed_signature

  if [ ! -f "$WORKBENCH_DEPENDENCY_MARKER" ]; then
    return 1
  fi

  current_signature="$(pyproject_dependency_signature)" || return 1
  installed_signature="$(cat "$WORKBENCH_DEPENDENCY_MARKER" 2>/dev/null)"

  if [ "$current_signature" != "$installed_signature" ]; then
    return 1
  fi

  backend_python_dependencies_available
}

install_project_dependencies() {
  echo "Upgrading pip..."
  "$VENV_PATH/bin/python" -m pip install --upgrade pip || return 1

  echo "Installing project dependencies from pyproject.toml..."
  (
    cd "$PROJECT_ROOT" || exit 1
    "$VENV_PATH/bin/python" -m pip install -e ".[dev]"
  ) || return 1

  pyproject_dependency_signature > "$WORKBENCH_DEPENDENCY_MARKER" || return 1
  backend_python_dependencies_available || return 1
}

ensure_project_dependencies() {
  if project_dependencies_current; then
    return 0
  fi

  install_project_dependencies
}

activate_venv() {
  source "$VENV_PATH/bin/activate"
  echo "Activated virtual environment at $VENV_PATH"
}

pid_running() {
  local pid_file="$1"
  local pid

  if [ ! -f "$pid_file" ]; then
    return 1
  fi

  pid="$(cat "$pid_file" 2>/dev/null)"
  if [ -z "$pid" ]; then
    return 1
  fi

  kill -0 "$pid" 2>/dev/null
}

port_listening() {
  local port="$1"

  case "$port" in
    ""|*[!0-9]*)
      return 1
      ;;
  esac

  (: < "/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
}

listener_pids_for_port() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null
    return 0
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | sed '/^$/d'
    return 0
  fi

  return 0
}

process_command() {
  local pid="$1"
  ps -p "$pid" -o command= 2>/dev/null || true
}

process_group_id() {
  local pid="$1"
  ps -p "$pid" -o pgid= 2>/dev/null | tr -d ' ' || true
}

workbench_service_command_matches() {
  local name="$1"
  local command="$2"

  case "$name" in
    backend)
      [[ "$command" == *"workbench.backend.api:app"* ]] ||
        [[ "$command" == *"spawn_main"* && "$command" == *"--multiprocessing-fork"* ]]
      ;;
    frontend)
      [[ "$command" == *"next dev"* && "$command" == *"-p $WORKBENCH_FRONTEND_PORT"* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

workbench_service_group_command_matches() {
  local name="$1"
  local command="$2"

  if workbench_service_command_matches "$name" "$command"; then
    return 0
  fi

  case "$name" in
    frontend)
      [[ "$command" == *"npm run dev"* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

workbench_service_pids_for_port() {
  local name="$1"
  local port="$2"
  local pid
  local command

  while IFS= read -r pid; do
    if [ -z "$pid" ]; then
      continue
    fi
    command="$(process_command "$pid")"
    if workbench_service_command_matches "$name" "$command"; then
      echo "$pid"
    fi
  done < <(listener_pids_for_port "$port")
}

can_signal_workbench_process_group() {
  local name="$1"
  local pid="$2"
  local pgid="$3"
  local group_command

  if [ -z "$pgid" ]; then
    return 1
  fi

  if [ "$pgid" = "$pid" ]; then
    return 0
  fi

  group_command="$(process_command "$pgid")"
  workbench_service_group_command_matches "$name" "$group_command"
}

signal_workbench_pid() {
  local name="$1"
  local signal="$2"
  local pid="$3"
  local pgid

  pgid="$(process_group_id "$pid")"
  if can_signal_workbench_process_group "$name" "$pid" "$pgid"; then
    kill "-$signal" -- "-$pgid" 2>/dev/null ||
      kill "-$signal" "$pid" 2>/dev/null ||
      true
    return 0
  fi

  kill "-$signal" "$pid" 2>/dev/null || true
}

stop_discovered_workbench_service() {
  local name="$1"
  local port="$2"
  local signal="${3:-TERM}"
  local -a pids=()
  local pid

  mapfile -t pids < <(workbench_service_pids_for_port "$name" "$port")
  if [ "${#pids[@]}" -eq 0 ]; then
    echo "Workbench $name is listening on port $port, but no env.sh pid file was found and the listener was not recognized as a workbench process"
    return 1
  fi

  for pid in "${pids[@]}"; do
    signal_workbench_pid "$name" "$signal" "$pid"
  done
  if [ "$signal" = "KILL" ]; then
    echo "Force-stopped workbench $name (${pids[*]}, discovered from port $port)"
  else
    echo "Stopped workbench $name (${pids[*]}, discovered from port $port)"
  fi
}

wait_for_port() {
  local port="$1"
  local attempts="${2:-40}"
  local delay="${3:-0.25}"
  local attempt=1

  while [ "$attempt" -le "$attempts" ]; do
    if port_listening "$port"; then
      return 0
    fi

    sleep "$delay"
    attempt=$((attempt + 1))
  done

  return 1
}

wait_for_port_to_close() {
  local port="$1"
  local attempts="${2:-40}"
  local delay="${3:-0.25}"
  local attempt=1

  while [ "$attempt" -le "$attempts" ]; do
    if ! port_listening "$port"; then
      return 0
    fi

    sleep "$delay"
    attempt=$((attempt + 1))
  done

  return 1
}

print_log_tail() {
  local log_path="$1"
  local line_count="${2:-40}"

  if [ ! -f "$log_path" ]; then
    echo "No backend log was written at $log_path" >&2
    return 0
  fi

  echo "Last $line_count lines from $log_path:" >&2
  tail -n "$line_count" "$log_path" >&2
}

workbench_status() {
  if pid_running "$WORKBENCH_BACKEND_PID"; then
    echo "Workbench backend running (pid $(cat "$WORKBENCH_BACKEND_PID"))"
  elif port_listening "$WORKBENCH_BACKEND_PORT"; then
    echo "Workbench backend listening on port $WORKBENCH_BACKEND_PORT (no env.sh pid file)"
  else
    echo "Workbench backend stopped"
  fi

  if pid_running "$WORKBENCH_FRONTEND_PID"; then
    echo "Workbench frontend running (pid $(cat "$WORKBENCH_FRONTEND_PID"))"
  elif port_listening "$WORKBENCH_FRONTEND_PORT"; then
    echo "Workbench frontend listening on port $WORKBENCH_FRONTEND_PORT (no env.sh pid file)"
  else
    echo "Workbench frontend stopped"
  fi
}

stop_workbench_service() {
  local name="$1"
  local pid_file="$2"
  local port="$3"
  local pid

  if ! pid_running "$pid_file"; then
    rm -f "$pid_file"
    if port_listening "$port"; then
      stop_discovered_workbench_service "$name" "$port"
      return $?
    fi
    echo "Workbench $name already stopped"
    return 0
  fi

  pid="$(cat "$pid_file")"
  kill "$pid" 2>/dev/null || true
  rm -f "$pid_file"
  echo "Stopped workbench $name (pid $pid)"
}

stop_workbench() {
  stop_workbench_service "backend" "$WORKBENCH_BACKEND_PID" "$WORKBENCH_BACKEND_PORT"
  stop_workbench_service "frontend" "$WORKBENCH_FRONTEND_PID" "$WORKBENCH_FRONTEND_PORT"
}

start_workbench_backend() {
  local -a reload_args=(
    --reload
    --reload-dir "$PROJECT_ROOT/emperor"
    --reload-dir "$PROJECT_ROOT/models"
    --reload-dir "$PROJECT_ROOT/workbench/backend"
  )

  if pid_running "$WORKBENCH_BACKEND_PID"; then
    if port_listening "$WORKBENCH_BACKEND_PORT"; then
      echo "Workbench backend already started; reusing existing server (pid $(cat "$WORKBENCH_BACKEND_PID"), port $WORKBENCH_BACKEND_PORT)"
      return 0
    fi

    echo "Workbench backend pid file exists, but port $WORKBENCH_BACKEND_PORT is not listening."
    print_log_tail "$WORKBENCH_BACKEND_LOG"
    rm -f "$WORKBENCH_BACKEND_PID"
    return 1
  fi

  rm -f "$WORKBENCH_BACKEND_PID"
  if port_listening "$WORKBENCH_BACKEND_PORT"; then
    echo "Workbench backend is listening on port $WORKBENCH_BACKEND_PORT without a pid file; restarting it to apply current launcher settings."
    stop_discovered_workbench_service "backend" "$WORKBENCH_BACKEND_PORT" || return 1
    if ! wait_for_port_to_close "$WORKBENCH_BACKEND_PORT"; then
      echo "Workbench backend port $WORKBENCH_BACKEND_PORT did not close after TERM; forcing the discovered workbench backend to stop." >&2
      stop_discovered_workbench_service "backend" "$WORKBENCH_BACKEND_PORT" "KILL" || return 1
      if ! wait_for_port_to_close "$WORKBENCH_BACKEND_PORT"; then
        echo "Workbench backend port $WORKBENCH_BACKEND_PORT did not close after force-stopping the discovered process." >&2
        return 1
      fi
    fi
  fi

  echo "Starting workbench backend in the background on port $WORKBENCH_BACKEND_PORT..."
  (
    cd "$PROJECT_ROOT" || exit 1
    if command -v setsid >/dev/null 2>&1; then
      MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
        WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS="${WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS:-true}" \
        setsid nohup "$VENV_PATH/bin/python" -m uvicorn workbench.backend.api:app \
          "${reload_args[@]}" --host 127.0.0.1 --port "$WORKBENCH_BACKEND_PORT" \
          </dev/null > "$WORKBENCH_BACKEND_LOG" 2>&1 &
    else
      MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
        WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS="${WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS:-true}" \
        nohup "$VENV_PATH/bin/python" -m uvicorn workbench.backend.api:app \
          "${reload_args[@]}" --host 127.0.0.1 --port "$WORKBENCH_BACKEND_PORT" \
          </dev/null > "$WORKBENCH_BACKEND_LOG" 2>&1 &
    fi
    echo "$!" > "$WORKBENCH_BACKEND_PID"
  ) || return 1

  if wait_for_port "$WORKBENCH_BACKEND_PORT"; then
    return 0
  fi

  echo "Workbench backend did not start listening on port $WORKBENCH_BACKEND_PORT." >&2
  print_log_tail "$WORKBENCH_BACKEND_LOG"
  rm -f "$WORKBENCH_BACKEND_PID"
  return 1
}

start_workbench_frontend() {
  if pid_running "$WORKBENCH_FRONTEND_PID"; then
    echo "Workbench frontend already started; reusing existing server (pid $(cat "$WORKBENCH_FRONTEND_PID"), port $WORKBENCH_FRONTEND_PORT)"
    return 0
  fi

  rm -f "$WORKBENCH_FRONTEND_PID"
  if port_listening "$WORKBENCH_FRONTEND_PORT"; then
    echo "Workbench frontend already started; port $WORKBENCH_FRONTEND_PORT is listening (no env.sh pid file)"
    return 0
  fi

  echo "Starting workbench frontend in the background on port $WORKBENCH_FRONTEND_PORT..."
  (
    cd "$FRONTEND_PATH" || exit 1
    if command -v setsid >/dev/null 2>&1; then
      NEXT_PUBLIC_WORKBENCH_API_URL="$WORKBENCH_API_URL" \
        PORT="$WORKBENCH_FRONTEND_PORT" setsid nohup mise exec -- npm run dev \
          </dev/null > "$WORKBENCH_FRONTEND_LOG" 2>&1 &
    else
      NEXT_PUBLIC_WORKBENCH_API_URL="$WORKBENCH_API_URL" \
        PORT="$WORKBENCH_FRONTEND_PORT" nohup mise exec -- npm run dev \
          </dev/null > "$WORKBENCH_FRONTEND_LOG" 2>&1 &
    fi
    echo "$!" > "$WORKBENCH_FRONTEND_PID"
  )
}

start_workbench() {
  ensure_mise || return 1
  mise install || return 1
  ensure_project_dependencies || return 1
  install_frontend_dependencies || return 1
  mkdir -p "$WORKBENCH_RUNTIME_PATH"

  start_workbench_backend || return 1
  start_workbench_frontend || return 1

  echo "Workbench running in the background at http://localhost:$WORKBENCH_FRONTEND_PORT"
  echo "Backend log: $WORKBENCH_BACKEND_LOG"
  echo "Frontend log: $WORKBENCH_FRONTEND_LOG"
}

if [ "$WORKBENCH_ACTION" = "stop" ]; then
  stop_workbench
  return 0
fi

if [ "$WORKBENCH_ACTION" = "status" ]; then
  workbench_status
  return 0
fi

ensure_mise || return 1

echo "Installing tools from mise.toml..."
mise install || return 1

EXPECTED_PYTHON_VERSION="$(mise_python_version)"
CURRENT_VENV_VERSION="$(venv_python_version)"

if [ -d "$VENV_PATH" ] && [ "$CURRENT_VENV_VERSION" != "$EXPECTED_PYTHON_VERSION" ]; then
  echo "Recreating virtual environment for Python $EXPECTED_PYTHON_VERSION..."
  rm -rf "$VENV_PATH"
fi

if [ ! -d "$VENV_PATH" ]; then
  create_venv || return 1
fi

activate_venv

ensure_project_dependencies || return 1

install_frontend_dependencies || return 1

echo "Environment is ready"
start_workbench || return 1

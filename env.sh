#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/torchenv"
FRONTEND_PATH="$PROJECT_ROOT/viewer/frontend"
VIEWER_RUNTIME_PATH="$PROJECT_ROOT/viewer/.runtime"
VIEWER_BACKEND_PID="$VIEWER_RUNTIME_PATH/backend.pid"
VIEWER_FRONTEND_PID="$VIEWER_RUNTIME_PATH/frontend.pid"
VIEWER_BACKEND_LOG="$VIEWER_RUNTIME_PATH/backend.log"
VIEWER_FRONTEND_LOG="$VIEWER_RUNTIME_PATH/frontend.log"
VIEWER_BACKEND_PORT="${VIEWER_BACKEND_PORT:-9999}"
VIEWER_FRONTEND_PORT="${VIEWER_FRONTEND_PORT:-9000}"
VIEWER_API_URL="${NEXT_PUBLIC_VIEWER_API_URL:-http://127.0.0.1:$VIEWER_BACKEND_PORT}"
VIEWER_ACTION="start"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --viewer-stop)
      VIEWER_ACTION="stop"
      ;;
    --viewer-status)
      VIEWER_ACTION="status"
      ;;
    -h|--help)
      echo "Usage: source env.sh [--viewer-stop] [--viewer-status]"
      echo ""
      echo "  --viewer-stop    Stop viewer servers started by env.sh."
      echo "  --viewer-status  Print viewer server PID status."
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

  echo "Installing viewer frontend dependencies..."
  (
    cd "$FRONTEND_PATH" || exit 1
    if [ -f package-lock.json ]; then
      mise exec -- npm ci
    else
      mise exec -- npm install
    fi
  )
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

viewer_status() {
  if pid_running "$VIEWER_BACKEND_PID"; then
    echo "Viewer backend running (pid $(cat "$VIEWER_BACKEND_PID"))"
  elif port_listening "$VIEWER_BACKEND_PORT"; then
    echo "Viewer backend listening on port $VIEWER_BACKEND_PORT (no env.sh pid file)"
  else
    echo "Viewer backend stopped"
  fi

  if pid_running "$VIEWER_FRONTEND_PID"; then
    echo "Viewer frontend running (pid $(cat "$VIEWER_FRONTEND_PID"))"
  elif port_listening "$VIEWER_FRONTEND_PORT"; then
    echo "Viewer frontend listening on port $VIEWER_FRONTEND_PORT (no env.sh pid file)"
  else
    echo "Viewer frontend stopped"
  fi
}

stop_viewer_service() {
  local name="$1"
  local pid_file="$2"
  local port="$3"
  local pid

  if ! pid_running "$pid_file"; then
    rm -f "$pid_file"
    if port_listening "$port"; then
      echo "Viewer $name is listening on port $port, but no env.sh pid file was found"
      return 0
    fi
    echo "Viewer $name already stopped"
    return 0
  fi

  pid="$(cat "$pid_file")"
  kill "$pid" 2>/dev/null || true
  rm -f "$pid_file"
  echo "Stopped viewer $name (pid $pid)"
}

stop_viewer() {
  stop_viewer_service "backend" "$VIEWER_BACKEND_PID" "$VIEWER_BACKEND_PORT"
  stop_viewer_service "frontend" "$VIEWER_FRONTEND_PID" "$VIEWER_FRONTEND_PORT"
}

start_viewer_backend() {
  if pid_running "$VIEWER_BACKEND_PID"; then
    echo "Viewer backend already started; reusing existing server (pid $(cat "$VIEWER_BACKEND_PID"), port $VIEWER_BACKEND_PORT)"
    return 0
  fi

  rm -f "$VIEWER_BACKEND_PID"
  if port_listening "$VIEWER_BACKEND_PORT"; then
    echo "Viewer backend already started; port $VIEWER_BACKEND_PORT is listening (no env.sh pid file)"
    return 0
  fi

  echo "Starting viewer backend in the background on port $VIEWER_BACKEND_PORT..."
  (
    cd "$PROJECT_ROOT" || exit 1
    if command -v setsid >/dev/null 2>&1; then
      MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
        setsid nohup "$VENV_PATH/bin/python" -m uvicorn viewer.backend.api:app \
          --reload --host 127.0.0.1 --port "$VIEWER_BACKEND_PORT" \
          </dev/null > "$VIEWER_BACKEND_LOG" 2>&1 &
    else
      MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
        nohup "$VENV_PATH/bin/python" -m uvicorn viewer.backend.api:app \
          --reload --host 127.0.0.1 --port "$VIEWER_BACKEND_PORT" \
          </dev/null > "$VIEWER_BACKEND_LOG" 2>&1 &
    fi
    echo "$!" > "$VIEWER_BACKEND_PID"
  )
}

start_viewer_frontend() {
  if pid_running "$VIEWER_FRONTEND_PID"; then
    echo "Viewer frontend already started; reusing existing server (pid $(cat "$VIEWER_FRONTEND_PID"), port $VIEWER_FRONTEND_PORT)"
    return 0
  fi

  rm -f "$VIEWER_FRONTEND_PID"
  if port_listening "$VIEWER_FRONTEND_PORT"; then
    echo "Viewer frontend already started; port $VIEWER_FRONTEND_PORT is listening (no env.sh pid file)"
    return 0
  fi

  echo "Starting viewer frontend in the background on port $VIEWER_FRONTEND_PORT..."
  (
    cd "$FRONTEND_PATH" || exit 1
    if command -v setsid >/dev/null 2>&1; then
      NEXT_PUBLIC_VIEWER_API_URL="$VIEWER_API_URL" \
        PORT="$VIEWER_FRONTEND_PORT" setsid nohup mise exec -- npm run dev \
          </dev/null > "$VIEWER_FRONTEND_LOG" 2>&1 &
    else
      NEXT_PUBLIC_VIEWER_API_URL="$VIEWER_API_URL" \
        PORT="$VIEWER_FRONTEND_PORT" nohup mise exec -- npm run dev \
          </dev/null > "$VIEWER_FRONTEND_LOG" 2>&1 &
    fi
    echo "$!" > "$VIEWER_FRONTEND_PID"
  )
}

start_viewer() {
  ensure_mise || return 1
  mise install || return 1
  install_frontend_dependencies || return 1
  mkdir -p "$VIEWER_RUNTIME_PATH"

  start_viewer_backend || return 1
  start_viewer_frontend || return 1

  echo "Viewer running in the background at http://localhost:$VIEWER_FRONTEND_PORT"
  echo "Backend log: $VIEWER_BACKEND_LOG"
  echo "Frontend log: $VIEWER_FRONTEND_LOG"
}

if [ "$VIEWER_ACTION" = "stop" ]; then
  stop_viewer
  return 0
fi

if [ "$VIEWER_ACTION" = "status" ]; then
  viewer_status
  return 0
fi

if [ -d "$VENV_PATH" ]; then
  activate_venv
  start_viewer || return 1
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

source "$VENV_PATH/bin/activate"
echo "Activated virtual environment at $VENV_PATH"

echo "Upgrading pip..."
pip install --upgrade pip || return 1

echo "Installing project dependencies from pyproject.toml..."
pip install -e . || return 1

install_frontend_dependencies || return 1

echo "Environment is ready"
start_viewer || return 1

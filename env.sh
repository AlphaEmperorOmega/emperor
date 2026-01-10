#!/usr/bin/env bash

VENV_PATH="$HOME/Development/001_emperor/torchenv"

if [ ! -d "$VENV_PATH" ]; then
  echo "Virtual environment not found at $VENV_PATH"
  return 1 2>/dev/null || exit 1
fi

source "$VENV_PATH/bin/activate"
echo "Activated virtual environment at $VENV_PATH"

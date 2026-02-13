#!/usr/bin/env bash

VENV_PATH="./torchenv"

if [ ! -d "$VENV_PATH" ]; then
  echo "Virtual environment not found at $VENV_PATH"
  echo "Creating virtual environment..."

  python3 -m venv "$VENV_PATH"

  if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment"
    return 1 2>/dev/null || exit 1
  fi

  echo "Virtual environment created successfully"
  source "$VENV_PATH/bin/activate"

  echo "Upgrading pip..."
  pip install --upgrade pip

  echo "Installing project dependencies from pyproject.toml..."
  pip install -e .

  if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    return 1 2>/dev/null || exit 1
  fi

  echo "Dependencies installed successfully"
else
  source "$VENV_PATH/bin/activate"
  echo "Activated virtual environment at $VENV_PATH"
fi

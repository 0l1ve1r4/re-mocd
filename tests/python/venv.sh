#!/bin/bash

set -e

VENV_DIR="./tests/python/.venv"

if [ -d "$VENV_DIR" ]; then
  echo "Virtual environment already exists at $VENV_DIR"
else
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

REQ_FILE="tests/python/requirements.txt"
if [ -f "$REQ_FILE" ]; then
  echo "Installing dependencies from $REQ_FILE..."
  pip install -r "$REQ_FILE"
else
  echo "Error: Requirements file not found at $REQ_FILE"
  deactivate
  exit 1
fi

echo "Setup complete."

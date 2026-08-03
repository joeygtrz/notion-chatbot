#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "Error: .env file not found."
  echo "Create it with: echo 'ANTHROPIC_API_KEY=your-key-here' > .env"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install -q fastapi uvicorn anthropic python-dotenv scikit-learn numpy
fi

echo "Starting Notion Chat at http://localhost:8766"
open http://localhost:8766 2>/dev/null || true
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8766

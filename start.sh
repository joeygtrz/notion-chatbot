#!/bin/bash
set -e
cd "$(dirname "$0")"

# Credentials: log in once with `ant auth login` (recommended), or put
# ANTHROPIC_API_KEY=... in .env. A key in .env takes priority over the login.
if ! grep -qs '^ANTHROPIC_API_KEY=.' .env && ! command -v ant >/dev/null; then
  echo "Error: no Anthropic credentials found."
  echo "Install the Anthropic CLI and log in once:"
  echo "  brew install anthropics/tap/ant"
  echo "  xattr -d com.apple.quarantine \"\$(brew --prefix)/bin/ant\""
  echo "  ant auth login"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
# Keep the SDK current so it can read the `ant auth login` profile.
.venv/bin/pip install -q --upgrade fastapi uvicorn anthropic python-dotenv scikit-learn numpy

echo "Starting Notion Chat at http://localhost:8766"
open http://localhost:8766 2>/dev/null || true
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8766

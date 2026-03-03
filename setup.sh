#!/usr/bin/env bash
# Cross-platform setup: install deps + git hooks.
# macOS / Linux: bash setup.sh
# Windows (Git Bash / WSL): bash setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"

echo ">>> uv sync"
uv sync --directory "$SRC"

echo ">>> pre-commit install"
uv run --directory "$SRC" pre-commit install

echo ""
echo "Setup complete. Git hooks are active."

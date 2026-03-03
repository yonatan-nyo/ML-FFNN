#!/usr/bin/env bash
# Strip notebook outputs, re-stage, and exit 0 so the commit proceeds.
set -euo pipefail

VENV_NBSTRIPOUT="$(git rev-parse --show-toplevel)/src/.venv/bin/nbstripout"

for nb in "$@"; do
    "$VENV_NBSTRIPOUT" "$nb"
    git add "$nb"
done

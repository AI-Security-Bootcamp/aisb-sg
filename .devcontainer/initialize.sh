#!/usr/bin/env bash
# Runs on the HOST before the container starts (devcontainer initializeCommand).
# Creates directories and files required by bind mounts / --env-file so that
# Docker doesn't fail on a fresh checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# SSH keys are bind-mounted into the container; the directory must exist.
mkdir -p "$REPO_ROOT/ssh"

# Pip cache shared with the container to speed up rebuilds.
mkdir -p "$REPO_ROOT/.package-cache"

# Docker requires --env-file to exist even if it is empty.
# Copy the example file on first checkout so the developer knows what to fill in.
if [ ! -f "$REPO_ROOT/.env" ]; then
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    echo "Created .env from .env.example — fill in your API keys."
fi

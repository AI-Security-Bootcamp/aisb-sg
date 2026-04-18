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

# docker-outside-of-docker bind-mounts /var/run/docker.sock by default. That matches
# the usual root/group Docker setup (team default). Rootless setups often have no
# socket there, so we pick a real socket below and point a stable repo-local symlink
# at it; devcontainer.json mounts that symlink to /var/run/docker-host.sock and
# overrides the feature's default mount (see devcontainers/features docker-outside-of-docker README).
RUN_DIR="$REPO_ROOT/.devcontainer/.run"
mkdir -p "$RUN_DIR"
host_sock=""
uid="$(id -u)"
# 1) Root daemon or docker group access — default for most Linux/macOS Docker installs.
if [ -S /var/run/docker.sock ]; then
	host_sock=/var/run/docker.sock
fi
# 2) Explicit unix socket (e.g. rootless context); only if classic path did not win.
if [ -z "$host_sock" ] && [ -n "${DOCKER_HOST:-}" ]; then
	case "$DOCKER_HOST" in
	unix://*)
		dh="${DOCKER_HOST#unix://}"
		if [ -S "$dh" ]; then
			host_sock="$dh"
		fi
		;;
	esac
fi
# 3) Rootless / user-session daemon (only when nothing above matched).
if [ -z "$host_sock" ] && [ -n "${XDG_RUNTIME_DIR:-}" ] && [ -S "${XDG_RUNTIME_DIR}/docker.sock" ]; then
	host_sock="${XDG_RUNTIME_DIR}/docker.sock"
fi
if [ -z "$host_sock" ] && [ -S "/run/user/${uid}/docker.sock" ]; then
	host_sock="/run/user/${uid}/docker.sock"
fi
if [ -z "$host_sock" ] && [ -n "${HOME:-}" ] && [ -S "${HOME}/.docker/run/docker.sock" ]; then
	host_sock="${HOME}/.docker/run/docker.sock"
fi
if [ -n "$host_sock" ]; then
	ln -sfn "$host_sock" "$RUN_DIR/docker-host.sock"
else
	rm -f "$RUN_DIR/docker-host.sock"
	echo "Warning: no Docker API socket found. Start Docker (root: /var/run/docker.sock) or rootless engine" \
		"(\$XDG_RUNTIME_DIR/docker.sock, ~/.docker/run/docker.sock, or set DOCKER_HOST=unix://...)." >&2
fi

# Docker requires --env-file to exist even if it is empty.
# Copy the example file on first checkout so the developer knows what to fill in.
if [ ! -f "$REPO_ROOT/.env" ]; then
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    echo "Created .env from .env.example — fill in your API keys."
fi

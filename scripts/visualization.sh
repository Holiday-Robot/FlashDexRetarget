#!/bin/bash
# Usage: bash scripts/visualization.sh <dataset dir> [saved rollouts dir ...]  (PORT=8080 GEOM=convex|visual ROBOT= FPS=60 CHECK=1)
set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # env.sh sets its own PROJECT_DIR
[ -f "$REPO_DIR/scripts/env.sh" ] && source "$REPO_DIR/scripts/env.sh"
DATASET=${1:?usage: bash scripts/visualization.sh <dataset dir> [saved rollouts dir ...]}
shift

# viser needs websockets>=13.1 but the Isaac freeze pins 12.0, so it lives outside the env
SITE="$REPO_DIR/third_party/viewer_site"
if ! PYTHONPATH="$SITE" python -c "import viser" 2>/dev/null; then
    echo "[visualization] installing viser into $SITE"
    pip install -q --no-deps --target "$SITE" viser==1.0.27 "websockets>=13.1,<17" "msgspec>=0.18.6,<1" \
        "zstandard>=0.20,<1"
fi

PORT=${PORT:-8080}
ARGS=(--dataset "$DATASET" --port "$PORT" --fps "${FPS:-60}" --geom "${GEOM:-convex}")
[ $# -gt 0 ] && ARGS+=(--saved "$@")
[ -n "${ROBOT:-}" ] && ARGS+=(--robot "$ROBOT")
[ "${CHECK:-0}" = 1 ] && ARGS+=(--check)
[ "${CHECK:-0}" = 1 ] || echo "[visualization] open http://$(hostname -I | awk '{print $1}'):$PORT (or ssh -L $PORT:localhost:$PORT $(hostname))"
PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}" exec python "$REPO_DIR/src/utils/visualization.py" "${ARGS[@]}"

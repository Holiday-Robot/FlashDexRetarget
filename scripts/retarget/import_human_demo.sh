#!/bin/bash
# Hydra overrides of config/retarget/import_human_demo.yaml, e.g. source=taco raw=DIR dataset=DIR
set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # env.sh sets its own PROJECT_DIR
[ -f "$REPO_DIR/scripts/env.sh" ] && source "$REPO_DIR/scripts/env.sh"
python "$REPO_DIR/src/retarget/import_human_demo.py" "$@"

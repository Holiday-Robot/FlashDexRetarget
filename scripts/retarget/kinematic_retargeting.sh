#!/bin/bash
# Hydra overrides of config/retarget/kinematic_retargeting.yaml, e.g. dataset=DIR steps.usd=false
set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # env.sh sets its own PROJECT_DIR
[ -f "$REPO_DIR/scripts/env.sh" ] && source "$REPO_DIR/scripts/env.sh"
python "$REPO_DIR/src/retarget/kinematic_retargeting.py" "$@"

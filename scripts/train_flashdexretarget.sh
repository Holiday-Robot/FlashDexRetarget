#!/bin/bash
# MOTION=<motion.pt> [ROBOT=xhand|sharpa] bash scripts/train_flashdexretarget.sh [hydra overrides ...]
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
unset HEADLESS   # Isaac Lab's own env var; use the headless=false override instead

robot="${ROBOT:-xhand}"
motion="$(realpath "${MOTION:?set MOTION=<dataset>/motion.pt}")"

dir="$(dirname "$motion")"
[ "$(basename "$dir")" = "$robot" ] && dir="$(dirname "$dir")"
dataset="$(basename "$dir")"

args=(--overrides envs/robot@robot=${robot}_bimanual)
if [ "$robot" = sharpa ]; then
    unset FDR_ROBOT_USD   # env.sh points it at the XHAND USD
fi
args+=(
    --overrides commands.motion_file="$motion"
    --overrides logger.wandb.run_name=flashdexretarget-${dataset}-${robot}
    --overrides "logger.wandb.tags=[${dataset},${robot}]"
)
for ov in "$@"; do args+=(--overrides "$ov"); done

python train.py "${args[@]}"

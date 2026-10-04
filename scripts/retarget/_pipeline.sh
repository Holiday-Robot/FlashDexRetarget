#!/bin/bash
# Release -> trainable motion files: import <raw_dataset_path> once into <output_dataset_path>
# (human_demo/, objects/), then retarget it to each robot of ROBOTS in <output_dataset_path>/<robot>/, which
# links the shared human_demo/ and objects/ (convex parts are made once). Run through taco.sh, oakink2.sh or
# hot3d_1obj.sh (they set SOURCE and IMPORT).
set -e
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -lt 2 ]; then
    cat >&2 <<USAGE
Usage: bash scripts/retarget/<dataset>.sh <raw_dataset_path> <output_dataset_path> [Hydra overrides of kinematic_retargeting.yaml]
  <raw_dataset_path>     the release (taco: its root, oakink2: anno_preview/ program/ object_repair/, hot3d: dataset/ _assets/)
  <output_dataset_path>  human_demo/ and objects/ (shared), <robot>/{retargeted/, report.csv, motion.pt}
  ROBOTS                 robots to retarget to, one after another (default: xhand), e.g. ROBOTS="xhand sharpa"
  REIMPORT               1: import again although <output_dataset_path>/human_demo/ has clips
  e.g. ROBOTS="xhand sharpa" bash scripts/retarget/taco.sh /data/taco /data/taco_fdr workers=64 steps.usd=false
USAGE
    exit 1
fi
RAW=$1 OUT=$2
shift 2
WORKERS=$(printf '%s\n' "$@" | grep -m1 '^workers=' || true)   # also used by the import

if [ -n "$(ls -A "$OUT/human_demo" 2>/dev/null)" ] && [ "${REIMPORT:-0}" != 1 ]; then
    echo "[retarget/$SOURCE] $OUT/human_demo has clips: import skipped (REIMPORT=1 to redo)"
else
    bash "$SCRIPTS/import_human_demo.sh" source="$SOURCE" raw="$RAW" dataset="$OUT" $WORKERS $IMPORT
fi
for robot in ${ROBOTS:-xhand}; do
    mkdir -p "$OUT/$robot"
    ln -sfn ../human_demo "$OUT/$robot/human_demo"
    ln -sfn ../objects "$OUT/$robot/objects"
    bash "$SCRIPTS/kinematic_retargeting.sh" dataset="$OUT/$robot" robot="$robot" "$@"
done

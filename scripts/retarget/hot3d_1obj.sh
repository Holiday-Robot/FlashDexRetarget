#!/bin/bash
# HOT3D recordings -> segments in which both hands handle one object (CHORD's mode B) -> <output_dataset_path>/<robot>/motion.pt:
# bash scripts/retarget/hot3d_1obj.sh <raw_dataset_path> <output_dataset_path> [overrides]
SOURCE=hot3d IMPORT="source.segment.modes=[B]" exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" "$@"

#!/bin/bash
# TACO -> <output_dataset_path>/<robot>/motion.pt:
# bash scripts/retarget/taco.sh <raw_dataset_path> <output_dataset_path> [overrides]
SOURCE=taco exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" "$@"

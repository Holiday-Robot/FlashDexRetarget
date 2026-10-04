#!/bin/bash
# OakInk2 stages -> <output_dataset_path>/<robot>/motion.pt:
# bash scripts/retarget/oakink2.sh <raw_dataset_path> <output_dataset_path> [overrides]
SOURCE=oakink2 exec bash "$(dirname "${BASH_SOURCE[0]}")/_pipeline.sh" "$@"

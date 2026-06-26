#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python SAGE_SAM_R4/tools/validate_dataset.py \
  --config SAGE_SAM_R4/configs/r4_3class_v100_tuned.yaml

python SAGE_SAM_R4/tools/verify_real_sam.py \
  --config SAGE_SAM_R4/configs/r4_3class_v100_tuned.yaml

mkdir -p outputs/SAGE_SAM_R4_3Class_V100_Tuned
python SAGE_SAM_R4/train_r4.py \
  --config SAGE_SAM_R4/configs/r4_3class_v100_tuned.yaml \
  2>&1 | tee outputs/SAGE_SAM_R4_3Class_V100_Tuned/stdout.log

#!/usr/bin/env bash
set -euo pipefail
python SAGE_SAM_R4/test_r4.py --config outputs/SAGE_SAM_R4_3Class/resolved_config.yaml --checkpoint outputs/SAGE_SAM_R4_3Class/checkpoints/best_val_dice.pth --save-pred --split test


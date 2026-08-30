#!/usr/bin/env bash
# Launch StyleGAN2-ADA transfer-learning training on the RunPod GPU pod.
# Run this ON THE POD, from anywhere (paths are absolute in the config).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../configs/train_config.env"

if [ ! -f "$DATA_ZIP" ]; then
  echo "error: dataset zip not found at $DATA_ZIP (run dataset_tool.py first)" >&2
  exit 1
fi

cd "$STYLEGAN_REPO"

python train.py \
  --outdir="$OUTDIR" \
  --data="$DATA_ZIP" \
  --gpus="$GPUS" \
  --cfg="$CFG" \
  --resume="$RESUME" \
  --mirror="$MIRROR" \
  --aug="$AUG" \
  --kimg="$KIMG" \
  --snap="$SNAP"

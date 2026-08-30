#!/usr/bin/env bash
# Assemble a PNG frame sequence (from generate_latent_walk.py) into an H.264
# video with a light contrast/saturation grade for that glossy Anadol look.
set -euo pipefail

FRAMES_DIR="${1:?usage: render_video.sh <frames_dir> <output.mp4> [fps]}"
OUTPUT="${2:?usage: render_video.sh <frames_dir> <output.mp4> [fps]}"
FPS="${3:-30}"

ffmpeg -y -framerate "$FPS" -i "$FRAMES_DIR/frame_%06d.png" \
  -vf "eq=saturation=1.15:contrast=1.05,format=yuv420p" \
  -c:v libx264 -pix_fmt yuv420p -crf 16 -preset slow \
  "$OUTPUT"

echo "wrote $OUTPUT"

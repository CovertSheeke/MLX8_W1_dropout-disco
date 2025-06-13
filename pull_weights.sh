#!/usr/bin/env bash
# assumes ssh config for remote is set up as 'mlx'
# also assumes setup.sh was run previously, so that remote has rsync
if [[ -z "${1-}" ]]; then
    REMOTE="mlx"
else
    REMOTE="$1"
fi
rsync -avz --exclude='checkpoint_*' "$REMOTE:MLX8_W1_dropout-disco/weights/" weights/

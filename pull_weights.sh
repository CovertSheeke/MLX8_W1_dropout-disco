#!/usr/bin/env bash
# assumes ssh config for remote is set up as 'mlx'
# also assumes setup.sh was run previously, so that remote has rsync
rsync -avz --exclude='checkpoint_*' mlx:MLX8_W1_dropout-disco/weights/ weights/

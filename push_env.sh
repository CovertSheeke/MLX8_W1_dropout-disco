#!/usr/bin/env bash
# assumes ssh config for remote is set up as 'mlx'
if [[ -z "${1-}" ]]; then
    REMOTE="mlx"
else
    REMOTE="$1"
fi
scp .env "$REMOTE:MLX8_W1_dropout-disco/.env" 

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <IP> [ssh_keyfile]" >&2
    exit 1
fi

IP=$1
if [[ -z "${2-}" ]]; then        # ${2-} is empty when $2 is unset
    PRIVATE_KEY="id_mlx_computa"
else
    PRIVATE_KEY="$2"
fi

ssh -i "$HOME/.ssh/$PRIVATE_KEY" -p 4422 "root@$IP" -v

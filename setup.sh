#!/usr/bin/env bash
apt update
# ensure we have all the utils we need
apt install -y vim rsync git
apt upgrade -y
# install uv and syn
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
# activate virtual environment for running python scripts
source .venv/bin/activate

#!/bin/bash

echo "Syncing uv environment..."
source /root/.venv/bin/activate
uv sync --active --no-install-package flash-attn
uv sync --active --no-build-isolation
uv add --active "huggingface_hub" "wandb"
export HF_HOME="/root/hf"
hf auth login --token $RUNPOD_HF_TOKEN --add-to-git-credential
wandb login $RUNPOD_WANDB_TOKEN

# Plotly pdf saving requires this
plotly_get_chrome -y
sudo apt update -y && sudo apt-get install -y libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2

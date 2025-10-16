#!/bin/bash

echo "Syncing uv environment..."
source /root/.venv/bin/activate
uv sync --active --no-install-package flash-attn
uv sync --active --no-build-isolation
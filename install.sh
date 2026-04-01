#!/bin/bash
echo "=== DeepBranchAI Installation ==="
conda create -n deepbranchai python=3.12 -y
conda activate deepbranchai
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
echo "=== Installation complete ==="
echo "Activate: conda activate deepbranchai"
echo "Then run: jupyter notebook"

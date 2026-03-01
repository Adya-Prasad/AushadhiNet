#!/bin/bash
set -e

echo "=== AushadhiNet-GATv2 DDI Workspace Starting ==="
cd /workspaces/ddi

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements_docker.txt -q

echo "Launching Streamlit DDI Inference App..."
nohup streamlit run inference_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  > /tmp/streamlit.log 2>&1 &

echo "Done. App running on port 8501."
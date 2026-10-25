#!/usr/bin/env bash
# ==============================================================================
# ClipSmith - Campus JupyterLab (TLJH CPU) Startup Script
# ==============================================================================

set -e

# Base directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 1. Environment Variables for CPU & Shared Server Safety
export DEVICE="cpu"
export COMPUTE_TYPE="int8"
export VIDEO_ENCODER="libx264"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OLLAMA_MAX_LOADED_MODELS=1
export FLASK_PORT=5000

echo ">>> Starting ClipSmith in CPU mode (Campus Profile)..."
echo ">>> Device: $DEVICE | Compute: $COMPUTE_TYPE | Threads: $OMP_NUM_THREADS"

# 2. Check if Ollama is running, start in background if needed
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo ">>> Starting Ollama service in background..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 3
fi

# 3. Activate Virtual Environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 4. Start ClipSmith Web & Worker
python run.py

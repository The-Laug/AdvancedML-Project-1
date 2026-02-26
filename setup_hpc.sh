#!/bin/bash
# ──────────────────────────────────────────────────────────
# HPC Environment Setup Script
# Run this ONCE on the DTU HPC to set up the project.
# ──────────────────────────────────────────────────────────
set -e

echo "========================================="
echo "  Source-Sink GNN — HPC Setup"
echo "========================================="

# ── Configuration (EDIT THESE) ────────────────────────────
# Where to put the project on HPC
REPO="$HOME/source_sink_gnn"

echo ""
echo "Project will be set up at: $REPO"
echo ""

# ── Create project directory ──────────────────────────────
mkdir -p "$REPO"
cd "$REPO"

# ── Install uv (fast Python package manager) ─────────────
echo "Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# ── Create Python virtual environment with uv ─────────────
echo ""
echo "Creating virtual environment..."
uv venv --python 3.12 "$REPO/.venv"
source "$REPO/.venv/bin/activate"

# ── Install PyTorch with CUDA support ─────────────────────
# For V100/A100: CUDA 12.x is appropriate
echo ""
echo "Installing PyTorch with CUDA 12.4..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ── Install torch-geometric and dependencies ──────────────
echo ""
echo "Installing torch-geometric..."
uv pip install torch-geometric

# ── Install remaining ML dependencies ─────────────────────
echo ""
echo "Installing other dependencies..."
uv pip install pandas matplotlib seaborn

# ── Verify installation ───────────────────────────────────
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:    {torch.version.cuda}')
    print(f'GPU count:       {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('(GPU check skipped - run on a GPU node to verify)')

import torch_geometric
print(f'PyG version:     {torch_geometric.__version__}')
import pandas
print(f'Pandas version:  {pandas.__version__}')
print()
print('All packages installed successfully!')
"

# ── Create directory structure ────────────────────────────
echo ""
echo "Creating directory structure..."
mkdir -p "$REPO/data/gnn_ready_source_sink"
mkdir -p "$REPO/hpc_experiments/configs"
mkdir -p "$REPO/hpc_experiments/jobs"
mkdir -p "$REPO/hpc_experiments/logs"
mkdir -p "$REPO/hpc_experiments/runs"
mkdir -p "$REPO/scripts/workbench/simon_workbench"

echo ""
echo "========================================="
echo "  Setup complete!"
echo "========================================="
echo ""
echo "Directory structure:"
echo "  $REPO/"
echo "  ├── .venv/                    (Python environment)"
echo "  ├── data/"
echo "  │   └── gnn_ready_source_sink/  (upload data here)"
echo "  ├── hpc_experiments/"
echo "  │   ├── configs/              (experiment configs)"
echo "  │   ├── jobs/                 (LSF job scripts)"
echo "  │   ├── logs/                 (stdout/stderr logs)"
echo "  │   └── runs/                 (training outputs)"
echo "  └── scripts/workbench/"
echo "      └── simon_workbench/      (upload code here)"
echo ""
echo "Next: Transfer data and code (see guide)."

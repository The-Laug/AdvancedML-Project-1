#!/bin/sh
### ── LSF options ──────────────────────────────────────────
#BSUB -J vae_10x3
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
#BSUB -oo vae_10x3_%J.out
#BSUB -eo vae_10x3_%J.err
#BSUB -B
#BSUB -N
### ── end of LSF options ──────────────────────────────────

# Move log files to logs dir after LSF creates them in cwd
trap 'mv -f vae_10x3_*.out vae_10x3_*.err "$REPO"/logs/ 2>/dev/null' EXIT

echo "========================================="
echo "Job: vae_10x3"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Date: $(date)"
echo "========================================="

# ── Environment setup ─────────────────────────────────────
export REPO=$(pwd)
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

# Activate virtual environment if present
if [ -f "$REPO/.venv/bin/activate" ]; then
    source "$REPO/.venv/bin/activate"
    echo "Activated .venv"
elif [ -f "$REPO/venv/bin/activate" ]; then
    source "$REPO/venv/bin/activate"
    echo "Activated venv"
fi

# Verify GPU is accessible
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

# ── Create output directory ───────────────────────────────
OUTPUT_DIR=$REPO/hpc_runs/vae_10x3
mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPO/logs"

# ── Run training and evaluation script ────────────────────
python3 "$REPO/train_eval_10x3.py" \
    --device cuda \
    --latent-dim 10 \
    --epochs 10 \
    --batch-size 32 \
    --runs 10 \
    --output "$OUTPUT_DIR/elbo_results.json" \
    --data-root "$REPO/data/"

echo ""
echo "========================================="
echo "Job vae_10x3 finished at $(date)"
echo "========================================="

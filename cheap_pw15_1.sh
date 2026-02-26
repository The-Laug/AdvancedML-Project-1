#!/bin/sh
### ── LSF options ──────────────────────────────────────────
#BSUB -J cheap_pw15_1
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 04:00
#BSUB -oo cheap_pw15_1_%J.out
#BSUB -eo cheap_pw15_1_%J.err
#BSUB -B
#BSUB -N
### ── end of LSF options ──────────────────────────────────

# Move log files to logs dir after LSF creates them in cwd
trap 'mv -f cheap_pw15_1_*.out cheap_pw15_1_*.err /zhome/fa/9/147496/source_sink_gnn/hpc_experiments/logs/ 2>/dev/null' EXIT

echo "========================================="
echo "Job: cheap_pw15_1"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Date: $(date)"
echo "========================================="

# ── Environment setup ─────────────────────────────────────
export REPO=/zhome/fa/9/147496/source_sink_gnn
export PATH="$HOME/.local/bin:$PATH"
cd $REPO

# Activate uv-managed virtual environment
source $REPO/.venv/bin/activate

# Verify GPU is accessible
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')"

# ── Create output directory ───────────────────────────────
OUTPUT_DIR=$REPO/hpc_experiments/runs/cheap_pw15_1
mkdir -p $OUTPUT_DIR

# ── Run training ──────────────────────────────────────────
python3 $REPO/scripts/workbench/simon_workbench/hpc/train_hpc.py \
    --config $REPO/hpc_experiments/configs/cheap_pw15_1.json \
    --output-dir $OUTPUT_DIR \
    --data-dir $REPO/data/gnn_ready_source_sink

echo ""
echo "========================================="
echo "Job cheap_pw15_1 finished at $(date)"
echo "========================================="

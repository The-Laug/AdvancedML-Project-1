#!/bin/bash
#SBATCH --job-name=vae_10x3
#SBATCH --output=logs/vae_10x3_%j.out
#SBATCH --error=logs/vae_10x3_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# -----------------------------------------------------------------------
# Train 10 VAE models for each of the 3 priors and save test ELBO results
# -----------------------------------------------------------------------

# Create log directory if it doesn't exist
mkdir -p logs

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Start time:  $(date)"
echo "Working dir: $PWD"

# Load modules (adjust to your cluster's module system)
# module load python/3.10
# module load cuda/12.1

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated .venv"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated venv"
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Run the training and evaluation script
python train_eval_10x3.py \
    --device cuda \
    --latent-dim 10 \
    --epochs 10 \
    --batch-size 32 \
    --runs 10 \
    --output elbo_results.json \
    --data-root data/

echo ""
echo "Finished at: $(date)"
echo "Exit code: $?"

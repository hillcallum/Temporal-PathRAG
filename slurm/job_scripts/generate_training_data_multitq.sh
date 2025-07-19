#!/bin/bash
#SBATCH --job-name=gen_data_multitq
#SBATCH --partition=gpgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@imperial.ac.uk
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/gen_multitq_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/gen_multitq_%j.err

# Note: No GPU requested - this is CPU-only workload

echo "Generating Training Data for MultiTQ Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Dataset: MultiTQ"
echo

# Set unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

# Set cache directory to bitbucket to avoid home directory quota issues
export TEMPORAL_PATHRAG_CACHE=/vol/bitbucket/$USER/.temporal_pathrag_cache

# Ensure output directory exists
mkdir -p /vol/bitbucket/"$USER"/temporal_pathrag_logs

# Change to project directory
cd /vol/bitbucket/"$USER"/Temporal_PathRAG || exit 1

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "Activating venv"
    source venv/bin/activate
else
    echo "Error: Virtual environment not found!"
    exit 1
fi

# Verify Python
which python
python --version

# Run the training data generation script with checkpointing
echo "Generating MultiTQ dataset training data (fast version)"
python -u scripts/training/generate_training_data_fast.py \
    --dataset MultiTQ \
    --num-quadruplet 350000 \
    --num-contrastive 50000 \
    --num-reconstruction 50000 \
    --output-dir data/training/MultiTQ \
    
echo
echo "Job completed at: $(date)"
echo "Exit code: $?"
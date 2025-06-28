#!/bin/bash
# SLURM job script for dataset processing - DO NOT RUN DIRECTLY
#SBATCH --job-name=process_datasets
#SBATCH --partition=gpgpuC
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/process_datasets_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/process_datasets_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cih124@ic.ac.uk

echo "Starting dataset processing job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Create log directory
mkdir -p /vol/bitbucket/${USER}/temporal_pathrag_logs

# Change to project directory
cd /vol/bitbucket/${USER}/Temporal_PathRAG

# Set up Python environment
export PYTHONUSERBASE="/vol/bitbucket/${USER}/python_packages"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH"

# Run dataset processing
echo "Running dataset processing pipeline"
python3 scripts/process_raw_datasets.py --clean

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Dataset processing completed successfully"
else
    echo "Dataset processing failed with exit code $EXIT_CODE"
fi

echo "Job finished at: $(date)"
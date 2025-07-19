#!/bin/bash
# Submit job to generate training data for temporal embeddings

# Configuration
export DATASET="${1:-combined}"  # Default to combined dataset

echo "Generate Training Data For Temporal Embeddings"
echo "Dataset: $DATASET"
echo ""

if [ "$DATASET" == "combined" ]; then
    echo "This will generate training data for both datasets:"
    echo "- MultiTQ: around 450K examples (350K quadruplet + 50K contrastive + 50K reconstruction)"
    echo "- TimeQuestions: around 240K examples (180K quadruplet + 30K contrastive + 30K reconstruction)"
    echo "- Total combined: around 690K examples"
elif [ "$DATASET" == "MultiTQ" ]; then
    echo "This will generate around 450K examples for MultiTQ"
elif [ "$DATASET" == "TimeQuestions" ]; then
    echo "This will generate around 240K examples for TimeQuestions"
fi

echo ""
# Confirm before submission
read -p "Submit data generation job? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Job submission cancelled"
    exit 1
fi

# Submit to cluster
echo "Submitting job"

# Get absolute path to job script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/../../job_scripts/generate_training_data.sh"

# Use the job script with DATASET environment variable
JOB_ID=$(sbatch --export=ALL,DATASET=$DATASET "${JOB_SCRIPT}" | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "Job submitted successfully with ID: $JOB_ID"
    echo ""
    echo "Monitor progress:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f /vol/bitbucket/${USER}/temporal_pathrag_logs/generate_training_data_${JOB_ID}.out"
    echo ""
else
    echo "Failed to submit job"
fi
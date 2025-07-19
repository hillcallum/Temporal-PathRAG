#!/bin/bash
# Submit separate training data generation jobs for each dataset

echo "Submitting training data generation jobs"
echo "These jobs will run on CPU-only nodes (no GPU required)"
echo

# Submit MultiTQ job
echo "Submitting MultiTQ data generation"
JOB1=$(sbatch ../../job_scripts/generate_training_data_multitq.sh | awk '{print $4}')
echo "MultiTQ job submitted with ID: $JOB1"

# Submit TimeQuestions job
echo "Submitting TimeQuestions data generation"
JOB2=$(sbatch ../../job_scripts/generate_training_data_timequestions.sh | awk '{print $4}')
echo "TimeQuestions job submitted with ID: $JOB2"

echo
echo "Both jobs submitted successfully"
echo "Monitor progress with: squeue -u $USER"
echo
echo "Progress can be tracked in real-time:"
echo "  MultiTQ: tail -f /vol/bitbucket/$USER/temporal_pathrag_logs/gen_multitq_${JOB1}.err"
echo "  TimeQuestions: tail -f /vol/bitbucket/$USER/temporal_pathrag_logs/gen_timeq_${JOB2}.err"
echo
echo "After both jobs complete, combine datasets with:"
echo "  python scripts/training/combine_datasets.py \\"
echo "    --input-dirs data/training/MultiTQ data/training/TimeQuestions \\"
echo "    --output-dir data/training/combined"
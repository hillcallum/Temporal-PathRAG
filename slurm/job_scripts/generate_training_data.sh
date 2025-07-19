#!/bin/bash
#SBATCH --job-name=generate_training_data
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/generate_training_data_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/generate_training_data_%j.err
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cih124

# Exit on error
set -e

echo "Generating Training Data for Temporal Embeddings"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Dataset: ${DATASET}"
echo ""

# Skip module loading if not available
if command -v module &> /dev/null; then
    module load python/3.10.12-gcc-13.1.0
    module load cuda/11.8
else
    echo "Module system not available, using existing environment"
fi

# Set up environment
export PROJECT_DIR="/vol/bitbucket/cih124/Temporal_PathRAG"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export HF_HOME="/vol/bitbucket/cih124/.cache/huggingface"

# Override cache directory to use /vol/bitbucket which has more space
export TEMPORAL_PATHRAG_CACHE="/vol/bitbucket/cih124/.temporal_pathrag_cache"
mkdir -p "${TEMPORAL_PATHRAG_CACHE}"

cd "${PROJECT_DIR}"

# Try different Python environment options
if [ -f "/vol/bitbucket/cih124/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "Found conda, activating"
    source /vol/bitbucket/cih124/miniconda3/etc/profile.d/conda.sh
    conda activate temporal_pathrag 2>/dev/null || conda activate pathrag 2>/dev/null || conda activate base
elif [ -f "/vol/bitbucket/cih124/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "Found anaconda, activating"
    source /vol/bitbucket/cih124/anaconda3/etc/profile.d/conda.sh
    conda activate temporal_pathrag 2>/dev/null || conda activate pathrag 2>/dev/null || conda activate base
elif [ -d "${PROJECT_DIR}/venv" ]; then
    echo "Activating venv"
    source ${PROJECT_DIR}/venv/bin/activate
elif [ -d "${PROJECT_DIR}/.venv" ]; then
    echo "Activating .venv"
    source ${PROJECT_DIR}/.venv/bin/activate
else
    echo "Warning - no virtual environment found, using system Python"
    # Try to find Python in common locations
    export PATH="/usr/local/bin:/usr/bin:$PATH"
fi

# Verify Python is available
which python
python --version

# Create output directories
mkdir -p "${PROJECT_DIR}/data/training/${DATASET}"
mkdir -p "/vol/bitbucket/cih124/temporal_pathrag_logs"

# Generate training data based on dataset
if [ "${DATASET}" == "MultiTQ" ]; then
    echo "Generating MultiTQ training data"
    echo "Total quadruplets in graph: 461,329"
    python scripts/training/generate_embedding_training_data.py \
        --dataset MultiTQ \
        --output-dir data/training \
        --num-quadruplet 350000 \
        --num-contrastive 50000 \
        --num-reconstruction 50000
        
elif [ "${DATASET}" == "TimeQuestions" ]; then
    echo "Generating TimeQuestions training data"
    echo "Total quadruplets in graph: 240,597"
    python scripts/training/generate_embedding_training_data.py \
        --dataset TimeQuestions \
        --output-dir data/training \
        --num-quadruplet 180000 \
        --num-contrastive 30000 \
        --num-reconstruction 30000
        
elif [ "${DATASET}" == "combined" ]; then
    echo "Generating combined dataset training data"
    echo "Processing MultiTQ first"
    python scripts/training/generate_embedding_training_data.py \
        --dataset MultiTQ \
        --output-dir data/training \
        --num-quadruplet 350000 \
        --num-contrastive 50000 \
        --num-reconstruction 50000
        
    echo ""
    echo "Processing TimeQuestions"
    python scripts/training/generate_embedding_training_data.py \
        --dataset TimeQuestions \
        --output-dir data/training \
        --num-quadruplet 180000 \
        --num-contrastive 30000 \
        --num-reconstruction 30000
        
    echo ""
    echo "Combining datasets"
    python -c "
import json
from pathlib import Path

# Load both datasets
data_dir = Path('data/training')
combined_dir = data_dir / 'combined'
combined_dir.mkdir(exist_ok=True)

for split in ['train', 'validation', 'test']:
    combined_data = []
    
    # Load MultiTQ
    multitq_file = data_dir / 'MultiTQ' / f'{split}.json'
    if multitq_file.exists():
        with open(multitq_file) as f:
            multitq_data = json.load(f)
            combined_data.extend(multitq_data)
            print(f'Added {len(multitq_data)} examples from MultiTQ {split}')
    
    # Load TimeQuestions
    tq_file = data_dir / 'TimeQuestions' / f'{split}.json'
    if tq_file.exists():
        with open(tq_file) as f:
            tq_data = json.load(f)
            combined_data.extend(tq_data)
            print(f'Added {len(tq_data)} examples from TimeQuestions {split}')
    
    # Save combined
    with open(combined_dir / f'{split}.json', 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f'Total {split} examples: {len(combined_data)}')

# Create combined metadata
metadata = {
    'dataset_name': 'combined',
    'datasets_included': ['MultiTQ', 'TimeQuestions'],
    'num_quadruplets': 701926,  # 461329 + 240597
    'splits': {
        'train': len(json.load(open(combined_dir / 'train.json'))),
        'validation': len(json.load(open(combined_dir / 'validation.json'))),
        'test': len(json.load(open(combined_dir / 'test.json')))
    }
}

with open(combined_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'\nCombined metadata saved')
"
else
    echo "Error: Unknown dataset ${DATASET}"
    exit 1
fi

echo ""
echo "Training data generation complete!"
echo "End time: $(date)"
echo "Output location: ${PROJECT_DIR}/data/training/${DATASET}"
echo ""

# List generated files
echo "Generated files:"
ls -la "${PROJECT_DIR}/data/training/${DATASET}/"
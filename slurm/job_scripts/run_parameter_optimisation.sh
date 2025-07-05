#!/bin/bash
# Internal SLURM job script for parameter optimisation - DO NOT RUN DIRECTLY
#SBATCH --job-name=temporal_param_opt
#SBATCH --partition=gpgpuC
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/param_opt_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/param_opt_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cih124

echo "Starting Temporal PathRAG Parameter Optimisation on GPU cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Create log directory
mkdir -p /vol/bitbucket/${USER}/temporal_pathrag_logs

# Load CUDA
echo "Loading CUDA"
. /vol/cuda/12.0.0/setup.sh

# Set up Python environment
echo "Setting up Python environment"
cd /vol/bitbucket/${USER}

# Create/activate virtual environment (Ubuntu 24.04 compatible)
if [ ! -d "temporal_pathrag_env" ]; then
    echo "Creating virtual environment with python3-full"
    # Use system python3 with --break-system-packages for cluster environment
    python3 -m venv temporal_pathrag_env --system-site-packages
    
    if [ $? -ne 0 ]; then
        echo "venv creation failed, trying alternative approach"
        # Alternative: use system python with --break-system-packages
        mkdir -p temporal_pathrag_env/bin
        ln -sf /usr/bin/python3 temporal_pathrag_env/bin/python
        ln -sf /usr/bin/pip3 temporal_pathrag_env/bin/pip
    fi
fi

echo "Activating environment"
if [ -f "temporal_pathrag_env/bin/activate" ]; then
    source temporal_pathrag_env/bin/activate
else
    # Fallback to system python
    export PATH="/usr/bin:$PATH"
fi

cd Temporal_PathRAG

echo "Installing packages to /vol/bitbucket (avoid disk quota)"
# Install to bitbucket directory to avoid home directory quota
export PYTHONUSERBASE="/vol/bitbucket/${USER}/python_packages"
mkdir -p "$PYTHONUSERBASE"

python3 -m pip install --user --break-system-packages --upgrade pip
python3 -m pip install --user --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install --user --break-system-packages transformers networkx numpy pandas scipy tqdm
python3 -m pip install --user --break-system-packages sentence-transformers
python3 -m pip install --user --break-system-packages python-dotenv pyyaml requests

# Add to Python path
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH"

# Set HuggingFace cache to avoid home directory quota issues
export HF_HOME="/vol/bitbucket/${USER}/hf_cache"
export TRANSFORMERS_CACHE="/vol/bitbucket/${USER}/hf_cache"
export HF_DATASETS_CACHE="/vol/bitbucket/${USER}/hf_cache"
mkdir -p "/vol/bitbucket/${USER}/hf_cache"

echo "Package installation completed"

echo "Python location: $(which python)"

# Check GPU availability
echo "Checking GPU availability"
nvidia-smi
echo ""

# Test CUDA setup
echo "Testing CUDA and PyTorch"
python3 -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU device: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
except ImportError:
    print('PyTorch not installed yet, will install now')
"

# Test required packages
echo "Testing required packages"
python3 -c "
try:
    import networkx as nx
    print('NetworkX: OK')
    import torch
    print(f'PyTorch: {torch.__version__}')
    import transformers
    print('Transformers: OK')
    import numpy as np
    print('NumPy: OK')
    import scipy
    print('SciPy: OK')
    print('Essential packages imported successfully!')
except ImportError as e:
    print(f'Import error: {e}')
    print('Continuing with available packages')
"

# Run Parameter Optimisation
echo ""
echo "Running Temporal PathRAG Parameter Optimisation"
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Force GPU usage if available
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Environment variables set for GPU usage"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Create results directory
mkdir -p test_results

# Run the parameter optimisation
echo "Starting parameter optimisation"
python3 scripts/testing/model_reuse_parameter_optimisation.py

OPTIMISATION_EXIT_CODE=$?

# Copy results to cluster-accessible location
if [ $OPTIMISATION_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Copying optimisation results to cluster logs"
    
    # Copy JSON reports if they exist
    if [ -f "test_results/model_reuse_parameter_optimisation_report.json" ]; then
        cp test_results/model_reuse_parameter_optimisation_report.json /vol/bitbucket/${USER}/temporal_pathrag_logs/param_opt_${SLURM_JOB_ID}_model.json
        echo "Optimisation report saved"
    fi
    
    if [ -f "test_results/temporal_path_retrieval_report.json" ]; then
        cp test_results/temporal_path_retrieval_report.json /vol/bitbucket/${USER}/temporal_pathrag_logs/param_opt_${SLURM_JOB_ID}_temporal.json
        echo "Temporal path retrieval report saved"
    fi
    
    echo ""
    echo "=== Parameter Optimisation Completed Successfully ==="
    echo "Results saved to /vol/bitbucket/${USER}/temporal_pathrag_logs/"
    
    # Show quick summary if available
    if [ -f "test_results/model_reuse_parameter_optimisation_report.json" ]; then
        echo ""
        echo "Quick Summary (Optimisation):"
        python3 -c "
import json
try:
    with open('test_results/model_reuse_parameter_optimisation_report.json', 'r') as f:
        report = json.load(f)
    
    optimal = report.get('optimal_parameters', {})
    alpha = optimal.get('alpha', 'N/A')
    theta = optimal.get('theta', 'N/A')
    
    training = report.get('performance_metrics', {}).get('training', {})
    reliability = training.get('avg_reliability', 'N/A')
    success_rate = training.get('success_rate', 'N/A')
    ci_lower = training.get('reliability_ci_lower', 'N/A')
    ci_upper = training.get('reliability_ci_upper', 'N/A')
    
    validation = report.get('performance_metrics', {}).get('validation', {})
    generalisation = validation.get('generalisation', 'N/A')
    
    print(f'Optimal Alpha: {alpha}')
    print(f'Optimal Theta: {theta}')
    print(f'Reliability: {reliability}')
    print(f'95% CI: [{ci_lower}, {ci_upper}]')
    print(f'Success Rate: {success_rate}')
    print(f'Generalisation: {generalisation}')
    
except Exception as e:
    print(f'Could not parse results: {e}')
"
    fi
    
else
    echo ""
    echo "=== Parameter Optimisation Failed with exit code $OPTIMISATION_EXIT_CODE ==="
fi

echo ""
echo "Job finished at: $(date)"
# Internal SLURM job script - DO NOT RUN DIRECTLY 

#!/bin/bash
#SBATCH --job-name=temporal_pathrag_demo
#SBATCH --partition=gpgpuC
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/temporal_pathrag_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/temporal_pathrag_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cih124

echo "Starting Temporal PathRAG demo on GPU cluster"
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

python3 -m pip install --user --upgrade pip
python3 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install --user transformers networkx numpy pandas scipy tqdm
python3 -m pip install --user python-dotenv pyyaml requests

# Add to Python path
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH"

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
    print('Essential packages imported successfully!')
except ImportError as e:
    print(f'Import error: {e}')
    print('Continuing with available packages...')
"

# Run Temporal_PathRAG
echo ""
echo "Running Temporal PathRAG"
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"
echo "Files in src/: $(ls -la src/)"

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Force GPU usage if available
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Environment variables set for GPU usage"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python3 src/main.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Temporal PathRAG Completed Successfully ==="
else
    echo ""
    echo "=== Temporal PathRAG Failed with exit code $EXIT_CODE ==="
fi

echo "Job finished at: $(date)"
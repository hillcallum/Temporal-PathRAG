#!/bin/bash
# Interactive LLM Setup and Testing Script for Imperial GPU Cluster
# Run this with: salloc --gres=gpu:1 --time=2:00:00 --partition=gpgpuC

echo "=== Imperial GPU Cluster - Interactive LLM Setup ==="
echo "Run this script after getting an interactive session with:"
echo "salloc --gres=gpu:1 --time=2:00:00 --partition=gpgpuC --mem=32G"
echo ""

# Check if we're in an interactive session
if [ -z "$SLURM_JOB_ID" ]; then
    echo "ERROR: No SLURM_JOB_ID detected. Please run 'salloc' first:"
    echo "salloc --gres=gpu:1 --time=2:00:00 --partition=gpgpuC --mem=32G"
    exit 1
fi

echo "Interactive session detected: Job ID $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Setup environment
echo "=== Setting up environment ==="
cd /vol/bitbucket/${USER}/Temporal_PathRAG

# Load CUDA
echo "Loading CUDA"
. /vol/cuda/12.0.0/setup.sh

# Check GPU
echo "Checking GPU availability"
nvidia-smi
echo ""

# Set up Python environment variables
export PYTHONUSERBASE="/vol/bitbucket/${USER}/python_packages"
export HF_HOME="/vol/bitbucket/${USER}/hf_cache"
export TRANSFORMERS_CACHE="/vol/bitbucket/${USER}/hf_cache"
export HF_DATASETS_CACHE="/vol/bitbucket/${USER}/hf_cache"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "$PYTHONUSERBASE" "$HF_HOME"

echo "Environment variables set:"
echo "PYTHONUSERBASE: $PYTHONUSERBASE"
echo "HF_HOME: $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Function to test package installation
test_package() {
    local package=$1
    local import_name=$2
    echo -n "Testing $package "
    if python3 -c "import $import_name" 2>/dev/null; then
        echo "Ok"
        return 0
    else
        echo "Failed"
        return 1
    fi
}

# Function to install package with progress
install_package() {
    local package=$1
    echo "Installing $package"
    python3 -m pip install --user --break-system-packages --progress-bar on "$package"
    echo "$package installed"
}

# Test existing packages
echo "=== Testing existing packages ==="
test_package "PyTorch" "torch"
test_package "Transformers" "transformers"
test_package "NetworkX" "networkx"

# Install missing packages
echo ""
echo "=== Installing packages (if needed) ==="
pip install --user --break-system-packages --upgrade pip

# Install essential packages
install_package "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
install_package "transformers"
install_package "sentence-transformers"
install_package "ollama"

echo ""
echo "=== Testing LLM Setup ==="

# Test 1: Check PyTorch CUDA
echo "Test 1: PyTorch CUDA setup"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print('CUDA setup working')
else:
    print('CUDA not available')
"

echo ""
echo "Test 2: Test HuggingFace model download (small model first)"
python3 -c "
import os
from transformers import AutoTokenizer, AutoModel
import torch

print('Testing HuggingFace download with small model')
print(f'Cache directory: {os.environ.get(\"HF_HOME\", \"default\")}')

try:
    # Use a small model first to test download
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f'Downloading tokenizer for {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f'Downloading model for {model_name}')
    model = AutoModel.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        print('Moving model to GPU')
        model = model.to('cuda')
        print('Model loaded on GPU')
    else:
        print('Model loaded on CPU')
        
    print('Small model download and loading successful')
    
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "Test 3: Test Ollama setup"
python3 -c "
try:
    import ollama
    print('Ollama package imported successfully')
    
    # Test if ollama service is available
    # Note: This might not work on cluster without ollama service running
    
except Exception as e:
    print(f'Ollama test: {e}')
"

echo ""
echo "=== Running Tests ==="

echo "Testing LLaMA model download"
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'meta-llama/Llama-3.2-1B'  # Smaller model for testing
print(f'Attempting to download {model_name}')

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('Tokenizer downloaded')
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    print('Model downloaded')
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        print('Model moved to GPU')
    
    # Test inference
    inputs = tokenizer('Hello, how are you?', return_tensors='pt')
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'Test generation: {response}')
        
    print('LLaMA model setup successful!')
    
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "Testing Temporal PathRAG code"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 src/main.py

echo ""
echo "=== Debug Session Complete ==="
echo "If everything works here, we can use the batch scripts in the cluster_setup files"
echo "If there are issues, debug them interactively first"
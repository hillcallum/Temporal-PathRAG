#!/bin/bash
echo "Setting up PathRAG GPU environment"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create base environment
echo "Creating temporal-pathrag environment"
conda env create -f environment.yml

# Activate environment
echo "Activating environment"
eval "$(conda shell.bash hook)"
conda activate temporal-pathrag

# Replace CPU PyTorch with GPU version
echo "Installing GPU-enabled PyTorch"
conda remove pytorch torchvision torchaudio cpuonly --force -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install GPU-optimised FAISS
echo "Installing GPU-optimised FAISS"
pip uninstall faiss-cpu -y
pip install faiss-gpu>=1.7.0

# Install spaCy language model
echo "Installing spaCy English model"
python -m spacy download en_core_web_sm

echo ""
echo "GPU environment setup complete!"
echo "Environment: temporal-pathrag"
echo "Activate with: conda activate temporal-pathrag"
echo ""

# Test GPU availability
echo "Testing GPU setup"
python -c "
import torch
from src.utils.device import setup_device_and_logging, test_pathrag_operations

print('PathRAG GPU Test Results:')
print('=' * 50)
device = setup_device_and_logging()
print('=' * 50)

if torch.cuda.is_available():
    print('CUDA is available')
    print(f'GPU device count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    test_pathrag_operations()
else:
    print('CUDA not available')
    print('Check your NVIDIA drivers and CUDA installation')
"

echo ""
echo "Your PathRAG GPU environment is ready!"
echo "Next steps:"
echo "   1. conda activate temporal-pathrag" 
echo "   2. python src/main.py  # Test your PathRAG implementation"
echo "   3. Start building temporal-aware multi-hop QA!"
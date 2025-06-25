#!/bin/bash
echo "Setting up PathRAG local development environment (CPU)"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create environment
echo "Creating temporal-pathrag environment"
conda env create -f environment.yml

# Activate environment
echo "Activating environment"
eval "$(conda shell.bash hook)"
conda activate temporal-pathrag

# Ensure CPU-only FAISS (already specified in environment.yml)
echo "Verifying CPU-optimised packages"
pip install --upgrade faiss-cpu>=1.7.0

# Install spaCy language model
echo "Installing spaCy English model"
python -m spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data"
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
print('NLTK data downloaded')
"

echo ""
echo "Local CPU environment setup complete!"
echo "Environment: temporal-pathrag"
echo "Activate with: conda activate temporal-pathrag"
echo ""

# Test CPU setup
echo "Testing CPU setup"
python -c "
import torch
from src.utils.device import setup_device_and_logging, test_pathrag_operations

print('PathRAG CPU Test Results:')
print('=' * 50)
device = setup_device_and_logging()
print('=' * 50)

print('CPU environment ready')
test_pathrag_operations()
"

echo ""
echo "Your PathRAG local environment is ready!"
echo "Next steps:"
echo "   1. conda activate temporal-pathrag"
echo "   2. python test_imports.py  # Verify all packages"
echo "   3. python src/main.py     # Test your PathRAG implementation"
echo "   4. Start building temporal-aware multi-hop QA!"
echo ""
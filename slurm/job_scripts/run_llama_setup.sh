#!/bin/bash
#SBATCH --job-name=llama_setup
#SBATCH --partition=gpgpuC
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/llama_setup_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/llama_setup_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cih124

echo "LLaMA 3.2 Setup on Imperial GPU Cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"
echo ""

# Create log directory
mkdir -p /vol/bitbucket/${USER}/temporal_pathrag_logs

# Load CUDA
echo "Loading CUDA"
. /vol/cuda/12.0.0/setup.sh

# Setup environment
cd /vol/bitbucket/${USER}/Temporal_PathRAG

# Environment variables
export PYTHONUSERBASE="/vol/bitbucket/${USER}/python_packages"
export HF_HOME="/vol/bitbucket/${USER}/hf_cache"
export TRANSFORMERS_CACHE="/vol/bitbucket/${USER}/hf_cache"
export HF_DATASETS_CACHE="/vol/bitbucket/${USER}/hf_cache"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mkdir -p "$PYTHONUSERBASE" "$HF_HOME"

echo "Environment setup complete"
echo "Cache directory: $HF_HOME"
echo "Python packages: $PYTHONUSERBASE"
echo ""

# Install packages with progress monitoring
echo "Installing Required Packages"
pip install --user --break-system-packages --upgrade pip

# Install with checkpointing - install one at a time to catch failures
install_with_checkpoint() {
    local package=$1
    local checkpoint_file="/vol/bitbucket/${USER}/.install_checkpoint_$(echo $package | sed 's/[^a-zA-Z0-9]/_/g')"
    
    if [ -f "$checkpoint_file" ]; then
        echo "$package already installed (checkpoint found)"
        return 0
    fi
    
    echo "Installing $package"
    if python3 -m pip install --user --break-system-packages --progress-bar on $package; then
        echo "$package installed successfully"
        touch "$checkpoint_file"
        return 0
    else
        echo "Failed to install $package"
        return 1
    fi
}

# Install packages with checkpointing
install_with_checkpoint "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
install_with_checkpoint "transformers"
install_with_checkpoint "sentence-transformers"
install_with_checkpoint "accelerate"
install_with_checkpoint "bitsandbytes"
install_with_checkpoint "ollama"

echo ""
echo "Testing Package Installation"
python3 -c "
import sys
sys.path.insert(0, '/vol/bitbucket/${USER}/python_packages/lib/python3.12/site-packages')

try:
    import torch
    print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')
    
    import transformers
    print(f'Transformers {transformers.__version__}')
    
    import sentence_transformers
    print('Sentence Transformers')
    
    import accelerate
    print('Accelerate')
    
    import bitsandbytes
    print('BitsAndBytes')
    
    import ollama
    print('Ollama')
    
    print('All packages installed successfully')
    
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "Package installation failed"
    exit 1
fi

echo ""
echo "Setting up LLaMA 3.2"

# Create LLaMA setup script
cat > /tmp/llama_setup.py << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import sys

def setup_llama_model():
    print("Setting up LLaMA model")
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"Model: {model_name}")
    print(f"Cache directory: {os.environ.get('HF_HOME', 'default')}")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {gpu_memory:.1f} GB")
        
        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None
        print("No GPU available, using CPU")
    
    try:
        print("Downloading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer downloaded")
        
        print("Downloading model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("Model downloaded and loaded")
        
        # Test inference
        print("Testing inference")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Test inference successful")
        print(f"Response: {response.split('assistant')[-1].strip()}")
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "model_size": f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters",
            "quantization": "4-bit" if quantization_config else "none",
            "gpu_memory_used": f"{torch.cuda.memory_allocated() / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"
        }
        
        import json
        with open("/vol/bitbucket/${USER}/llama_model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print("LLaMA setup completed successfully")
        return True
        
    except Exception as e:
        print(f"Error setting up LLaMA: {e}")
        return False

if __name__ == "__main__":
    success = setup_llama_model()
    sys.exit(0 if success else 1)
EOF

# Run LLaMA setup
echo "Running LLaMA setup"
export PYTHONPATH="/vol/bitbucket/${USER}/python_packages/lib/python3.12/site-packages:$PYTHONPATH"
python3 /tmp/llama_setup.py

LLAMA_EXIT_CODE=$?

echo ""
echo "Setting up Ollama Alternative"

# Start ollama service in background
echo "Starting Ollama service"
ollama serve &
OLLAMA_PID=$!
sleep 10

# Pull model
echo "Pulling Llama 3.2 1B model via Ollama"
ollama pull llama3.2:1b

# Test ollama
echo "Testing Ollama"
echo "What is machine learning?" | ollama run llama3.2:1b

# Stop ollama service
kill $OLLAMA_PID 2>/dev/null || true

echo ""
echo "Setup Summary"
echo "Job ID: $SLURM_JOB_ID"
echo "Completion time: $(date)"

if [ $LLAMA_EXIT_CODE -eq 0 ]; then
    echo "LLaMA setup successful"
    echo "Model info saved to: /vol/bitbucket/${USER}/llama_model_info.json"
else
    echo "LLaMA setup failed"
fi

echo ""
echo "Job Complete"
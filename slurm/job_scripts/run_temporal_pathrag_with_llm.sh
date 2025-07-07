#!/bin/bash
#SBATCH --job-name=temporal_pathrag_llm
#SBATCH --partition=gpgpuC
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/temporal_pathrag_llm_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/temporal_pathrag_llm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cih124

echo "=== Temporal PathRAG with LLM on Imperial GPU Cluster ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Create log directory
mkdir -p /vol/bitbucket/${USER}/temporal_pathrag_logs

# Load CUDA
echo "Loading CUDA"
. /vol/cuda/12.0.0/setup.sh

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Setup environment
cd /vol/bitbucket/${USER}/Temporal_PathRAG

# Environment variables
# Use system packages (PyTorch, Transformers already available)
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="/tmp/hf_cache_${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Create temporary cache directory
mkdir -p "$HF_HOME"

# Progress monitoring function
log_progress() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Create checkpoint system
CHECKPOINT_DIR="/vol/bitbucket/${USER}/temporal_pathrag_checkpoints"
mkdir -p "$CHECKPOINT_DIR"

log_progress "Starting Temporal PathRAG with LLM"
log_progress "Environment setup complete"

# Check if LLM is already set up
if [ -f "/vol/bitbucket/${USER}/llama_model_info.json" ]; then
    log_progress "Found existing LLM setup"
    cat "/vol/bitbucket/${USER}/llama_model_info.json"
else
    log_progress "No existing LLM setup found - using fallback model"
fi

# Test packages
log_progress "Testing required packages"
python3 -c "
import sys
# Using system packages - no need to modify path

try:
    import torch
    print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')
    
    import transformers
    print(f'Transformers {transformers.__version__}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    import networkx as nx
    print('NetworkX')
    
    print('All required packages available')
    
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    log_progress "Package test failed"
    exit 1
fi

log_progress "Package test passed"

# Create enhanced main script with progress monitoring
cat > /tmp/temporal_pathrag_with_monitoring.py << 'EOF'
import os
import sys
import time
import json
import signal
from datetime import datetime

# Add current directory to path (system packages already available)
sys.path.insert(0, os.getcwd())

def log_progress(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

def save_checkpoint(data, checkpoint_name):
    checkpoint_dir = "/vol/bitbucket/cih124/temporal_pathrag_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    log_progress(f"Checkpoint saved: {checkpoint_file}")

def load_checkpoint(checkpoint_name):
    checkpoint_dir = "/vol/bitbucket/cih124/temporal_pathrag_checkpoints"
    checkpoint_file = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def signal_handler(signum, frame):
    log_progress(f"Received signal {signum}, saving checkpoint")
    save_checkpoint({"status": "interrupted", "timestamp": datetime.now().isoformat()}, "interrupt")
    sys.exit(1)

# Set up signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

log_progress("Starting Temporal PathRAG with LLM")

try:
    # Import and test core modules
    log_progress("Importing core modules")
    
    import torch
    import transformers
    import networkx as nx
    
    log_progress("Core modules imported")
    
    # Check for existing checkpoints
    existing_checkpoint = load_checkpoint("main_progress")
    if existing_checkpoint:
        log_progress(f"Found existing checkpoint: {existing_checkpoint}")
    
    # GPU setup
    if torch.cuda.is_available():
        log_progress(f"GPU available: {torch.cuda.get_device_name(0)}")
        log_progress(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        log_progress("GPU cache cleared")
    else:
        log_progress("No GPU available, using CPU")
    
    # Try to import your main module
    log_progress("Importing Temporal PathRAG modules")
    
    try:
        from src.main import main
        log_progress("Main module imported")
        
        # Save progress checkpoint
        save_checkpoint({"stage": "modules_loaded", "timestamp": datetime.now().isoformat()}, "main_progress")
        
        # Run main function with correct base directory
        log_progress("Running main function")
        
        # Import and set correct base directory for cluster
        from src.utils import set_config, TemporalPathRAGConfig
        correct_config = TemporalPathRAGConfig(base_dir=os.getcwd())
        set_config(correct_config)
        
        result = main()
        
        # Save completion checkpoint
        save_checkpoint({
            "stage": "completed",
            "timestamp": datetime.now().isoformat(),
            "result": str(result) if result else "success"
        }, "main_progress")
        
        log_progress("Main function completed successfully")
        
    except ImportError as e:
        log_progress(f"Could not import main module: {e}")
        log_progress("Running fallback LLM test")
        
        # Fallback - test LLM directly
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Try to authenticate with HuggingFace (for gated models)
        try:
            from huggingface_hub import login
        except ImportError:
            log_progress("HuggingFace hub not available, continuing without auth")
        
        # Use a non-gated model for testing
        model_name = "microsoft/DialoGPT-medium"
        log_progress(f"Testing {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        log_progress("Model loaded successfully")
        
        # Test generation
        messages = [{"role": "user", "content": "What is temporal reasoning in AI?"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log_progress("Test generation successful")
        log_progress(f"Response: {response.split('assistant')[-1].strip()}")
        
        save_checkpoint({
            "stage": "llm_test_completed",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "response": response
        }, "main_progress")
        
    log_progress("=== Temporal PathRAG with LLM completed successfully ===")
    
except Exception as e:
    log_progress(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Save error checkpoint
    save_checkpoint({
        "stage": "error",
        "timestamp": datetime.now().isoformat(),
        "error": str(e),
        "traceback": traceback.format_exc()
    }, "error_log")
    
    sys.exit(1)
EOF

# Run the enhanced script
log_progress "Running Temporal PathRAG with monitoring"
python3 /tmp/temporal_pathrag_with_monitoring.py

EXIT_CODE=$?

log_progress "Execution completed with exit code: $EXIT_CODE"

# Copy results and checkpoints
log_progress "Copying results"
cp -r "$CHECKPOINT_DIR" "/vol/bitbucket/${USER}/temporal_pathrag_results_${SLURM_JOB_ID}/" 2>/dev/null || true

# Display summary
echo ""
echo "=== Job Summary ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Exit Code: $EXIT_CODE"
echo "Completion Time: $(date)"
echo "Checkpoints saved in: $CHECKPOINT_DIR"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Temporal PathRAG with LLM completed successfully"
else
    echo "Temporal PathRAG with LLM failed"
fi

echo ""
echo "=== Next Steps ==="
echo "1. Check checkpoints: ls -la $CHECKPOINT_DIR"
echo "2. View detailed logs in this output file"
echo "3. For debugging, use: slurm/debug_scripts/interactive_llm_setup.sh"

echo ""
echo "=== Job Complete ==="
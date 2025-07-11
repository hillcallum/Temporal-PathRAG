#!/bin/bash
#SBATCH --job-name=eval_ollama
#SBATCH --partition=gpgpuC
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/vol/bitbucket/%u/temporal_pathrag_logs/eval_ollama_%j.out
#SBATCH --error=/vol/bitbucket/%u/temporal_pathrag_logs/eval_ollama_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cih124

echo "Temporal PathRAG Evaluation with Ollama"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Create directories
mkdir -p /vol/bitbucket/${USER}/temporal_pathrag_logs
mkdir -p /vol/bitbucket/${USER}/ollama_models
mkdir -p /vol/bitbucket/${USER}/evaluation_results

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
export PYTHONUSERBASE="/vol/bitbucket/${USER}/python_packages"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.12/site-packages:$(pwd):$PYTHONPATH"

# Ollama settings
export OLLAMA_MODELS="/vol/bitbucket/${USER}/ollama_models"
export OLLAMA_HOST="0.0.0.0:11434"

# HuggingFace cache
export HF_HOME="/vol/bitbucket/${USER}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME"

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create directories
mkdir -p "$OLLAMA_MODELS"
mkdir -p "$HF_HOME"

# Progress monitoring
log_progress() {
    echo "[$(date '+%H:%M:%S')] $1"
}

log_progress "Environment setup complete"

# Install Ollama if not already installed
if ! command -v ollama &> /dev/null; then
    log_progress "Installing Ollama"
    curl -fsSL https://ollama.ai/install.sh | sh
else
    log_progress "Ollama already installed"
fi

# Start Ollama service in background
log_progress "Starting Ollama service"
ollama serve &
OLLAMA_PID=$!
sleep 10  # Wait for service to start

# Function to stop Ollama on exit
cleanup() {
    log_progress "Cleaning up Ollama service"
    kill $OLLAMA_PID 2>/dev/null || true
}
trap cleanup EXIT

# Check Ollama is running
log_progress "Checking Ollama service"
curl -s http://localhost:11434/api/tags || {
    log_progress "Ollama service failed to start"
    exit 1
}

# Use default model llama3.2:3b
MODEL_NAME="llama3.2:3b"
log_progress "Using default model: $MODEL_NAME"

# List available models
AVAILABLE_MODELS=$(ollama list | tail -n +2 | awk '{print $1}')
if echo "$AVAILABLE_MODELS" | grep -q "^${MODEL_NAME}$"; then
    log_progress "Model $MODEL_NAME already available"
else
    log_progress "Downloading model $MODEL_NAME (this may take a while)"
    ollama pull $MODEL_NAME
fi

# Create evaluation script
cat > /tmp/run_evaluation.py << 'EOF'
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

def log_progress(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

# Parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MultiTQ', help='Dataset to evaluate')
parser.add_argument('--baselines', nargs='+', default=['vanilla_llm', 'temporal_pathrag'], 
                   help='Baselines to run')
parser.add_argument('--max-questions', type=int, default=100, help='Max questions to evaluate')
parser.add_argument('--model', type=str, default='llama3.2:3b', help='Ollama model to use')
args = parser.parse_args()

log_progress(f"Configuration: {args}")

try:
    # Set up environment for Ollama
    log_progress("Configuring for Ollama")
    os.environ['LOCAL_LLM_ENABLED'] = 'true'
    os.environ['LOCAL_LLM_MODEL'] = args.model
    os.environ['LOCAL_LLM_HOST'] = 'localhost'
    os.environ['LOCAL_LLM_PORT'] = '11434'
    
    # Test Ollama connection
    log_progress("Testing Ollama connection")
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            log_progress(f"Available Ollama models: {[m['name'] for m in models]}")
        else:
            log_progress("Warning: Could not list Ollama models")
    except Exception as e:
        log_progress(f"Warning: Ollama connection test failed: {e}")
    
    # Import evaluation modules
    log_progress("Importing evaluation modules")
    from evaluation.baseline_runners import run_baseline_comparison
    from src.utils.dataset_loader import get_cache_info
    
    # Show cache info
    cache_info = get_cache_info()
    log_progress(f"Dataset cache: {cache_info}")
    
    # Run evaluation
    log_progress(f"Starting evaluation on {args.dataset}")
    log_progress(f"Baselines: {args.baselines}")
    log_progress(f"Max questions: {args.max_questions}")
    
    start_time = time.time()
    
    results = run_baseline_comparison(
        dataset_name=args.dataset,
        baselines=args.baselines,
        max_questions=args.max_questions,
        output_dir=Path(f"/vol/bitbucket/{os.environ['USER']}/evaluation_results/ollama_{args.model.replace(':', '_')}")
    )
    
    elapsed_time = time.time() - start_time
    log_progress(f"Evaluation completed in {elapsed_time:.1f} seconds")
    
    # Print results summary
    log_progress("\nSummary")
    for baseline_name, baseline_results in results.items():
        log_progress(f"\n{baseline_name}:")
        
        if 'error' in baseline_results:
            log_progress(f"ERROR: {baseline_results['error']}")
        elif 'metrics' in baseline_results:
            metrics = baseline_results['metrics']
            log_progress(f"Exact Match: {metrics.exact_match:.3f}")
            log_progress(f"F1 Score: {metrics.f1_score:.3f}")
            log_progress(f"Temporal Accuracy: {metrics.temporal_accuracy:.3f}")
            log_progress(f"Avg Time: {metrics.avg_retrieval_time + metrics.avg_reasoning_time:.3f}s")
    
    # Save summary
    summary_path = Path(f"/vol/bitbucket/{os.environ['USER']}/evaluation_results/ollama_summary_{args.model.replace(':', '_')}.json")
    summary = {
        'dataset': args.dataset,
        'model': args.model,
        'baselines': args.baselines,
        'max_questions': args.max_questions,
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().isoformat(),
        'results': {
            name: {
                'exact_match': res.get('metrics', {}).exact_match if 'metrics' in res else None,
                'f1_score': res.get('metrics', {}).f1_score if 'metrics' in res else None,
                'error': res.get('error')
            }
            for name, res in results.items()
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_progress(f"Summary saved to: {summary_path}")
    
except Exception as e:
    log_progress(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

log_progress("Evaluation script completed successfully")
EOF

# Run evaluation
log_progress "Starting evaluation with Ollama model: $MODEL_NAME"
python3 /tmp/run_evaluation.py \
    --dataset "${DATASET:-MultiTQ}" \
    --baselines ${BASELINES:-vanilla_llm temporal_pathrag} \
    --max-questions "${MAX_QUESTIONS:-100}" \
    --model "$MODEL_NAME"

EXIT_CODE=$?

log_progress "Evaluation completed with exit code: $EXIT_CODE"

# Stop Ollama service
cleanup

# Copy results to persistent location
RESULTS_DIR="/vol/bitbucket/${USER}/evaluation_results"
if [ -d "$RESULTS_DIR" ]; then
    log_progress "Results saved in: $RESULTS_DIR"
    ls -la "$RESULTS_DIR"
fi

echo ""
echo "Summary"
echo "Job ID: $SLURM_JOB_ID"
echo "Exit Code: $EXIT_CODE"
echo "Completion Time: $(date)"
echo "Model Used: $MODEL_NAME"
echo "Results: $RESULTS_DIR"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation with Ollama completed successfully"
else
    echo "Evaluation with Ollama failed"
fi

echo ""
echo "Job Complete"
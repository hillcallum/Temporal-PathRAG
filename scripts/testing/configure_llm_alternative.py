"""
Script to configure local LLM when OpenAI quota is exceeded
"""

import os
import json
from pathlib import Path


def configure_local_llm():
    """Configure local LLaMA model"""
    
    print("\nConfiguring Local LLM")
    
    # Default local LLM configuration
    config = {
        "provider": "local",
        "model": "llama2-7b",
        "endpoint": "http://localhost:8000"
    }
    
    # Allow customisation
    print("\nDefault configuration:")
    print(f"Model: {config['model']}")
    print(f"Endpoint: {config['endpoint']}")
    
    customise = input("\nCustomise settings? (y/N): ").strip().lower()
    
    if customise == 'y':
        model = input(f"Model name [{config['model']}]: ").strip()
        if model:
            config['model'] = model
            
        endpoint = input(f"Endpoint [{config['endpoint']}]: ").strip()
        if endpoint:
            config['endpoint'] = endpoint
    
    # Create configuration
    llm_config = {
        "default_provider": "local",
        "providers": {
            "local": config
        }
    }
    
    # Save configuration
    config_dir = Path.home() / ".temporal_pathrag"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "llm_config.json"
    with open(config_file, 'w') as f:
        json.dump(llm_config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_file}")
    
    # Provide setup instructions
    print("\nSetup instructions for local LLaMA:")
    print("""
1. Install LLaMA model locally:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make
   
2. Download model weights:
   # Download from HuggingFace or other source
   
3. Start the server:
   python3 -m llama_cpp.server --model path/to/llama-2-7b.bin --port 8000
   
4. Verify server is running:
   curl http://localhost:8000/v1/models
""")
    
    print(f"\nServer should be accessible at: {config['endpoint']}")
    print("\nConfiguration complete - run evaluation again to use local LLM")


if __name__ == "__main__":
    configure_local_llm()
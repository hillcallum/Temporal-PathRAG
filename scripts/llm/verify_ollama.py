#!/usr/bin/env python3
"""
Verification script to check Ollama setup for Temporal PathRAG
"""

import sys
import subprocess
import requests
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Ollama installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("Ollama not installed")
    print("Install from: https://ollama.ai")
    return False


def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("Ollama service running")
            print(f"Available models: {[m['name'] for m in models]}")
            return True, models
    except requests.exceptions.ConnectionError:
        pass
    
    print("Ollama service not running")
    print("Start with: ollama serve")
    return False, []


def check_default_model(models):
    """Check if default model is available"""
    DEFAULT_MODEL = "llama3.2:3b"
    model_names = [m['name'] for m in models]
    
    if any(DEFAULT_MODEL in name for name in model_names):
        print(f"Default model {DEFAULT_MODEL} available")
        return True
    else:
        print(f"Default model {DEFAULT_MODEL} not found")
        print(f"Download with: ollama pull {DEFAULT_MODEL}")
        return False


def test_model_generation():
    """Test basic model generation"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": "What is 2+2?",
                "stream": False,
                "options": {"num_predict": 10, "temperature": 0}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            print(f"Model generation working")
            print(f"Test response: {answer}")
            return True
    except Exception as e:
        print(f"Model generation failed: {e}")
    
    return False


def check_env_setup():
    """Check environment configuration"""
    env_path = project_root / '.env'
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            content = f.read()
            
        has_config = all(key in content for key in [
            'LOCAL_LLM_ENABLED',
            'LOCAL_LLM_MODEL',
            'LOCAL_LLM_HOST',
            'LOCAL_LLM_PORT'
        ])
        
        if has_config:
            print("Environment configuration found")
            return True
        else:
            print("Environment configuration incomplete")
    else:
        print("No .env file found")
    
    return False


def test_pathrag_integration():
    """Test PathRAG can connect to Ollama"""
    try:
        # Set environment variables for testing
        os.environ['LOCAL_LLM_ENABLED'] = 'true'
        os.environ['LOCAL_LLM_MODEL'] = 'llama3.2:3b'
        os.environ['LOCAL_LLM_HOST'] = 'localhost'
        os.environ['LOCAL_LLM_PORT'] = '11434'
        
        from src.llm.local_llm import LocalLLMClient
        from src.llm.config import LLMConfig
        
        config = LLMConfig()
        config.local_llm_enabled = True
        
        client = LocalLLMClient(config)
        if client.test_connection():
            print("PathRAG integration working")
            return True
        else:
            print("PathRAG cannot connect to Ollama")
            return False
            
    except Exception as e:
        print(f"PathRAG integration error: {e}")
        return False


def main():
    print("Ollama Setup Verification for Temporal PathRAG")
    
    # Track overall status
    all_checks_passed = True
    
    # 1. Check Ollama installation
    if not check_ollama_installed():
        all_checks_passed = False
        print()
        return
    
    print()
    
    # 2. Check Ollama service
    service_running, models = check_ollama_service()
    if not service_running:
        all_checks_passed = False
        print()
        return
    
    print()
    
    # 3. Check default model
    if not check_default_model(models):
        all_checks_passed = False
    
    print()
    
    # 4. Test model generation
    if models and not test_model_generation():
        all_checks_passed = False
    
    print()
    
    # 5. Check environment setup
    if not check_env_setup():
        all_checks_passed = False
    
    print()
    
    # 6. Test PathRAG integration
    if service_running and models and not test_pathrag_integration():
        all_checks_passed = False
    
    print()
    
    if all_checks_passed:
        print("All checks passed - Ollama is ready")
    else:
        print("Some checks failed")


if __name__ == "__main__":
    main()
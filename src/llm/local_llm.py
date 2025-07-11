"""
Local LLM Client for PathRAG
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from .config import llm_config

logger = logging.getLogger(__name__)

class LocalLLMClient:
    """Local LLM client for PathRAG (supporting LLaMA, Ollama, etc.)"""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialise local LLM client"""
        self.config = config or llm_config
        
        if not self.config.local_llm_enabled:
            raise ValueError("Local LLM is not enabled in configuration")
        
        self.base_url = f"http://{self.config.local_llm_host}:{self.config.local_llm_port}"
        
        logger.info(f"Local LLM client initialised: {self.config.local_llm_model} at {self.base_url}")
    
    def generate_response(self, 
                         prompt: str,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> str:
        """Generate response using local LLM"""
        
        max_tokens = max_tokens or self.config.openai_max_tokens
        temperature = temperature or self.config.openai_temperature
        
        # Format for different local LLM servers
        # Check port first to determine server type (Ollama uses 11434 by default)
        if self.config.local_llm_port == 11434:
            return self.call_ollama_api(prompt, max_tokens, temperature)
        elif "ollama" in self.config.local_llm_model.lower():
            return self.call_ollama_api(prompt, max_tokens, temperature)
        elif "llama" in self.config.local_llm_model.lower():
            return self.call_llama_api(prompt, max_tokens, temperature)
        else:
            return self.call_generic_api(prompt, max_tokens, temperature)
    
    def call_llama_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call LLaMA API"""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": ["Human:", "Assistant:", "\n\n"]
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('choices', [{}])[0].get('text', '').strip()
            
        except Exception as e:
            logger.error(f"LLaMA API error: {e}")
            raise
    
    def call_ollama_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Ollama API"""
        try:
            # Check if model is available
            self.ensure_ollama_model_available()
            
            payload = {
                "model": self.config.local_llm_model,
                "prompt": prompt,
                "stream": False,  # Disable streaming for simpler response handling
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_k": 40,
                    "top_p": 0.9,
                    "seed": 42  # For reproducibility
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60  # Increased timeout for larger models
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.Timeout:
            logger.error("Ollama API timeout - model may be loading or response is slow")
            raise
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def call_generic_api(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call generic local LLM API"""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('text', result.get('response', '')).strip()
            
        except Exception as e:
            logger.error(f"Generic LLM API error: {e}")
            raise
    
    def answer_question_with_paths(self, 
                                  question: str, 
                                  paths: List[Any]) -> str:
        """Answer question using PathRAG paths with local LLM"""
        
        # Create context from paths
        context = self.format_paths_for_context(paths)
        
        # Use a simpler prompt format for local LLMs
        prompt = f"""Question: {question}

        Context from knowledge graph:
        {context}

        Answer based only on the provided context:"""

        return self.generate_response(prompt)
    
    def format_paths_for_context(self, paths: List[Any]) -> str:
        """Format paths as context for local LLM"""
        if not paths:
            return "No relevant information found."
        
        formatted_paths = []
        for i, path in enumerate(paths):
            try:
                if hasattr(path, 'nodes') and hasattr(path, 'edges'):
                    node_names = [node.name for node in path.nodes]
                    path_str = " -> ".join(node_names)
                    formatted_paths.append(f"{i+1}. {path_str}")
                else:
                    formatted_paths.append(f"{i+1}. {str(path)}")
            except Exception as e:
                logger.warning(f"Error formatting path {i}: {e}")
                continue
        
        return "\n".join(formatted_paths)
    
    def ensure_ollama_model_available(self):
        """Ensure the Ollama model is available locally"""
        try:
            # Check if model exists
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.config.local_llm_model not in model_names:
                    logger.warning(f"Model {self.config.local_llm_model} not found locally - "
                                 f"Available models: {model_names}")
                    logger.info(f"To download: ollama pull {self.config.local_llm_model}")
        except Exception as e:
            logger.debug(f"Could not check Ollama models: {e}")
    
    def list_available_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
        except Exception:
            pass
        return []
    
    def test_connection(self) -> bool:
        """Test local LLM connection"""
        try:
            # For Ollama, check if service is running
            if self.config.local_llm_port == 11434 or "ollama" in self.config.local_llm_model.lower():
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    available_models = self.list_available_models()
                    logger.info(f"Ollama connection successful. Available models: {available_models}")
                    return True
            else:
                # For other services, try a simple generation
                response = self.generate_response("Hello", max_tokens=10)
                logger.info("Local LLM connection successful")
                return True
        except Exception as e:
            logger.error(f"Local LLM connection failed: {e}")
            return False
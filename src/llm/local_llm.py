"""
Local LLM Client for PathRAG (e.g., LLaMA2-Chat-7B)
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
            payload = {
                "model": self.config.local_llm_model,
                "prompt": prompt,
                "stream": False,  # Disable streaming for simpler response handling
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
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
    
    def test_connection(self) -> bool:
        """Test local LLM connection"""
        try:
            response = self.generate_response("Hello", max_tokens=10)
            logger.info("Local LLM connection successful")
            return True
        except Exception as e:
            logger.error(f"Local LLM connection failed: {e}")
            return False
    
    @staticmethod
    def setup_ollama_instructions() -> str:
        """Return instructions for setting up Ollama"""
        return """
To set up Ollama for local LLM inference:

1. Install Ollama:
   curl -fsSL https://ollama.ai/install.sh | sh

2. Download a model (e.g., LLaMA2):
   ollama pull llama2:7b-chat

3. Start the server:
   ollama serve

4. Update your .env file:
   LOCAL_LLM_ENABLED=true
   LOCAL_LLM_MODEL=llama2:7b-chat
   LOCAL_LLM_HOST=localhost
   LOCAL_LLM_PORT=11434

5. Test the connection:
   python -c "from src.llm import LocalLLMClient; client = LocalLLMClient(); print(client.test_connection())"
"""
    
    @staticmethod
    def setup_textgen_instructions() -> str:
        """Return instructions for setting up text-generation-webui"""
        return """
To set up text-generation-webui for local LLM inference:

1. Clone the repository:
   git clone https://github.com/oobabooga/text-generation-webui.git
   cd text-generation-webui

2. Install dependencies:
   pip install -r requirements.txt

3. Download a model (e.g., LLaMA2-7B-Chat):
   python download-model.py microsoft/DialoGPT-medium

4. Start with API mode:
   python server.py --api --listen

5. Update your .env file:
   LOCAL_LLM_ENABLED=true
   LOCAL_LLM_MODEL=llama2-7b-chat
   LOCAL_LLM_HOST=localhost
   LOCAL_LLM_PORT=5000
"""
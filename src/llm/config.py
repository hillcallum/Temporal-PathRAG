"""
LLM Configuration for PathRAG
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for LLM integration in PathRAG"""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    openai_model: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    openai_model_backup: str = os.getenv('OPENAI_MODEL_BACKUP', 'gpt-3.5-turbo-0125')
    openai_model_large: str = os.getenv('OPENAI_MODEL_LARGE', 'gpt-4o')
    openai_max_tokens: int = int(os.getenv('OPENAI_MAX_TOKENS', '2048'))
    openai_temperature: float = float(os.getenv('OPENAI_TEMPERATURE', '0.0'))
    
    # Local LLM Configuration
    local_llm_enabled: bool = os.getenv('LOCAL_LLM_ENABLED', 'false').lower() == 'true'
    local_llm_model: str = os.getenv('LOCAL_LLM_MODEL', 'llama3.2:3b')  # Default Ollama model
    local_llm_host: str = os.getenv('LOCAL_LLM_HOST', 'localhost')
    local_llm_port: int = int(os.getenv('LOCAL_LLM_PORT', '11434'))  # Default Ollama port
    
    # PathRAG Configuration
    pathrag_max_hops: int = int(os.getenv('PATHRAG_MAX_HOPS', '3'))
    pathrag_top_k_paths: int = int(os.getenv('PATHRAG_TOP_K_PATHS', '10'))
    pathrag_flow_threshold: float = float(os.getenv('PATHRAG_FLOW_THRESHOLD', '0.5'))
    
    # PathRAG Query Presets
    pathrag_simple_hops: int = int(os.getenv('PATHRAG_SIMPLE_HOPS', '2'))
    pathrag_complex_hops: int = int(os.getenv('PATHRAG_COMPLEX_HOPS', '5'))
    pathrag_deep_analysis_hops: int = int(os.getenv('PATHRAG_DEEP_ANALYSIS_HOPS', '7'))
    
    # Development Settings
    debug: bool = os.getenv('DEBUG', 'true').lower() == 'true'
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    def __post_init__(self):
        """Validate configuration after initialisation"""
        if not self.openai_api_key and not self.local_llm_enabled:
            raise ValueError("Either OpenAI API key must be provided or local LLM must be enabled")
        
        if self.openai_api_key and not self.openai_api_key.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format")
    
    @property
    def preferred_model(self) -> str:
        """Get the preferred model based on configuration"""
        if self.local_llm_enabled:
            return self.local_llm_model
        return self.openai_model
    
    @property
    def is_openai_available(self) -> bool:
        """Check if OpenAI is available"""
        return bool(self.openai_api_key)
    
    @property
    def is_local_llm_available(self) -> bool:
        """Check if local LLM is available"""
        return self.local_llm_enabled
    
    def get_model_for_task(self, task: str) -> str:
        """Get appropriate model for specific tasks"""
        task_model_mapping = {
            'question_answering': self.openai_model,
            'path_ranking': self.openai_model_backup,
            'complex_reasoning': self.openai_model_large,
            'simple_queries': self.openai_model_backup
        }
        
        return task_model_mapping.get(task, self.openai_model)
    
    def get_hops_for_query(self, query_type: str = 'default', custom_hops: int = None) -> int:
        """Get appropriate number of hops based on query complexity"""
        if custom_hops is not None:
            return custom_hops
            
        hop_mapping = {
            'simple': self.pathrag_simple_hops,
            'complex': self.pathrag_complex_hops,
            'deep': self.pathrag_deep_analysis_hops,
            'default': self.pathrag_max_hops
        }
        
        return hop_mapping.get(query_type, self.pathrag_max_hops)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (excluding sensitive data)"""
        return {
            'openai_model': self.openai_model,
            'openai_max_tokens': self.openai_max_tokens,
            'openai_temperature': self.openai_temperature,
            'local_llm_enabled': self.local_llm_enabled,
            'local_llm_model': self.local_llm_model,
            'pathrag_max_hops': self.pathrag_max_hops,
            'pathrag_top_k_paths': self.pathrag_top_k_paths,
            'pathrag_flow_threshold': self.pathrag_flow_threshold,
            'preferred_model': self.preferred_model
        }

# Global configuration instance
llm_config = LLMConfig()
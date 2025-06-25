"""
LLM Manager for PathRAG - handles fallbacks and client selection
"""

import logging
from typing import List, Dict, Any, Optional, Union
from .config import llm_config
from .openai_client import OpenAIClient
from .local_llm import LocalLLMClient

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manages multiple LLM clients with automatic fallback
    Handles OpenAI quota issues and local LLM alternatives
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialise LLM manager with fallback strategy"""
        self.config = config or llm_config
        self.clients = {}
        self.active_client = None
        
        # Initialise available clients
        self.initialise_clients()
        
        # Set primary client
        self.set_primary_client()
        
        logger.info(f"LLM Manager initialised with active client: {type(self.active_client).__name__}")
    
    def initialise_clients(self):
        """Initialise all available LLM clients"""
        
        # Try to initialise OpenAI client
        if self.config.is_openai_available:
            try:
                self.clients['openai'] = OpenAIClient(self.config)
                logger.info("OpenAI client available")
            except Exception as e:
                logger.warning(f"OpenAI client initialisation failed: {e}")
        
        # Try to initialise local LLM client
        if self.config.is_local_llm_available:
            try:
                self.clients['local'] = LocalLLMClient(self.config)
                logger.info("Local LLM client available")
            except Exception as e:
                logger.warning(f"Local LLM client initialisation failed: {e}")
    
    def set_primary_client(self):
        """Set the primary client based on availability and preference"""
        
        # Prefer OpenAI if available and working
        if 'openai' in self.clients:
            try:
                if self.clients['openai'].test_connection():
                    self.active_client = self.clients['openai']
                    logger.info("Using OpenAI as primary client")
                    return
                else:
                    logger.info("OpenAI connection failed, trying fallbacks")
            except Exception as e:
                logger.info(f"OpenAI connection test failed: {e}")
        
        # Fallback to local LLM
        if 'local' in self.clients:
            try:
                if self.clients['local'].test_connection():
                    self.active_client = self.clients['local']
                    logger.info("Using Local LLM as primary client")
                    return
                else:
                    logger.warning("Local LLM connection failed")
            except Exception as e:
                logger.warning(f"Local LLM connection test failed: {e}")
        
        # No working clients
        logger.error("No working LLM clients available")
        self.active_client = None
    
    def generate_response(self, 
                         prompt: str,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None,
                         fallback: bool = True) -> str:
        """
        Generate response with automatic fallback
        """
        
        if not self.active_client:
            raise RuntimeError("No LLM clients available")
        
        # Try primary client
        try:
            return self.active_client.generate_response(prompt, max_tokens, temperature)
        except Exception as e:
            logger.warning(f"Primary client failed: {e}")
            
            if not fallback:
                raise
            
            # Try fallback clients
            return self.try_fallback_clients(prompt, max_tokens, temperature)
    
    def try_fallback_clients(self, 
                             prompt: str,
                             max_tokens: Optional[int],
                             temperature: Optional[float]) -> str:
        """Try fallback clients in order"""
        
        # Get list of clients excluding the current active one
        fallback_clients = [
            (name, client) for name, client in self.clients.items()
            if client != self.active_client
        ]
        
        for client_name, client in fallback_clients:
            try:
                logger.info(f"Trying fallback client: {client_name}")
                response = client.generate_response(prompt, max_tokens, temperature)
                
                # Update active client if this one works
                self.active_client = client
                logger.info(f"Switched to fallback client: {client_name}")
                
                return response
                
            except Exception as e:
                logger.warning(f"Fallback client {client_name} failed: {e}")
                continue
        
        # All clients failed
        raise RuntimeError("All LLM clients failed to generate response")
    
    def answer_question_with_paths(self, 
                                  question: str, 
                                  paths: List[Any],
                                  fallback: bool = True) -> str:
        """Answer question using PathRAG paths with fallback"""
        
        if not self.active_client:
            raise RuntimeError("No LLM clients available")
        
        # Try primary client
        try:
            return self.active_client.answer_question_with_paths(question, paths)
        except Exception as e:
            logger.warning(f"Primary client failed for PathRAG QA: {e}")
            
            if not fallback:
                raise
            
            # Try fallback clients
            return self.try_fallback_pathrag(question, paths)
    
    def try_fallback_pathrag(self, question: str, paths: List[Any]) -> str:
        """Try fallback clients for PathRAG question answering"""
        
        fallback_clients = [
            (name, client) for name, client in self.clients.items()
            if client != self.active_client
        ]
        
        for client_name, client in fallback_clients:
            try:
                logger.info(f"Trying fallback PathRAG with: {client_name}")
                response = client.answer_question_with_paths(question, paths)
                
                # Update active client
                self.active_client = client
                logger.info(f"Switched to fallback client: {client_name}")
                
                return response
                
            except Exception as e:
                logger.warning(f"Fallback PathRAG client {client_name} failed: {e}")
                continue
        
        # All clients failed - return basic fallback
        return self.basic_fallback_answer(question, paths)
    
    def basic_fallback_answer(self, question: str, paths: List[Any]) -> str:
        """Basic fallback when all LLM clients fail"""
        if not paths:
            return "I couldn't find any relevant information to answer your question"
        
        # Create a simple response based on path information
        path_info = []
        for i, path in enumerate(paths[:3]):  # Show top 3 paths
            try:
                if hasattr(path, 'nodes'):
                    node_names = [node.name for node in path.nodes]
                    path_info.append(f"Path {i+1}: {' -> '.join(node_names)}")
                else:
                    path_info.append(f"Path {i+1}: {str(path)}")
            except:
                continue
        
        if path_info:
            return f"Based on the knowledge graph, I found these relevant paths:\n" + "\n".join(path_info)
        else:
            return "I found some relevant information, but couldn't process it properly."
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all LLM clients"""
        status = {
            'active_client': type(self.active_client).__name__ if self.active_client else None,
            'available_clients': list(self.clients.keys()),
            'client_status': {}
        }
        
        for name, client in self.clients.items():
            try:
                is_working = client.test_connection()
                status['client_status'][name] = 'working' if is_working else 'failed'
            except Exception as e:
                status['client_status'][name] = f'error: {str(e)[:50]}'
        
        return status
    
    def switch_client(self, client_name: str) -> bool:
        """Manually switch to a specific client"""
        if client_name not in self.clients:
            logger.error(f"Client {client_name} not available")
            return False
        
        try:
            if self.clients[client_name].test_connection():
                self.active_client = self.clients[client_name]
                logger.info(f"Switched to client: {client_name}")
                return True
            else:
                logger.error(f"Client {client_name} connection test failed")
                return False
        except Exception as e:
            logger.error(f"Failed to switch to client {client_name}: {e}")
            return False

# Global LLM manager instance
llm_manager = LLMManager()
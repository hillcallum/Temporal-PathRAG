"""
OpenAI Client for PathRAG
"""

import openai
import logging
from typing import List, Dict, Any, Optional
from .config import llm_config

logger = logging.getLogger(__name__)

class OpenAIClient:
    """OpenAI client for PathRAG question answering"""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialise OpenAI client"""
        self.config = config or llm_config
        
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not found in configuration")
        
        # Initialise OpenAI client
        openai.api_key = self.config.openai_api_key
        self.client = openai.OpenAI(api_key=self.config.openai_api_key)
        
        logger.info(f"OpenAI client initialised with model: {self.config.openai_model}")
    
    def generate_response(self, 
                         prompt: str, 
                         model: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None) -> str:
        """Generate response using OpenAI API"""
        
        model = model or self.config.openai_model
        max_tokens = max_tokens or self.config.openai_max_tokens
        temperature = temperature or self.config.openai_temperature
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Try backup model
            if model != self.config.openai_model_backup:
                logger.info(f"Retrying with backup model: {self.config.openai_model_backup}")
                return self.generate_response(prompt, self.config.openai_model_backup, max_tokens, temperature)
            raise
    
    def answer_question_with_paths(self, 
                                  question: str, 
                                  paths: List[Any]) -> str:
        """Answer question using PathRAG paths"""
        
        # Create context from paths
        context = self.format_paths_for_context(paths)
        
        prompt = f"""
You are an expert question answering system using PathRAG (Path-based Retrieval Augmented Generation).

You have been provided with relevant paths from a knowledge graph to answer the following question.

Question: {question}

Knowledge Graph Paths:
{context}

Instructions:
1. Use ONLY the information provided in the paths above
2. If the paths don't contain enough information to answer the question, say so
3. Provide a clear, concise answer
4. Cite which path(s) you used if relevant

Answer:"""

        return self.generate_response(prompt)
    
    def rank_paths(self, 
                   question: str, 
                   paths: List[Any]) -> List[Dict[str, Any]]:
        """Rank paths by relevance to question"""
        
        path_descriptions = []
        for i, path in enumerate(paths):
            path_desc = self.format_single_path(path)
            path_descriptions.append(f"Path {i+1}: {path_desc}")
        
        paths_text = "\n".join(path_descriptions)
        
        prompt = f"""
Question: {question}

Paths to rank:
{paths_text}

Rank these paths by relevance to the question from 1 (most relevant) to {len(paths)} (least relevant).
Provide your ranking as a simple list: 1, 3, 2, ... (path numbers in order of relevance)

Ranking:"""

        try:
            response = self.generate_response(prompt, model=self.config.openai_model_backup)
            
            # Parse ranking
            ranking = []
            for num in response.strip().split(','):
                try:
                    ranking.append(int(num.strip()) - 1)  # Convert to 0-based indexing
                except ValueError:
                    continue
            
            # Return ranked paths
            ranked_paths = []
            for idx in ranking:
                if 0 <= idx < len(paths):
                    ranked_paths.append({
                        'path': paths[idx],
                        'rank': len(ranked_paths) + 1,
                        'relevance_score': 1.0 - (len(ranked_paths) / len(paths))
                    })
            
            return ranked_paths
            
        except Exception as e:
            logger.error(f"Error ranking paths: {e}")
            # Return original order with default scores
            return [
                {
                    'path': path,
                    'rank': i + 1,
                    'relevance_score': 1.0 - (i / len(paths))
                }
                for i, path in enumerate(paths)
            ]
    
    def format_paths_for_context(self, paths: List[Any]) -> str:
        """Format paths as context for LLM"""
        if not paths:
            return "No relevant paths found."
        
        formatted_paths = []
        for i, path in enumerate(paths):
            path_str = self.format_single_path(path)
            formatted_paths.append(f"Path {i+1}: {path_str}")
        
        return "\n".join(formatted_paths)
    
    def format_single_path(self, path: Any) -> str:
        """Format a single path for display"""
        try:
            if hasattr(path, 'nodes') and hasattr(path, 'edges'):
                # PathRAG Path object
                node_names = [node.name for node in path.nodes]
                if path.edges:
                    relations = [edge.relation_type for edge in path.edges]
                    path_parts = []
                    for i, node_name in enumerate(node_names):
                        path_parts.append(node_name)
                        if i < len(relations):
                            path_parts.append(f"--{relations[i]}-->")
                    return " ".join(path_parts)
                else:
                    return " -> ".join(node_names)
            else:
                # Fallback string representation
                return str(path)
        except Exception as e:
            logger.warning(f"Error formatting path: {e}")
            return str(path)
    
    def test_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            response = self.generate_response("Hello, this is a test.", max_tokens=10)
            logger.info("OpenAI API connection successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection failed: {e}")
            return False
"""
Enhanced Temporal PathRAG that integrates trained temporal embeddings
with the existing PathRAG framework
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging
import networkx as nx
from datetime import datetime

from ..utils.temporal_path_filtering import TemporalPathFilter
from .temporal_embedding_retriever import TemporalEmbeddingRetriever
from ..scoring.updated_temporal_scoring import UpdatedTemporalScorer
from ..utils.entity_resolution import EntityResolver
from ..utils.updated_query_decomposer import UpdatedQueryDecomposer
from ..models import PathRAG

logger = logging.getLogger(__name__)


class EnhancedTemporalPathRAG(PathRAG):
    """Enhanced PathRAG with trained temporal embeddings and improved components"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_embeddings: bool = True,
        use_entity_resolution: bool = True,
        use_temporal_filter: bool = True,
        embedding_weight: float = 0.5,
        **kwargs
    ):
        """
        Initialise Enhanced Temporal PathRAG
        """
        super().__init__(**kwargs)
        
        self.use_embeddings = use_embeddings
        self.use_entity_resolution = use_entity_resolution
        self.use_temporal_filter = use_temporal_filter
        self.embedding_weight = embedding_weight
        
        # Initialise components
        if use_embeddings and model_path:
            self.embedding_retriever = TemporalEmbeddingRetriever(model_path)
            logger.info("Initialised temporal embedding retriever")
        else:
            self.embedding_retriever = None
            
        if use_entity_resolution:
            self.entity_resolver = EntityResolver()
            logger.info("Initialised entity resolver")
        else:
            self.entity_resolver = None
            
        if use_temporal_filter:
            self.temporal_filter = TemporalPathFilter()
            logger.info("Initialised temporal path filter")
        else:
            self.temporal_filter = None
            
        # Use updated components
        self.query_decomposer = UpdatedQueryDecomposer(
            entity_resolver=self.entity_resolver
        )
        self.temporal_scorer = UpdatedTemporalScorer()
        
    def retrieve_paths(
        self,
        query: str,
        graph: nx.Graph,
        top_k: int = 10,
        max_path_length: int = 3,
        temporal_context: Optional[Dict] = None
    ) -> List[Tuple[List[Tuple[str, str, str]], float]]:
        """
        Retrieve relevant paths for a query
        """
        # Decompose query
        sub_queries = self.query_decomposer.decompose(query, graph)
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries")
        
        all_paths = []
        
        for sub_query in sub_queries:
            # Get candidate paths using base retriever
            paths = self.get_candidate_paths(
                sub_query,
                graph,
                max_path_length
            )
            
            # Apply temporal filtering if enabled
            if self.temporal_filter and temporal_context:
                paths = self.temporal_filter.filter_paths(
                    paths,
                    graph,
                    temporal_context
                )
            
            all_paths.extend(paths)
        
        # Score paths
        if self.embedding_retriever and self.use_embeddings:
            # Use hybrid scoring
            path_scores = self.hybrid_score_paths(
                query,
                all_paths,
                graph,
                temporal_context
            )
        else:
            # Use traditional scoring
            path_scores = self.traditional_score_paths(
                query,
                all_paths,
                graph,
                temporal_context
            )
        
        # Sort and return top k
        path_scores.sort(key=lambda x: x[1], reverse=True)
        return path_scores[:top_k]
    
    def hybrid_score_paths(
        self,
        query: str,
        paths: List[List[Tuple[str, str, str]]],
        graph: nx.Graph,
        temporal_context: Optional[Dict] = None
    ) -> List[Tuple[List[Tuple[str, str, str]], float]]:
        """
        Score paths using both embeddings and traditional scoring
        """
        if not paths:
            return []
        
        # Get embedding scores
        embedding_scores = self.embedding_retriever.score_paths(
            query,
            paths,
            graph,
            temporal_context
        )
        
        # Get traditional scores
        traditional_scores = []
        for path in paths:
            score = self.temporal_scorer.score_path(
                path,
                query,
                graph
            )
            traditional_scores.append(score)
        
        # Normalise scores
        embedding_scores = self.normalise_scores(embedding_scores)
        traditional_scores = self.normalise_scores(traditional_scores)
        
        # Combine scores
        path_scores = []
        for i, path in enumerate(paths):
            combined_score = (
                self.embedding_weight * embedding_scores[i] +
                (1 - self.embedding_weight) * traditional_scores[i]
            )
            path_scores.append((path, combined_score))
        
        return path_scores
    
    def traditional_score_paths(
        self,
        query: str,
        paths: List[List[Tuple[str, str, str]]],
        graph: nx.Graph,
        temporal_context: Optional[Dict] = None
    ) -> List[Tuple[List[Tuple[str, str, str]], float]]:
        """
        Score paths using traditional scoring
        """
        path_scores = []
        
        for path in paths:
            score = self.temporal_scorer.score_path(
                path,
                query,
                graph
            )
            path_scores.append((path, score))
        
        return path_scores
    
    def get_candidate_paths(
        self,
        query: str,
        graph: nx.Graph,
        max_length: int
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Get candidate paths for a query
        """
        # Extract entities from query
        entities = self.extract_entities(query, graph)
        
        if not entities:
            logger.warning(f"No entities found in query: {query}")
            return []
        
        paths = []
        
        # Find paths between entities
        for i, start_entity in enumerate(entities):
            for end_entity in entities[i+1:]:
                # Find paths in graph
                entity_paths = self.find_paths_between(
                    start_entity,
                    end_entity,
                    graph,
                    max_length
                )
                paths.extend(entity_paths)
        
        return paths
    
    def extract_entities(
        self,
        query: str,
        graph: nx.Graph
    ) -> List[str]:
        """
        Extract entities from query
        """
        if self.entity_resolver:
            # Use entity resolver
            entities = self.entity_resolver.extract_entities(query)
            resolved = []
            
            for entity in entities:
                resolved_entity = self.entity_resolver.resolve_entity(
                    entity,
                    list(graph.nodes())
                )
                if resolved_entity:
                    resolved.append(resolved_entity)
            
            return resolved
        else:
            # Simple entity extraction
            # This will be replaced with more sophisticated NER
            entities = []
            for node in graph.nodes():
                if node.lower() in query.lower():
                    entities.append(node)
            return entities
    
    def find_paths_between(
        self,
        start: str,
        end: str,
        graph: nx.Graph,
        max_length: int
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find paths between two entities
        """
        paths = []
        
        try:
            # Use networkx to find simple paths
            simple_paths = nx.all_simple_paths(
                graph,
                start,
                end,
                cutoff=max_length
            )
            
            # Convert to triple format
            for node_path in simple_paths:
                triple_path = []
                for i in range(len(node_path) - 1):
                    source = node_path[i]
                    target = node_path[i + 1]
                    
                    # Get edge relation
                    if graph.has_edge(source, target):
                        edge_data = graph[source][target]
                        relation = edge_data.get('relation', 'connected_to')
                    else:
                        relation = 'connected_to'
                    
                    triple_path.append((source, relation, target))
                
                if triple_path:
                    paths.append(triple_path)
                    
        except nx.NetworkXNoPath:
            # No path exists
            pass
        
        return paths
    
    def normalise_scores(self, scores: List[float]) -> List[float]:
        """
        Normalise scores to [0, 1] range
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def answer_query(
        self,
        query: str,
        graph: nx.Graph,
        llm_client: Any,
        temporal_context: Optional[Dict] = None,
        top_k: int = 10
    ) -> str:
        """
        Answer a query using retrieved paths
        """
        # Retrieve relevant paths
        path_scores = self.retrieve_paths(
            query,
            graph,
            top_k,
            temporal_context=temporal_context
        )
        
        if not path_scores:
            return "I couldn't find relevant information to answer your query."
        
        # Format paths as context
        context = self.format_paths_as_context(path_scores, graph)
        
        # Generate answer using LLM
        prompt = f"""Based on the following information from the knowledge graph, please answer the query.

        Query: {query}

        Relevant information:
        {context}

        Please provide a concise and accurate answer based only on the given information."""

        try:
            response = llm_client.complete(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def format_paths_as_context(
        self,
        path_scores: List[Tuple[List[Tuple[str, str, str]], float]],
        graph: nx.Graph
    ) -> str:
        """
        Format paths as context for LLM
        """
        context_parts = []
        
        for i, (path, score) in enumerate(path_scores[:5]):  # Use top 5 paths
            context_parts.append(f"Path {i+1} (relevance: {score:.2f}):")
            
            for source, relation, target in path:
                # Basic triple
                context_parts.append(f" - {source} {relation} {target}")
                
                # Add temporal context if available
                if graph.has_edge(source, target):
                    edge_data = graph[source][target]
                    if 'te' in edge_data:
                        start, end = edge_data['te']
                        if start and end:
                            context_parts.append(f"(from {start} to {end})")
                        elif start:
                            context_parts.append(f"(since {start})")
                        elif end:
                            context_parts.append(f"(until {end})")
            
            context_parts.append("")  # Empty line between paths
        
        return "\n".join(context_parts)
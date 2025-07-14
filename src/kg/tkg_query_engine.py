"""
TKG Query Engine for Top-K Temporal Path Retrieval

This is a high-level interface for querying the TKG and retrieving top-K 
temporally relevant paths with reliability scoring
"""

import re
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import networkx as nx

from .temporal_path_retriever import TemporalPathRetriever
from .temporal_reliability_scorer import TemporalReliabilityScorer
from .temporal_scoring import TemporalWeightingFunction
from .updated_temporal_scoring import UpdatedTemporalScorer, create_updated_temporal_scorer
from .models import (
    Path, TemporalQuery, TemporalReliabilityMetrics, QueryResult, 
    PathExplanation, GraphStatistics
)


class TemporalQueryProcessor:
    """Processes natural language queries to extract temporal constraints"""
    
    def __init__(self):
        # Temporal patterns for extraction
        self.temporal_patterns = {
            'years': r'(\d{4})',
            'year_ranges': r'(\d{4})\s*[-–]\s*(\d{4})',
            'decades': r'(\d{4})s',
            'relative_time': r'(before|after|during|in|since|until)\s+(\d{4})',
            'recent': r'(recent|latest|current|now)',
            'historical': r'(historical|past|earlier|former)',
            'temporal_order': r'(first|last|then|next|previously|subsequently)'
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'person': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            'location': r'(in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            'organisation': r'([A-Z][A-Z]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(Inc|Corp|Ltd|University|Company)))'
        }
    
    def process_query(self, query_text: str) -> TemporalQuery:
        """Process natural language query into structured TemporalQuery"""
        
        # Extract temporal constraints
        temporal_constraints = self.extract_temporal_constraints(query_text)
        
        # Extract entities
        source_entities, target_entities = self.extract_entities(query_text)
        
        # Determine query time
        query_time = self.determine_query_time(query_text, temporal_constraints)
        
        # Set parameters based on query complexity
        max_hops = self.determine_max_hops(query_text)
        top_k = self.determine_top_k(query_text)
        
        # Create TemporalQuery - the model will extract patterns automatically
        return TemporalQuery(
            query_text=query_text,
            source_entities=source_entities,
            target_entities=target_entities,
            temporal_constraints=temporal_constraints,
            query_time=query_time,
            max_hops=max_hops,
            top_k=top_k
        )
    
    def extract_temporal_constraints(self, query_text: str) -> Dict[str, Any]:
        """Extract temporal constraints from query text"""
        constraints = {}
        
        # Extract year ranges
        year_range_match = re.search(self.temporal_patterns['year_ranges'], query_text)
        if year_range_match:
            start_year, end_year = year_range_match.groups()
            constraints['time_range'] = (
                datetime(int(start_year), 1, 1),
                datetime(int(end_year), 12, 31)
            )
        
        # Extract single years
        elif re.search(self.temporal_patterns['years'], query_text):
            years = re.findall(self.temporal_patterns['years'], query_text)
            if years:
                year = int(years[0])
                constraints['time_range'] = (
                    datetime(year, 1, 1),
                    datetime(year, 12, 31)
                )
        
        # Extract relative temporal references
        relative_match = re.search(self.temporal_patterns['relative_time'], query_text)
        if relative_match:
            relation, year = relative_match.groups()
            reference_date = datetime(int(year), 1, 1)
            
            if relation.lower() in ['before', 'until']:
                constraints['time_range'] = (datetime(1900, 1, 1), reference_date)
            elif relation.lower() in ['after', 'since']:
                constraints['time_range'] = (reference_date, datetime.now())
            elif relation.lower() in ['during', 'in']:
                constraints['time_range'] = (
                    datetime(int(year), 1, 1),
                    datetime(int(year), 12, 31)
                )
        
        # Extract temporal direction preferences
        if re.search(self.temporal_patterns['recent'], query_text):
            constraints['temporal_preference'] = 'recent'
        elif re.search(self.temporal_patterns['historical'], query_text):
            constraints['temporal_preference'] = 'historical'
        
        return constraints
    
    def extract_entities(self, query_text: str) -> Tuple[List[str], List[str]]:
        """Extract source and target entities from query text"""
        # Simplified entity extraction - in future, will use NER
        
        # Look for quoted entities
        quoted_entities = re.findall(r'"([^"]+)"', query_text)
        
        # Look for capitalised words (potential proper nouns)
        potential_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query_text)
        
        # For now, treat all as potential source entities
        # In future, will use dependency parsing to distinguish source vs target
        all_entities = quoted_entities + potential_entities
        
        # Remove common words that aren't entities
        stop_words = {'The', 'This', 'That', 'When', 'Where', 'Who', 'What', 'How', 'Why'}
        entities = [e for e in all_entities if e not in stop_words]
        
        # Simple heuristic for now - first half as sources, second half as targets
        mid_point = len(entities) // 2
        source_entities = entities[:mid_point] if entities else []
        target_entities = entities[mid_point:] if entities else []
        
        return source_entities, target_entities
    
    def determine_query_time(self, query_text: str, temporal_constraints: Dict[str, Any]) -> str:
        """Determine appropriate query time for temporal scoring"""
        
        # If specific time range is given, use the end of the range
        if 'time_range' in temporal_constraints:
            end_time = temporal_constraints['time_range'][1]
            return end_time.isoformat()
        
        # If recent preference, use current time
        if temporal_constraints.get('temporal_preference') == 'recent':
            return datetime.now().isoformat()
        
        # If historical preference, use earlier time
        if temporal_constraints.get('temporal_preference') == 'historical':
            return datetime(2000, 1, 1).isoformat()
        
        # Default to current time
        return datetime.now().isoformat()
    
    def determine_max_hops(self, query_text: str) -> int:
        """Determine maximum hops based on query complexity"""
        # Simple heuristic based on query length and complexity indicators
        
        complexity_indicators = ['through', 'via', 'connected', 'relationship', 'path']
        complex_query = any(indicator in query_text.lower() for indicator in complexity_indicators)
        
        if complex_query or len(query_text.split()) > 15:
            return 4  # More complex queries allow deeper exploration
        else:
            return 3  # Standard depth
    
    def determine_top_k(self, query_text: str) -> int:
        """Determine top-K value based on query"""
        # Look for explicit numbers in query
        numbers = re.findall(r'\b(\d+)\b', query_text)
        
        # Check for quantity indicators
        if any(word in query_text.lower() for word in ['all', 'every', 'complete']):
            return 20
        elif any(word in query_text.lower() for word in ['few', 'some', 'several']):
            return 5
        elif numbers:
            # Use the first number found, within reasonable bounds
            try:
                k = int(numbers[0])
                return max(5, min(k, 25))  # Bound between 5 and 25
            except ValueError:
                pass
        
        return 10  # Default


class TKGQueryEngine:
    """
    High-level interface for querying TKG and retrieving top-K temporal paths
    """
    
    def __init__(self, 
                 graph: nx.DiGraph,
                 graph_statistics: Dict[str, Any] = None,
                 alpha: float = 0.01,
                 base_theta: float = 0.1,
                 reliability_threshold: float = 0.6,
                 diversity_threshold: float = 0.7,
                 device: Optional[str] = None,
                 use_updated_scoring: bool = True,
                 embedding_integration: Optional[Any] = None,
                 temporal_adapter_path: Optional[str] = None):
        """
        Initialise TKG Query Engine
        
        Args:
            graph: The temporal knowledge graph
            graph_statistics: Pre-computed graph statistics
            alpha: Temporal decay rate
            base_theta: Base pruning threshold
            reliability_threshold: Reliability threshold for filtering
            diversity_threshold: Diversity threshold for path selection
            device: Device for computation (CPU/GPU)
            use_updated_scoring: Whether to use updated temporal scoring
            embedding_integration: Optional embedding integration instance
            temporal_adapter_path: Path to pre-trained temporal adapter
        """

        self.graph = graph
        self.graph_statistics = graph_statistics or {}
        self.use_updated_scoring = use_updated_scoring
        
        # Initialise query processor
        self.query_processor = TemporalQueryProcessor()
        
        # Initialse temporal weighting
        self.temporal_weighting = TemporalWeightingFunction(
            decay_rate=alpha,
            temporal_window=365,
            chronological_weight=0.3,
            proximity_weight=0.4,
            consistency_weight=0.3
        )
        
        # Initialise updated temporal scorer if requested
        self.updated_scorer = None
        if self.use_updated_scoring:
            try:
                from .updated_temporal_scoring import UpdatedTemporalScorer
                self.updated_scorer = UpdatedTemporalScorer(
                    graph=self.graph,
                    temporal_weighting=self.temporal_weighting,
                    embedding_integration=embedding_integration,
                    temporal_adapter_path=temporal_adapter_path
                )
                print("Updated temporal scoring enabled")
            except Exception as e:
                print(f"Failed to initialise updated scorer: {e}")
                self.use_updated_scoring = False
        
        # Initialise path retriever
        self.path_retriever = TemporalPathRetriever(
            graph=graph,
            temporal_weighting=self.temporal_weighting,
            device=device,
            alpha=alpha,
            base_theta=base_theta,
            diversity_threshold=diversity_threshold,
            updated_scorer=self.updated_scorer if use_updated_scoring else None
        )
        
        # Initialise reliability scorer
        self.reliability_scorer = TemporalReliabilityScorer(
            temporal_weighting=self.temporal_weighting,
            graph_statistics=graph_statistics,
            reliability_threshold=reliability_threshold,
            enable_cross_validation=True
        )
        
        print(f"TKGQueryEngine initialised with {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        if use_updated_scoring:
            print("Updated temporal scoring enabled")
    
    def query(self, 
              query_text: str,
              enable_flow_pruning: bool = True,
              enable_reliability_filtering: bool = True,
              enable_diversity: bool = True,
              verbose: bool = False) -> QueryResult:
        """
        Execute temporal query and retrieve top-K relevant paths
        """
        start_time = datetime.now()
        
        if verbose:
            print(f"Processing query: {query_text}")
        
        # Stage 1: Process natural language query
        temporal_query = self.query_processor.process_query(query_text)
        
        if verbose:
            print(f"Extracted entities: {temporal_query.source_entities + temporal_query.target_entities}")
            print(f"Temporal constraints: {temporal_query.temporal_constraints}")
        
        # Stage 2: Retrieve temporal paths
        retrieved_paths = self.path_retriever.retrieve_temporal_paths(
            query=temporal_query,
            enable_flow_pruning=enable_flow_pruning,
            enable_diversity=enable_diversity,
            verbose=verbose
        )
        
        total_paths_discovered = len(retrieved_paths)
        
        # Stage 3: Apply advanced reliability scoring
        paths_only = [path for path, _ in retrieved_paths]
        query_context = {
            'query_text': query_text,
            'temporal_constraints': temporal_query.temporal_constraints
        }
        
        # Use updated scoring if available
        if self.use_updated_scoring and self.updated_scorer is not None:
            # Apply updated temporal scoring
            reliability_scored_paths = self.updated_scorer.batch_score_paths(
                paths_only, 
                temporal_query.query_time,
                query_context
            )
        else:
            # Fall back to standard reliability scoring
            reliability_scored_paths = self.reliability_scorer.rank_paths_by_reliability(
                paths_only, temporal_query.query_time, query_context
            )
        
        # Stage 4: Filter by reliability threshold
        if enable_reliability_filtering:
            if self.use_updated_scoring and self.updated_scorer is not None:
                # Filter based on updated scores (simple threshold)
                final_paths = [
                    (path, score) for path, score in reliability_scored_paths
                    if score >= self.reliability_threshold
                ]
            else:
                final_paths = self.reliability_scorer.filter_reliable_paths(
                    paths_only, temporal_query.query_time, query_context
                )
        else:
            final_paths = reliability_scored_paths
        
        # Limit to requested top-K
        final_paths = final_paths[:temporal_query.top_k]
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = QueryResult(
            query_text=query_text,
            paths=final_paths,
            execution_time=execution_time,
            total_paths_discovered=total_paths_discovered,
            total_paths_after_pruning=len(paths_only),
            query_metadata={
                'temporal_query': temporal_query,
                'query_context': query_context,
                'processing_options': {
                    'flow_pruning': enable_flow_pruning,
                    'reliability_filtering': enable_reliability_filtering,
                    'diversity': enable_diversity
                }
            },
            reliability_threshold=self.reliability_scorer.reliability_threshold
        )
        
        if verbose:
            print(f"Query completed in {execution_time:.3f}s")
            print(f"Returned {len(final_paths)} reliable paths")
        
        return result
    
    def explain_path(self, path: Path, reliability_metrics: TemporalReliabilityMetrics) -> PathExplanation:
        """Generate explanation for why a path was selected"""
        
        # Generate path summary
        if path.nodes and len(path.nodes) >= 2:
            source_name = path.nodes[0].name
            target_name = path.nodes[-1].name
            path_length = len(path.nodes) - 1
            path_summary = f"{source_name} → {target_name} ({path_length} hops)"
        else:
            path_summary = "Simple path"
        
        # Reliability breakdown
        reliability_breakdown = {
            'Temporal Consistency': reliability_metrics.temporal_consistency,
            'Chronological Coherence': reliability_metrics.chronological_coherence,
            'Source Credibility': reliability_metrics.source_credibility,
            'Cross Validation': reliability_metrics.cross_validation_score,
            'Pattern Strength': reliability_metrics.temporal_pattern_strength,
            'Flow Reliability': reliability_metrics.flow_reliability,
            'Semantic Coherence': reliability_metrics.semantic_coherence
        }
        
        # Temporal highlights
        temporal_highlights = []
        if path.edges:
            timestamps = [edge.timestamp for edge in path.edges if edge.timestamp]
            if timestamps:
                temporal_highlights.append(f"Contains {len(timestamps)} timestamped events")
                
                # Find temporal span
                try:
                    dates = [datetime.fromisoformat(ts.replace('T', ' ')) for ts in timestamps]
                    span = (max(dates) - min(dates)).days
                    if span > 0:
                        temporal_highlights.append(f"Spans {span} days")
                except:
                    pass
        
        # Confidence level
        overall_score = reliability_metrics.overall_reliability
        if overall_score >= 0.8:
            confidence_level = "High"
        elif overall_score >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Supporting evidence
        supporting_evidence = []
        if reliability_metrics.temporal_consistency > 0.7:
            supporting_evidence.append("Strong temporal consistency")
        if reliability_metrics.source_credibility > 0.7:
            supporting_evidence.append("High source credibility")
        if reliability_metrics.chronological_coherence > 0.7:
            supporting_evidence.append("Good chronological ordering")
        
        return PathExplanation(
            path_summary=path_summary,
            reliability_breakdown=reliability_breakdown,
            temporal_highlights=temporal_highlights,
            confidence_level=confidence_level,
            supporting_evidence=supporting_evidence
        )
    
    def batch_query(self, 
                   queries: List[str],
                   **query_options) -> List[QueryResult]:
        """Execute multiple queries in batch"""
        results = []
        
        for query_text in queries:
            try:
                result = self.query(query_text, **query_options)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = QueryResult(
                    query_text=query_text,
                    paths=[],
                    execution_time=0.0,
                    total_paths_discovered=0,
                    total_paths_after_pruning=0,
                    query_metadata={'error': str(e)},
                    reliability_threshold=self.reliability_scorer.reliability_threshold
                )
                results.append(error_result)
        
        return results
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system performance and configuration statistics"""
        return {
            'graph_statistics': {
                'nodes': len(self.graph.nodes()),
                'edges': len(self.graph.edges()),
                'graph_type': type(self.graph).__name__
            },
            'retriever_stats': self.path_retriever.get_performance_stats(),
            'reliability_config': {
                'threshold': self.reliability_scorer.reliability_threshold,
                'weights': self.reliability_scorer.scoring_weights
            },
            'temporal_config': {
                'decay_rate': self.temporal_weighting.decay_rate,
                'temporal_window': self.temporal_weighting.temporal_window
            }
        }
    
    def validate_system(self, test_queries: List[str]) -> Dict[str, Any]:
        """Validate system performance with test queries"""
        validation_start = datetime.now()
        
        results = self.batch_query(test_queries, verbose=False)
        
        # Calculate validation metrics
        successful_queries = [r for r in results if r.paths and 'error' not in r.query_metadata]
        
        validation_metrics = {
            'total_queries': len(test_queries),
            'successful_queries': len(successful_queries),
            'success_rate': len(successful_queries) / len(test_queries) if test_queries else 0,
            'average_execution_time': np.mean([r.execution_time for r in successful_queries]) if successful_queries else 0,
            'average_paths_returned': np.mean([len(r.paths) for r in successful_queries]) if successful_queries else 0,
            'average_reliability_score': 0.0,
            'validation_time': (datetime.now() - validation_start).total_seconds()
        }
        
        # Calculate average reliability
        if successful_queries:
            all_reliability_scores = []
            for result in successful_queries:
                scores = [metrics.overall_reliability for _, metrics in result.paths]
                all_reliability_scores.extend(scores)
            
            if all_reliability_scores:
                validation_metrics['average_reliability_score'] = np.mean(all_reliability_scores)
        
        return validation_metrics
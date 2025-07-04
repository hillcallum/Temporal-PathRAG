"""
Temporal Reliability Scorer for TKG Path Ranking

Temporal reliability scoring that accurately identifies and ranks paths 
by using improved temporal reliability metrics compared to previous basic integration 

Key Features:
1. Reliability scoring focusing on the temporal factors
2. Temporal validation and consistency verification
3. Evaluation of source credibility and temporal coherence
4. Recognition and scoring of temporal patterns
"""

import math
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
import statistics

from .models import (
    TemporalPathRAGNode, TemporalPathRAGEdge, Path, TemporalReliabilityMetrics,
    GraphStatistics
)
from .temporal_scoring import TemporalWeightingFunction, TemporalPath, TemporalRelevanceMode


class TemporalReliabilityScorer:
    """
    Temporal reliability scoring system for path ranking
    """
    
    def __init__(self, 
                 temporal_weighting: TemporalWeightingFunction,
                 graph_statistics: Dict[str, Any] = None,
                 reliability_threshold: float = 0.6,
                 enable_cross_validation: bool = True):
        """
        Initialise temporal reliability scorer
        """
        self.temporal_weighting = temporal_weighting
        self.graph_statistics = graph_statistics or {}
        self.reliability_threshold = reliability_threshold
        self.enable_cross_validation = enable_cross_validation
        
        # Initialise credibility assessment components
        self.entity_credibility_cache = {}
        self.relation_credibility_cache = {}
        self.temporal_pattern_cache = {}
        
        # Reliability scoring weights (learned/tuned parameters)
        self.scoring_weights = {
            'temporal_consistency': 0.25,      # How well timestamps align
            'chronological_coherence': 0.20,   # Logical temporal ordering
            'source_credibility': 0.15,        # Entity/relation credibility
            'cross_validation': 0.15,          # Multi-source validation
            'pattern_strength': 0.10,          # Temporal pattern recognition
            'flow_reliability': 0.10,          # Flow-based reliability
            'semantic_coherence': 0.05         # Semantic consistency
        }
        
        print("TemporalReliabilityScorer initialised with advanced scoring metrics")
    
    def score_path_reliability(self, 
                             path: Path, 
                             query_time: str,
                             query_context: Dict[str, Any] = None) -> TemporalReliabilityMetrics:
        """
        Compute comprehensive temporal reliability score for a path
        """
        metrics = TemporalReliabilityMetrics()
        query_context = query_context or {}
        
        # Component 1: Temporal Consistency Assessment
        metrics.temporal_consistency = self.assess_temporal_consistency(path, query_time)
        
        # Component 2: Chronological Coherence Analysis
        metrics.chronological_coherence = self.assess_chronological_coherence(path)
        
        # Component 3: Source Credibility Evaluation
        metrics.source_credibility = self.assess_source_credibility(path)
        
        # Component 4: Cross-Validation Scoring 
        if self.enable_cross_validation:
            metrics.cross_validation_score = self.assess_cross_validation(path, query_context)
        else:
            metrics.cross_validation_score = 0.5  # Neutral score
        
        # Component 5: Temporal Pattern Strength
        metrics.temporal_pattern_strength = self.assess_temporal_pattern_strength(path, query_context)
        
        # Component 6: Flow-Based Reliability
        metrics.flow_reliability = self.assess_flow_reliability(path, query_time)
        
        # Component 7: Semantic Coherence
        metrics.semantic_coherence = self.assess_semantic_coherence(path)
        
        # Compute overall reliability score
        metrics.overall_reliability = self.compute_overall_reliability(metrics)
        
        return metrics
    
    def rank_paths_by_reliability(self, 
                                paths: List[Path], 
                                query_time: str,
                                query_context: Dict[str, Any] = None) -> List[Tuple[Path, TemporalReliabilityMetrics]]:
        """
        Rank paths by temporal reliability scores
        """
        scored_paths = []
        
        for path in paths:
            reliability_metrics = self.score_path_reliability(path, query_time, query_context)
            scored_paths.append((path, reliability_metrics))
        
        # Sort by overall reliability score (descending)
        scored_paths.sort(key=lambda x: x[1].overall_reliability, reverse=True)
        
        return scored_paths
    
    def filter_reliable_paths(self, 
                            paths: List[Path], 
                            query_time: str,
                            query_context: Dict[str, Any] = None) -> List[Tuple[Path, TemporalReliabilityMetrics]]:
        """
        Filter paths to only include those meeting reliability threshold
        """
        ranked_paths = self.rank_paths_by_reliability(paths, query_time, query_context)
        
        reliable_paths = [
            (path, metrics) for path, metrics in ranked_paths
            if metrics.overall_reliability >= self.reliability_threshold
        ]
        
        return reliable_paths
    
    # Component scoring methods
    
    def assess_temporal_consistency(self, path: Path, query_time: str) -> float:
        """
        Assess temporal consistency of path with respect to query time
        """
        if not path.edges:
            return 0.5  # Neutral for paths without edges
        
        temporal_info = path.get_temporal_info()
        timestamps = temporal_info.get('timestamps', [])
        
        if not timestamps:
            return 0.3  # Penalty for missing temporal information
        
        try:
            query_dt = datetime.fromisoformat(query_time.replace('T', ' '))
            edge_dates = []
            
            # Parse valid timestamps
            for timestamp in timestamps:
                try:
                    edge_dt = datetime.fromisoformat(timestamp.replace('T', ' '))
                    edge_dates.append(edge_dt)
                except (ValueError, TypeError):
                    continue
            
            if not edge_dates:
                return 0.3
            
            # Calculate temporal proximity scores
            proximity_scores = []
            for edge_dt in edge_dates:
                days_diff = abs((query_dt - edge_dt).days)
                # Use temporal weighting function for consistency
                proximity = self.temporal_weighting.temporal_decay_factor(
                    edge_dt.isoformat(), query_time, TemporalRelevanceMode.EXPONENTIAL_DECAY
                )
                proximity_scores.append(proximity)
            
            # Calculate consistency metrics
            avg_proximity = np.mean(proximity_scores)
            proximity_variance = np.var(proximity_scores)
            
            # Consistency score: high proximity, low variance is better
            consistency_score = avg_proximity * (1.0 - min(proximity_variance, 0.5))
            
            return max(0.0, min(1.0, consistency_score))
            
        except Exception:
            return 0.3  # Default low score for errors
    
    def assess_chronological_coherence(self, path: Path) -> float:
        """
        Assess chronological coherence of events in path
        """
        if not path.edges or len(path.edges) < 2:
            return 1.0  # Single or no edges are trivially coherent
        
        temporal_info = path.get_temporal_info()
        timestamps = temporal_info.get('timestamps', [])
        
        if len(timestamps) < 2:
            return 0.7  # Partial penalty for missing temporal info
        
        try:
            edge_dates = []
            for timestamp in timestamps:
                try:
                    edge_dt = datetime.fromisoformat(timestamp.replace('T', ' '))
                    edge_dates.append(edge_dt)
                except (ValueError, TypeError):
                    edge_dates.append(None)  # Mark invalid timestamps
            
            # Count violations and assess coherence
            violations = 0
            valid_pairs = 0
            temporal_gaps = []
            
            for i in range(len(edge_dates) - 1):
                if edge_dates[i] is not None and edge_dates[i + 1] is not None:
                    valid_pairs += 1
                    
                    # Check chronological ordering
                    if edge_dates[i] > edge_dates[i + 1]:
                        violations += 1
                    
                    # Calculate temporal gap
                    gap_days = abs((edge_dates[i + 1] - edge_dates[i]).days)
                    temporal_gaps.append(gap_days)
            
            if valid_pairs == 0:
                return 0.5
            
            # Calculate coherence metrics
            violation_rate = violations / valid_pairs
            
            # Assess temporal gap consistency (prefer moderate, consistent gaps)
            if temporal_gaps:
                gap_consistency = 1.0 - min(np.std(temporal_gaps) / max(np.mean(temporal_gaps), 1), 1.0)
            else:
                gap_consistency = 0.5
            
            # Combined coherence score
            coherence_score = (1.0 - violation_rate) * 0.7 + gap_consistency * 0.3
            
            return max(0.0, min(1.0, coherence_score))
            
        except Exception:
            return 0.5
    
    def assess_source_credibility(self, path: Path) -> float:
        """
        Assess credibility of sources (entities and relations) in path
        """
        if not path.nodes or not path.edges:
            return 0.5
        
        entity_scores = []
        relation_scores = []
        
        # Assess entity credibility
        for node in path.nodes:
            entity_score = self.get_entity_credibility(node.id)
            entity_scores.append(entity_score)
        
        # Assess relation credibility
        for edge in path.edges:
            relation_score = self.get_relation_credibility(edge.relation_type)
            relation_scores.append(relation_score)
        
        # Combine scores (geometric mean for conservative estimate)
        if entity_scores and relation_scores:
            entity_credibility = np.exp(np.mean(np.log(np.maximum(entity_scores, 1e-6))))
            relation_credibility = np.exp(np.mean(np.log(np.maximum(relation_scores, 1e-6))))
            
            overall_credibility = (entity_credibility * 0.6 + relation_credibility * 0.4)
        else:
            overall_credibility = 0.5
        
        return max(0.0, min(1.0, overall_credibility))
    
    def assess_cross_validation(self, path: Path, query_context: Dict[str, Any]) -> float:
        """
        Assess path reliability through cross-validation with multiple sources
        """
        # Simplified cross-validation - later on we will query multiple sources
        
        if not path.edges:
            return 0.5
        
        # Count supporting evidence patterns
        support_score = 0.0
        evidence_count = 0
        
        for edge in path.edges:
            # Check for supporting patterns in edge properties
            edge_properties = getattr(edge, 'properties', {})
            
            # Look for validation indicators
            if 'confidence' in edge_properties:
                try:
                    confidence = float(edge_properties['confidence'])
                    support_score += confidence
                    evidence_count += 1
                except (ValueError, TypeError):
                    pass
            
            # Check for source multiplicity
            if 'sources' in edge_properties:
                source_count = len(edge_properties.get('sources', []))
                if source_count > 1:
                    support_score += min(0.2 * source_count, 1.0)
                    evidence_count += 1
        
        # Default neutral score if no evidence
        if evidence_count == 0:
            return 0.5
        
        avg_support = support_score / evidence_count
        return max(0.0, min(1.0, avg_support))
    
    def assess_temporal_pattern_strength(self, path: Path, query_context: Dict[str, Any]) -> float:
        """
        Assess strength of temporal patterns in path
        """
        if not path.edges:
            return 0.5
        
        temporal_info = path.get_temporal_info()
        timestamps = temporal_info.get('timestamps', [])
        
        if len(timestamps) < 2:
            return 0.4
        
        try:
            # Parse timestamps
            edge_dates = []
            for timestamp in timestamps:
                try:
                    edge_dt = datetime.fromisoformat(timestamp.replace('T', ' '))
                    edge_dates.append(edge_dt)
                except (ValueError, TypeError):
                    continue
            
            if len(edge_dates) < 2:
                return 0.4
            
            # Assess temporal patterns
            pattern_scores = []
            
            # 1. Regular intervals pattern
            if len(edge_dates) >= 3:
                intervals = [(edge_dates[i+1] - edge_dates[i]).days for i in range(len(edge_dates)-1)]
                interval_consistency = 1.0 - min(np.std(intervals) / max(np.mean(intervals), 1), 1.0)
                pattern_scores.append(interval_consistency)
            
            # 2. Temporal clustering
            spans = [(edge_dates[-1] - edge_dates[0]).days]
            if spans[0] > 0:
                clustering_score = min(len(edge_dates) / spans[0] * 365, 1.0)  # Events per year
                pattern_scores.append(clustering_score)
            
            # 3. Query context matching
            query_text = query_context.get('query_text', '').lower()
            if any(keyword in query_text for keyword in ['recent', 'latest', 'current']):
                # Prefer recent events
                most_recent = max(edge_dates)
                recency_days = (datetime.now() - most_recent).days
                recency_score = max(0, 1.0 - recency_days / 365)
                pattern_scores.append(recency_score)
            elif any(keyword in query_text for keyword in ['historical', 'past', 'earlier']):
                # Prefer historical events
                oldest = min(edge_dates)
                age_days = (datetime.now() - oldest).days
                historical_score = min(age_days / (365 * 10), 1.0)  # Older = better, cap at 10 years
                pattern_scores.append(historical_score)
            
            # Average pattern strength
            if pattern_scores:
                return np.mean(pattern_scores)
            else:
                return 0.5
                
        except Exception:
            return 0.4
    
    def assess_flow_reliability(self, path: Path, query_time: str) -> float:
        """
        Assess flow-based reliability using temporal flow principles
        """
        if not path.edges:
            return 1.0  # No edges = no flow issues
        
        # Calculate flow capacities
        flow_capacities = []
        for edge in path.edges:
            capacity = getattr(edge, 'flow_capacity', 1.0)
            flow_capacities.append(capacity)
        
        if not flow_capacities:
            return 0.5
        
        # Flow reliability metrics
        min_capacity = min(flow_capacities)
        avg_capacity = np.mean(flow_capacities)
        capacity_variance = np.var(flow_capacities)
        
        # Reliability assessment
        # 1. Bottleneck assessment (minimum capacity impact)
        bottleneck_score = min_capacity
        
        # 2. Flow consistency (low variance is better)
        consistency_score = 1.0 - min(capacity_variance, 0.5)
        
        # 3. Overall capacity level
        capacity_score = avg_capacity
        
        # Combined flow reliability
        flow_reliability = (
            bottleneck_score * 0.4 +
            consistency_score * 0.3 +
            capacity_score * 0.3
        )
        
        return max(0.0, min(1.0, flow_reliability))
    
    def assess_semantic_coherence(self, path: Path) -> float:
        """
        Assess semantic coherence of path components
        """
        if not path.nodes or not path.edges:
            return 0.5
        
        coherence_factors = []
        
        # 1. Entity type diversity assessment
        entity_types = [node.entity_type for node in path.nodes if hasattr(node, 'entity_type')]
        if entity_types:
            type_diversity = len(set(entity_types)) / len(entity_types)
            # Moderate diversity is preferred (not too uniform, not too chaotic is good )
            diversity_score = 1.0 - abs(type_diversity - 0.6)
            coherence_factors.append(diversity_score)
        
        # 2. Relation meaningfulness
        meaningful_relations = {
            'born_in', 'died_in', 'worked_at', 'graduated_from', 'founded',
            'located_in', 'part_of', 'member_of', 'led_by', 'created_by',
            'occurred_in', 'participated_in', 'resulted_in', 'caused_by'
        }
        
        relation_types = [edge.relation_type for edge in path.edges]
        if relation_types:
            meaningful_count = sum(1 for rel in relation_types if rel in meaningful_relations)
            meaningfulness_score = meaningful_count / len(relation_types)
            coherence_factors.append(meaningfulness_score)
        
        # 3. Path length reasonableness
        path_length = len(path.nodes)
        if path_length > 0:
            # Prefer paths of moderate length (2-4 hops)
            length_score = max(0, 1.0 - abs(path_length - 3) * 0.2)
            coherence_factors.append(length_score)
        
        # Average coherence
        if coherence_factors:
            return np.mean(coherence_factors)
        else:
            return 0.5
    
    def compute_overall_reliability(self, metrics: TemporalReliabilityMetrics) -> float:
        """Compute weighted overall reliability score"""
        overall_score = (
            metrics.temporal_consistency * self.scoring_weights['temporal_consistency'] +
            metrics.chronological_coherence * self.scoring_weights['chronological_coherence'] +
            metrics.source_credibility * self.scoring_weights['source_credibility'] +
            metrics.cross_validation_score * self.scoring_weights['cross_validation'] +
            metrics.temporal_pattern_strength * self.scoring_weights['pattern_strength'] +
            metrics.flow_reliability * self.scoring_weights['flow_reliability'] +
            metrics.semantic_coherence * self.scoring_weights['semantic_coherence']
        )
        
        return max(0.0, min(1.0, overall_score))
    
    # Helper methods for credibility assessment
    
    def get_entity_credibility(self, entity_id: str) -> float:
        """Get credibility score for an entity"""
        if entity_id in self.entity_credibility_cache:
            return self.entity_credibility_cache[entity_id]
        
        # Calculate credibility based on graph statistics
        credibility = 0.5  # Default neutral credibility
        
        if self.graph_statistics:
            entity_stats = self.graph_statistics.get('entities', {})
            entity_info = entity_stats.get(entity_id, {})
            
            # Factor 1: Frequency in knowledge graph
            frequency = entity_info.get('frequency', 0)
            if frequency > 0:
                # Normalise frequency score (log scale to prevent dominance)
                max_frequency = self.graph_statistics.get('max_entity_frequency', 1)
                frequency_score = min(math.log(frequency + 1) / math.log(max_frequency + 1), 1.0)
                credibility = 0.4 + 0.6 * frequency_score
        
        self.entity_credibility_cache[entity_id] = credibility
        return credibility
    
    def get_relation_credibility(self, relation_type: str) -> float:
        """Get credibility score for a relation type"""
        if relation_type in self.relation_credibility_cache:
            return self.relation_credibility_cache[relation_type]
        
        # Calculate credibility based on relation reliability
        credibility = 0.5  # Default neutral credibility
        
        # High-credibility relation types
        high_credibility_relations = {
            'born_in', 'died_in', 'graduated_from', 'worked_at',
            'founded', 'located_in', 'capital_of', 'part_of'
        }
        
        # Medium-credibility relation types
        medium_credibility_relations = {
            'member_of', 'participated_in', 'led_by', 'created_by',
            'occurred_in', 'influenced_by', 'associated_with'
        }
        
        if relation_type in high_credibility_relations:
            credibility = 0.8
        elif relation_type in medium_credibility_relations:
            credibility = 0.6
        elif self.graph_statistics:
            # Use graph statistics if available
            relation_stats = self.graph_statistics.get('relations', {})
            relation_info = relation_stats.get(relation_type, {})
            
            frequency = relation_info.get('frequency', 0)
            if frequency > 0:
                max_frequency = self.graph_statistics.get('max_relation_frequency', 1)
                frequency_score = min(frequency / max_frequency, 1.0)
                credibility = 0.3 + 0.4 * frequency_score
        
        self.relation_credibility_cache[relation_type] = credibility
        return credibility
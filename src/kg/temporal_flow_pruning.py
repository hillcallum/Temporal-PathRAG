"""
Temporal Flow-Based Pruning module for PathRAG

This module implements the enhanced temporal resource propagation algorithm
that modifies PathRAG's flow-based pruning to incorporate temporal weighting
"""

import math
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .models import TemporalPathRAGNode, TemporalPathRAGEdge, Path
from .temporal_scoring import TemporalWeightingFunction, TemporalPath, TemporalRelevanceMode


class TemporalFlowPruning:
    """
    Temporal-aware flow-based pruning system that enhances PathRAG's 
    resource propagation with temporal weighting mechanisms
    """
    
    def __init__(self, 
                 temporal_weighting: TemporalWeightingFunction,
                 temporal_mode: TemporalRelevanceMode = TemporalRelevanceMode.EXPONENTIAL_DECAY,
                 alpha: float = 0.1,  # Temporal decay rate
                 base_theta: float = 1.0):  # Base pruning threshold
        """
        Initialise temporal flow pruning system with parameter validation
        """
        self.temporal_weighting = temporal_weighting
        self.temporal_mode = temporal_mode
        
        # Validate and set temporal decay rate alpha
        if not 0.001 <= alpha <= 1.0:
            raise ValueError(f"Alpha (temporal decay rate) must be between 0.001 and 1.0, got {alpha}")
        self.alpha = alpha
        
        # Validate and set base pruning threshold theta
        if not 0.1 <= base_theta <= 10.0:
            raise ValueError(f"Base theta (pruning threshold) must be between 0.1 and 10.0, got {base_theta}")
        self.base_theta = base_theta
        
        # Track parameter interactions for monitoring
        self.alpha_theta_interaction_strength = alpha * base_theta
        self.parameter_history = []
    
    def flow_based_pruning_with_temporal_weighting(self, 
                                                  paths: List[Path], 
                                                  top_k: int, 
                                                  query_time: str) -> List[Path]:
        """
        Apply temporal-aware flow-based pruning inspired by PathRAG paper
        
        Enhanced with temporal weighting that incorporates:
        - Temporal resource propagation with decay rate alpha
        - Adaptive pruning thresholds theta based on temporal context
        - Temporal flow capacity adjustments
        
        Mathes:
        Flow_temporal(P) = min(temporal_edge_capacity) * temporal_decay_factor(P)
        Where temporal_edge_capacity = base_capacity * temporal_weight(edge)
        """
        if not paths:
            return []
        
        # Step 1: Update temporal edge capacities
        self.update_temporal_edge_capacities(paths, query_time)
        
        # Step 2: Group paths by source-target pairs
        path_groups = defaultdict(list)
        for path in paths:
            if path.nodes:
                key = (path.nodes[0].id, path.nodes[-1].id)
                path_groups[key].append(path)
        
        pruned_paths = []
        
        # Step 3: Apply temporal-aware pruning to each group
        for (source, target), group_paths in path_groups.items():
            # Sort by enhanced temporal score
            group_paths.sort(key=lambda p: p.score, reverse=True)
            
            # Calculate adaptive temporal threshold for this group
            adaptive_threshold = self.calculate_adaptive_threshold(group_paths, query_time, top_k)
            
            # Apply temporal-aware flow capacity constraints
            selected_paths = []
            total_temporal_flow = 0.0
            
            for path in group_paths:
                # Calculate temporal flow with resource propagation
                temporal_flow = self.calculate_temporal_path_flow(path, query_time)
                
                # Apply temporal decay rate α in flow propagation
                decayed_flow = self.apply_temporal_decay_propagation(temporal_flow, path, query_time)
                
                # Check if path passes adaptive threshold
                if total_temporal_flow + decayed_flow <= adaptive_threshold:
                    selected_paths.append(path)
                    total_temporal_flow += decayed_flow
                
                # Adaptive stopping based on temporal relevance
                group_limit = self.calculate_group_limit(group_paths, top_k, query_time)
                if len(selected_paths) >= group_limit:
                    break
            
            pruned_paths.extend(selected_paths)
        
        # Step 4: Final ranking and truncation
        return sorted(pruned_paths, key=lambda p: p.score, reverse=True)[:top_k]
    
    def update_temporal_edge_capacities(self, paths: List[Path], query_time: str):
        """
        Update edge flow capacities with temporal weighting 
        
        This method implements the fundamental modification to PathRAG's resource propagation by:
        1. Applying temporal decay factors directly to edge capacities
        2. Ensuring temporal relevance affects resource flow at the edge level
        3. Creating temporal-aware bottlenecks in the flow network
        
        Maths:
        temporal_edge_capacity = base_capacity * f_temporal(edge_timestamp, query_time, alpha)
        
        Where f_temporal incorporates the decay rate alpha in the temporal weighting function
        """
        temporal_modifications = 0
        total_edges = 0
        
        for path in paths:
            for edge in path.edges:
                total_edges += 1
                
                # Preserve original capacity for restoration if needed
                if not hasattr(edge, '_original_capacity'):
                    edge.original_capacity = getattr(edge, 'flow_capacity', 1.0)
                
                if hasattr(edge, 'timestamp') and edge.timestamp:
                    # Calculate temporal weight with alpha parameter influence
                    temporal_weight = self.temporal_weighting.temporal_decay_factor(
                        edge.timestamp, query_time, self.temporal_mode
                    )
                    
                    # Apply alpha-influenced temporal adjustment
                    # Higher alpha = stronger temporal decay influence on capacity
                    alpha_adjusted_weight = temporal_weight ** (1.0 + self.alpha)
                    
                    # Update edge capacity with temporal weighting
                    edge.flow_capacity = edge.original_capacity * alpha_adjusted_weight
                    temporal_modifications += 1
                else:
                    # For edges without timestamps, apply neutral capacity with alpha penalty
                    neutral_penalty = 1.0 - (self.alpha * 0.1)  # Small penalty for missing temporal data
                    edge.flow_capacity = edge.original_capacity * max(0.1, neutral_penalty)
        
        # Track modification statistics for validation
        modification_rate = temporal_modifications / max(total_edges, 1)
        self.parameter_history.append({
            'query_time': query_time,
            'modification_rate': modification_rate,
            'alpha': self.alpha,
            'temporal_edges': temporal_modifications,
            'total_edges': total_edges
        })
    
    def calculate_adaptive_threshold(self, group_paths: List[Path], query_time: str, base_top_k: int) -> float:
        """
        Calculate adaptive pruning threshold theta with strong alpha-theta interaction
        
        This method ensures that decay rate (alpha) and pruning threshold (theta) interact effectively by:
        1. Making threshold adaptation sensitive to alpha parameter
        2. Creating feedback between temporal decay and flow capacity limits
        3. Enabling dynamic threshold adjustment based on temporal context
        
        Maths:
        theta_adaptive = theta_base × (1 + alpha_influence * temporal_context_factor)
        
        Where:
        - alpha_influence = alpha * temporal_sensitivity_multiplier
        - temporal_context_factor = temporal_richness * temporal_quality * alpha_boost
        """
        if not group_paths:
            return float(self.base_theta * base_top_k)
        
        temporal_densities = []
        temporal_proximities = []
        alpha_weighted_proximities = []
        
        for path in group_paths:
            # Calculate temporal density (proportion of edges with timestamps)
            temporal_info = path.get_temporal_info()
            temporal_density = temporal_info.get('temporal_density', 0.0)
            temporal_densities.append(temporal_density)
            
            # Calculate average temporal proximity to query
            if temporal_info['timestamps']:
                proximity_scores = []
                for timestamp in temporal_info['timestamps']:
                    proximity = self.temporal_weighting.temporal_decay_factor(
                        timestamp, query_time, self.temporal_mode
                    )
                    proximity_scores.append(proximity)
                avg_proximity = sum(proximity_scores) / len(proximity_scores)
                
                # Apply alpha-weighted proximity calculation
                alpha_weighted_proximity = avg_proximity ** (1.0 + self.alpha)
                alpha_weighted_proximities.append(alpha_weighted_proximity)
            else:
                avg_proximity = 0.5  # Neutral for paths without temporal data
                alpha_weighted_proximities.append(avg_proximity * (1.0 - self.alpha * 0.2))
            
            temporal_proximities.append(avg_proximity)
        
        # Calculate group temporal characteristics with alpha influence
        avg_temporal_density = sum(temporal_densities) / len(temporal_densities)
        avg_temporal_proximity = sum(temporal_proximities) / len(temporal_proximities)
        avg_alpha_weighted_proximity = sum(alpha_weighted_proximities) / len(alpha_weighted_proximities)
        
        # Enhanced adaptive threshold calculation with strong alpha-theta interaction
        temporal_richness_factor = avg_temporal_density * (0.5 + self.alpha * 1.0)  # Stronger alpha influence
        temporal_quality_factor = avg_alpha_weighted_proximity * (0.3 + self.alpha * 1.2)  # Stronger alpha influence
        
        # alpha-theta interaction strength: higher alpha makes threshold more sensitive to temporal context
        alpha_sensitivity_multiplier = 1.0 + (self.alpha * 5.0)  # Stronger multiplier
        temporal_context_factor = temporal_richness_factor * temporal_quality_factor * alpha_sensitivity_multiplier
        
        # Final adaptive threshold with alpha-theta coupling
        base_threshold = self.base_theta * base_top_k
        
        # Apply stronger alpha influence on threshold adaptation
        alpha_threshold_modifier = 1.0 + (self.alpha * temporal_context_factor * 2.0)
        adaptive_threshold = base_threshold * alpha_threshold_modifier
        
        # Ensure alpha-theta interaction creates meaningful threshold variation
        min_threshold = base_threshold * (0.5 + self.alpha * 0.3)  # alpha affects minimum
        max_threshold = base_threshold * (1.0 + self.alpha * 5.0)   # alpha affects maximum range
        adaptive_threshold = max(min_threshold, min(max_threshold, adaptive_threshold))
        
        return adaptive_threshold
    
    def calculate_temporal_path_flow(self, path: Path, query_time: str) -> float:
        """
        Calculate temporal-aware flow capacity of a path
        
        Maths:
        Flow_temporal(P) = min(temporal_edge_capacity_i) * temporal_decay_factor(P)
        
        Where:
        - temporal_edge_capacity_i = base_capacity * temporal_weight(edge_i)
        - temporal_decay_factor considers path length and temporal span
        """
        if not path.edges:
            return 1.0
        
        # Extract temporal information
        temporal_info = path.get_temporal_info()
        
        # Calculate temporal edge capacities (should already be updated)
        temporal_capacities = []
        for edge in path.edges:
            capacity = getattr(edge, 'flow_capacity', 1.0)
            temporal_capacities.append(capacity)
        
        # Flow limited by minimum temporal capacity (bottleneck principle)
        min_temporal_capacity = min(temporal_capacities)
        
        # Enhanced temporal decay factor considering both length and temporal span
        path_length_decay = 1.0 / (1.0 + len(path.edges) * 0.1)
        
        # Temporal span decay (how spread out the events are)
        temporal_span_decay = 1.0
        if len(temporal_info['timestamps']) > 1:
            try:
                timestamps = [datetime.fromisoformat(ts.replace('T', ' ')) for ts in temporal_info['timestamps']]
                span_days = (max(timestamps) - min(timestamps)).days
                temporal_span_decay = 1.0 / (1.0 + span_days * 0.001)  # Gentle decay for temporal span
            except (ValueError, TypeError):
                pass
        
        # Combined temporal flow
        temporal_flow = min_temporal_capacity * path_length_decay * temporal_span_decay
        
        return temporal_flow
    
    def apply_temporal_decay_propagation(self, base_flow: float, path: Path, query_time: str) -> float:
        """
        Apply temporal decay rate alpha in resource propagation 

        This method implements the core part of temporal resource propagation by:
        1. Applying exponential decay based on temporal distance
        2. Using alpha parameter to control decay strength
        3. Creating temporal-aware resource flow attenuation
        
        Maths:
        Flow_decayed = base_flow * e^(-alpha * temporal_distance_factor) * temporal_coherence_bonus
        
        Where:
        - alpha: Temporal decay rate parameter (configurable, validated)
        - temporal_distance_factor: Normalised temporal distance from query
        - temporal_coherence_bonus: Additional factor for chronologically coherent paths
        """
        # Calculate temporal distance factor
        temporal_info = path.get_temporal_info()
        if not temporal_info['timestamps']:
            # Apply mild penalty for non-temporal paths when α is high
            non_temporal_penalty = 1.0 - (self.alpha * 0.15)
            return base_flow * max(0.1, non_temporal_penalty)
        
        # Calculate comprehensive temporal metrics
        temporal_distances = []
        chronological_violations = 0
        
        try:
            query_date = datetime.fromisoformat(query_time.replace('T', ' '))
            timestamps = [datetime.fromisoformat(ts.replace('T', ' ')) for ts in temporal_info['timestamps']]
            
            # Calculate temporal distances
            for timestamp in timestamps:
                distance_days = abs((query_date - timestamp).days)
                temporal_distances.append(distance_days)
            
            # Check chronological ordering
            for i in range(len(timestamps) - 1):
                if timestamps[i] > timestamps[i + 1]:
                    chronological_violations += 1
            
            # Calculate primary temporal distance factor
            avg_distance_days = sum(temporal_distances) / len(temporal_distances)
            temporal_window = getattr(self.temporal_weighting, 'temporal_window', 365)
            temporal_distance_factor = avg_distance_days / temporal_window
            
            # Calculate temporal span factor (paths spanning long periods get penalty)
            if len(timestamps) > 1:
                span_days = (max(timestamps) - min(timestamps)).days
                span_factor = span_days / temporal_window
            else:
                span_factor = 0.0
            
        except (ValueError, TypeError):
            temporal_distance_factor = 0.5  # Neutral factor for invalid timestamps
            span_factor = 0.5
            chronological_violations = len(temporal_info['timestamps']) // 2
        
        # Apply multi-component temporal decay with alpha parameter
        
        # 1. Primary exponential decay based on temporal distance
        primary_decay = math.exp(-self.alpha * temporal_distance_factor)
        
        # 2. Span penalty (longer temporal spans get additional decay)
        span_penalty = math.exp(-self.alpha * 0.5 * span_factor)
        
        # 3. Chronological coherence bonus/penalty
        if chronological_violations == 0:
            coherence_factor = 1.0 + (self.alpha * 0.1)  # Bonus for good chronological order
        else:
            violation_rate = chronological_violations / len(temporal_info['timestamps'])
            coherence_factor = 1.0 - (self.alpha * violation_rate * 0.3)  # Penalty for violations
        
        # 4. Combine all factors
        combined_decay = primary_decay * span_penalty * max(0.1, coherence_factor)
        decayed_flow = base_flow * combined_decay
        
        # Ensure alpha parameter creates meaningful flow variation
        min_flow = base_flow * (0.1 + self.alpha * 0.05)  # Higher alpha preserves more flow for good temporal paths
        decayed_flow = max(min_flow, decayed_flow)
        
        return decayed_flow
    
    def calculate_group_limit(self, group_paths: List[Path], top_k: int, query_time: str) -> int:
        """
        Calculate adaptive group size limit based on temporal relevance
        Paths with higher temporal relevance allow for larger group sizes
        """
        if not group_paths:
            return 1
        
        # Calculate average temporal score for the group
        temporal_scores = []
        for path in group_paths:
            temporal_info = path.get_temporal_info()
            if temporal_info['timestamps']:
                avg_proximity = sum(
                    self.temporal_weighting.temporal_decay_factor(ts, query_time, self.temporal_mode)
                    for ts in temporal_info['timestamps']
                ) / len(temporal_info['timestamps'])
            else:
                avg_proximity = 0.5
            temporal_scores.append(avg_proximity)
        
        avg_temporal_relevance = sum(temporal_scores) / len(temporal_scores)
        
        # Base group limit with temporal adjustment
        unique_endpoints = len({(p.nodes[0].id, p.nodes[-1].id) for p in group_paths if p.nodes})
        base_limit = max(1, top_k // max(unique_endpoints, 1))
        temporal_adjustment = int(avg_temporal_relevance * 2)  # Up to 2x increase for highly relevant paths
        
        return base_limit + temporal_adjustment
    
    def validate_alpha_theta_interaction(self, test_paths: List[Path], query_time: str) -> Dict[str, float]:
        """
        Validate that decay rate (alpha) and pruning threshold (theta) interact effectively
        """
        if not test_paths:
            return {'error': 'No test paths provided'}
        
        # Test different α values to demonstrate interaction
        alpha_values = [0.05, 0.1, 0.2, 0.4]
        theta_values = [0.5, 1.0, 1.5, 2.0]
        
        results = {
            'alpha_sensitivity': [],
            'theta_sensitivity': [],
            'interaction_strength': 0.0,
            'resource_modification_rate': 0.0,
            'temporal_flow_variance': 0.0
        }
        
        original_alpha = self.alpha
        original_theta = self.base_theta
        
        try:
            # Test alpha sensitivity
            for alpha in alpha_values:
                self.alpha = alpha
                pruned = self.flow_based_pruning_with_temporal_weighting(test_paths, 5, query_time)
                results['alpha_sensitivity'].append({
                    'alpha': alpha,
                    'paths_selected': len(pruned),
                    'avg_score': sum(p.score for p in pruned) / len(pruned) if pruned else 0.0
                })
            
            # Test theta sensitivity  
            self.alpha = original_alpha
            for theta in theta_values:
                self.base_theta = theta
                pruned = self.flow_based_pruning_with_temporal_weighting(test_paths, 5, query_time)
                results['theta_sensitivity'].append({
                    'theta': theta,
                    'paths_selected': len(pruned),
                    'avg_score': sum(p.score for p in pruned) / len(pruned) if pruned else 0.0
                })
            
            # Calculate interaction strength
            self.alpha = 0.2
            self.base_theta = 1.5
            high_params = self.flow_based_pruning_with_temporal_weighting(test_paths, 10, query_time)
            
            self.alpha = 0.05
            self.base_theta = 0.5
            low_params = self.flow_based_pruning_with_temporal_weighting(test_paths, 10, query_time)
            
            if high_params and low_params:
                score_difference = abs(sum(p.score for p in high_params) / len(high_params) - 
                                     sum(p.score for p in low_params) / len(low_params))
                results['interaction_strength'] = score_difference
            
            # Measure resource modification effectiveness
            self.alpha = original_alpha
            self.base_theta = original_theta
            self.update_temporal_edge_capacities(test_paths, query_time)
            
            modified_edges = 0
            total_edges = 0
            temporal_flows = []
            
            for path in test_paths:
                for edge in path.edges:
                    total_edges += 1
                    if hasattr(edge, '_original_capacity'):
                        if edge.flow_capacity != edge.original_capacity:
                            modified_edges += 1
                
                temporal_flow = self.calculate_temporal_path_flow(path, query_time)
                temporal_flows.append(temporal_flow)
            
            results['resource_modification_rate'] = modified_edges / max(total_edges, 1)
            results['temporal_flow_variance'] = np.var(temporal_flows) if temporal_flows else 0.0
            
            # Overall validation score
            validation_score = (
                results['interaction_strength'] * 0.4 +
                results['resource_modification_rate'] * 0.3 +
                min(results['temporal_flow_variance'], 1.0) * 0.3
            )
            results['validation_score'] = validation_score
            results['implementation_robust'] = validation_score > 0.3
            
        finally:
            # Restore original parameters
            self.alpha = original_alpha
            self.base_theta = original_theta
        
        return results
"""
Temporal scoring functions designed to improve the reliability scores of PathRAG's paths

We incorporate temporal weighting that add temporal relevance into the structural flow of PathRAG:
    - Temporal decay factors to account for time-based relevance
    - Temporal alignment scores to ensure chronological consistency of paths
    - An enhanced reliability score S(P) that includes temporal dimensions

Mathematical foundations and implementation specifics are detailed for each function - we've included lots of different ones
because temporal questions involve implicit time constraints, and so different question need different temporal relevance patterns 
(e.g. exponential for receny bias, but for hard cutoffs, linear probably works better)
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class TemporalRelevanceMode(Enum):
    """Temporal relevance calculation modes"""
    EXPONENTIAL_DECAY = "exponential_decay"
    LINEAR_DECAY = "linear_decay" 
    GAUSSIAN_PROXIMITY = "gaussian_proximity"
    SIGMOID_TRANSITION = "sigmoid_transition"


@dataclass
class TemporalPath:
    """Represents a path with temporal annotations"""
    nodes: List[str]
    edges: List[Tuple[str, str, str, str]]  # (subj, pred, obj, timestamp)
    timestamps: List[str]
    original_score: float = 0.0
    temporal_score: float = 0.0
    combined_score: float = 0.0

@dataclass
class TemporalWeightingFunction:
    """
    Core temporal weighting function that implements multiple temporal decay models, 
    which is designed to enhance PathRAG's reliability score S(P) with temporal relevance
    
    Instead of TimeR4's binary temporal validity, this implements
    continuous temporal scoring that considers:
        - Temporal proximity to query time
        - Chronological consistency within paths
        - Temporal patterns (day, month, year granularities)
    """
    
    def __init__(self, 
                 decay_rate: float = 0.1,
                 temporal_window: int = 365,  # days
                 chronological_weight: float = 0.3,
                 proximity_weight: float = 0.4,
                 consistency_weight: float = 0.3):
        """
        Initialise temporal weighting parameters
        
        Args:
            decay_rate: Rate of temporal decay (x in exponential decay)
            temporal_window: Time window in days for calculating how relevannt it is
            chronological_weight: Weight for chronological ordering score
            proximity_weight: Weight for temporal proximity score
            consistency_weight: Weight for temporal consistency score
        """
        self.decay_rate = decay_rate
        self.temporal_window = temporal_window
        self.chronological_weight = chronological_weight
        self.proximity_weight = proximity_weight
        self.consistency_weight = consistency_weight
    
    def temporal_decay_factor(self, 
                            timestamp: str, 
                            query_time: str,
                            mode: TemporalRelevanceMode = TemporalRelevanceMode.EXPONENTIAL_DECAY) -> float:
        """
        Calculate temporal decay factor based on the time difference from query
        
        Mathematical Formulation:
        - Exponential: f(t) = e^(-x|t_q - t_e|)
        - Linear: f(t) = max(0, 1 - |t_q - t_e|/W)  
        - Gaussian: f(t) = e^(-(t_q - t_e) squared / (2 * sigma squared))
        - Sigmoid: f(t) = 1/(1 + e^(x(|t_q - t_e| - W/2)))
        
        Where:
        - t_q: query timestamp
        - t_e: event timestamp  
        - x: decay rate parameter
        - W: temporal window
        - sigma: standard deviation for Gaussian
        
        Args:
            timestamp: Event timestamp
            query_time: Query reference time
            mode: Temporal decay mode
            
        Returns:
            Temporal decay factor [0, 1]
        """
        try:
            event_date = datetime.fromisoformat(timestamp.replace('T', ' '))
            query_date = datetime.fromisoformat(query_time.replace('T', ' '))
            time_diff_days = abs((query_date - event_date).days)
            
            if mode == TemporalRelevanceMode.EXPONENTIAL_DECAY:
                # Exponential decay so that more recent events aare weighted higher
                return math.exp(-self.decay_rate * time_diff_days / 365.0)
                
            elif mode == TemporalRelevanceMode.LINEAR_DECAY:
                # Linear decay within temporal window
                if time_diff_days > self.temporal_window:
                    return 0.0
                return max(0.0, 1.0 - time_diff_days / self.temporal_window)
                
            elif mode == TemporalRelevanceMode.GAUSSIAN_PROXIMITY:
                # Gaussian centred on query time
                sigma = self.temporal_window / 4.0  # Standard deviation
                return math.exp(-0.5 * (time_diff_days / sigma) ** 2)
                
            elif mode == TemporalRelevanceMode.SIGMOID_TRANSITION:
                # Sigmoid transition for soft temporal boundaries
                midpoint = self.temporal_window / 2.0
                # Prevent overflow by clamping the exponent
                exponent = self.decay_rate * (time_diff_days - midpoint)
                if exponent > 500:  # Prevent overflow
                    return 0.0
                elif exponent < -500:
                    return 1.0
                return 1.0 / (1.0 + math.exp(exponent))
                
        except (ValueError, TypeError):
            # Handle invalid timestamps 
            return 0.5  # Neutral score for unparseable timestamps
    
    def chronological_alignment_score(self, path: TemporalPath) -> float:
        """
        Calculate chronological score for temporal consistency within paths
        
        Mathematical Formulation:
        S_chrono(P) = (1/|P|-1) * Sigma(sum of)[i=1 to |P|-1] indicator(t_i <= t_{i+1}) * w_i
        
        Where:
        - P: temporal path with timestamps t_1, t_2, t_n
        - indicator(t_i <= t_{i+1}): 1 if chronological order maintained, 0 otherwise
        - w_i: weight based on temporal gap between consecutive events
        
        Enhanced with temporal gap weighting:
        w_i = e^(-o * |t_{i+1} - t_i|) where o controls sensitivity to temporal gaps
        
        Args:
            path: TemporalPath object with timestamps
            
        Returns:
            Chronological alignment score [0, 1]
        """
        if len(path.timestamps) < 2:
            return 1.0  # Single event is always chronologically consistent
            
        total_weight = 0.0
        chronological_score = 0.0
        gap_sensitivity = 0.001  # alpha parameter for gap weighting
        
        for i in range(len(path.timestamps) - 1):
            try:
                current_time = datetime.fromisoformat(path.timestamps[i].replace('T', ' '))
                next_time = datetime.fromisoformat(path.timestamps[i + 1].replace('T', ' '))
                
                # Calculate temporal gap in days
                time_gap = abs((next_time - current_time).days)
                gap_weight = math.exp(-gap_sensitivity * time_gap)
                
                # Check chronological ordering
                if current_time <= next_time:
                    chronological_score += gap_weight
                total_weight += gap_weight
                
            except (ValueError, TypeError):
                # Handle invalid timestamps - assign neutral weight
                total_weight += 0.5
                chronological_score += 0.25
        
        return chronological_score / total_weight if total_weight > 0 else 0.0
    
    def temporal_proximity_score(self, 
                                path: TemporalPath, 
                                query_time: str,
                                aggregation: str = "harmonic_mean") -> float:
        """
        Calculate temporal proximity score for the entire path relative to query time
        
        Mathematical Formulation:
        Multiple aggregation strategies:
        
        1. Harmonic Mean: S_prox(P) = n / sigma(sumf of)[i=1 to n] (1/f(t_i))
        2. Geometric Mean: S_prox(P) = (Pi(product)[i=1 to n] f(t_i))^(1/n)  
        3. Weighted Average: S_prox(P) = sigma(sum of)[i=1 to n] w_i * f(t_i) / sigma(sum of)w_i
        4. Maximum: S_prox(P) = max{f(t_i) for i in 1..n}
        
        Where f(t_i) is the temporal decay factor for timestamp t_i
        
        Args:
            path: TemporalPath object
            query_time: Reference time for proximity calculation
            aggregation: Aggregation method for multiple timestamps
            
        Returns:
            Temporal proximity score [0, 1]
        """
        if not path.timestamps:
            return 0.5  # Neutral score for paths without timestamps
            
        proximity_scores = []
        for timestamp in path.timestamps:
            score = self.temporal_decay_factor(timestamp, query_time)
            proximity_scores.append(score)
        
        if not proximity_scores:
            return 0.5
            
        if aggregation == "harmonic_mean":
            # Harmonic mean - sensitive to low values, good for identifying outliers
            harmonic_sum = sum(1.0/max(score, 1e-6) for score in proximity_scores)
            return len(proximity_scores) / harmonic_sum
            
        elif aggregation == "geometric_mean":
            # Geometric mean - more balanced approach
            product = 1.0
            for score in proximity_scores:
                product *= max(score, 1e-6)
            return product ** (1.0 / len(proximity_scores))
            
        elif aggregation == "weighted_average":
            # Weighted by chronological position (later events weighted more)
            weights = np.linspace(0.5, 1.0, len(proximity_scores))
            return np.average(proximity_scores, weights=weights)
            
        elif aggregation == "maximum":
            # Maximum proximity - optimistic approach
            return max(proximity_scores)
            
        else:
            # Default: arithmetic mean
            return sum(proximity_scores) / len(proximity_scores)
    
    def temporal_consistency_score(self, path: TemporalPath) -> float:
        """
        Calculate temporal consistency score measuring the internal temporal 'coherence'
        
        Mathematical Formulation:
        S_consist(P) = 1 - (o_normalised + outlier_penalty)
        
        Where:
        - o_normalised: Normalised standard deviation of time intervals
        - outlier_penalty: Penalty for timestamps far from temporal cluster
        
        Components:
        1. Temporal spread: How spread out are the different timestamps?
        2. Outlier detection: Are there any temporally disconnected events?
        3. Interval consistency: Are time gaps between events 'reasonable'?
        
        Args:
            path: TemporalPath object
            
        Returns:
            Temporal consistency score [0, 1]
        """
        if len(path.timestamps) < 2:
            return 1.0
            
        try:
            # Convert timestamps to numerical values (days since epoch)
            timestamp_values = []
            for ts in path.timestamps:
                dt = datetime.fromisoformat(ts.replace('T', ' '))
                days_since_epoch = (dt - datetime(1970, 1, 1)).days
                timestamp_values.append(days_since_epoch)
            
            # Calculate temporal spread (normalised standard deviation)
            mean_time = np.mean(timestamp_values)
            std_time = np.std(timestamp_values)
            time_range = max(timestamp_values) - min(timestamp_values)
            
            # Normalise standard deviation by range (handles single-day events)
            if time_range > 0:
                normalised_std = std_time / time_range
            else:
                normalised_std = 0.0
            
            # Outlier detection using IQR method
            q75, q25 = np.percentile(timestamp_values, [75, 25])
            iqr = q75 - q25
            outlier_penalty = 0.0
            
            if iqr > 0:
                for value in timestamp_values:
                    if value < (q25 - 1.5 * iqr) or value > (q75 + 1.5 * iqr):
                        outlier_penalty += 0.1  # Penalty per outlier
            
            # Calculate final consistency score
            consistency = 1.0 - min(1.0, normalised_std + outlier_penalty)
            return max(0.0, consistency)
            
        except (ValueError, TypeError):
            return 0.5  # Neutral score for invalid timestamps
    
    def enhanced_reliability_score(self, 
                                  path: TemporalPath,
                                  query_time: str,
                                  original_pathrag_score: float) -> float:
        """
        Calculate enhanced reliability score S'(P) combining PathRAG's structural
        score with temporal dimensions
        
        Mathematical Formulation:
        S'(P) = α * S_PathRAG(P) + beta * S_temporal(P)
        
        Where:
        S_temporal(P) = w1 * S_chrono(P) + w2 * S_prox(P) + w3 * S_consist(P)
        
        And:
        - o, beta: Balancing parameters between structural and temporal components
        - w1, w2, w3: Weights for different temporal aspects
        - S_PathRAG(P): Original PathRAG reliability score
        - S_chrono(P): Chronological alignment score
        - S_prox(P): Temporal proximity score  
        - S_consist(P): Temporal consistency score
        
        Args:
            path: TemporalPath object
            query_time: Reference time for tje temporal calculations
            original_pathrag_score: Original PathRAG reliability score S(P)
            
        Returns:
            Enhanced reliability score S'(P) [0, 1]
        """
        # Calculate individual temporal components
        chrono_score = self.chronological_alignment_score(path)
        proximity_score = self.temporal_proximity_score(path, query_time)
        consistency_score = self.temporal_consistency_score(path)
        
        # Combine temporal components
        temporal_score = (
            self.chronological_weight * chrono_score +
            self.proximity_weight * proximity_score +
            self.consistency_weight * consistency_score
        )
        
        # Balance structural and temporal components
        # Higher temporal weight for temporally-rich paths
        temporal_richness = min(1.0, len(path.timestamps) / 3.0)  # Normalise by typical path length
        structural_weight = 0.7 - 0.2 * temporal_richness  # Adaptive weighting
        temporal_weight = 0.3 + 0.2 * temporal_richness
        
        # Calculate final enhanced score
        enhanced_score = (
            structural_weight * original_pathrag_score +
            temporal_weight * temporal_score
        )
        
        # Store components for analysis
        path.temporal_score = temporal_score
        path.combined_score = enhanced_score
        
        return enhanced_score


class TemporalPathRanker:
    """
    Temporal path ranking system that integrates multiple temporal
    signals for a more robust path evaluation
    """
    
    def __init__(self, weighting_function: TemporalWeightingFunction):
        self.weighting_function = weighting_function
    
    def rank_paths(self, 
                   paths: List[TemporalPath],
                   query_time: str,
                   top_k: int = 10) -> List[Tuple[TemporalPath, float]]:
        """
        Rank paths using enhanced temporal reliability scores
        
        Args:
            paths: List of TemporalPath objects to rank
            query_time: Reference time for temporal calculations
            top_k: Number of top paths to return
            
        Returns:
            List of (path, enhanced_score) tuples, sorted by score descending
        """
        scored_paths = []
        
        for path in paths:
            enhanced_score = self.weighting_function.enhanced_reliability_score(
                path, query_time, path.original_score
            )
            scored_paths.append((path, enhanced_score))
        
        # Sort by enhanced score (descending)
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        return scored_paths[:top_k]
    
    def analyse_temporal_patterns(self, paths: List[TemporalPath]) -> Dict[str, float]:
        """
        Analyse temporal patterns across a set of different paths
        
        Returns:
            Dictionary of temporal pattern statistics
        """
        if not paths:
            return {}
        
        all_timestamps = []
        chronological_scores = []
        consistency_scores = []
        
        for path in paths:
            all_timestamps.extend(path.timestamps)
            chronological_scores.append(
                self.weighting_function.chronological_alignment_score(path)
            )
            consistency_scores.append(
                self.weighting_function.temporal_consistency_score(path)
            )
        
        # Convert timestamps to years for analysis
        years = []
        for ts in all_timestamps:
            try:
                year = datetime.fromisoformat(ts.replace('T', ' ')).year
                years.append(year)
            except (ValueError, TypeError):
                continue
        
        return {
            "temporal_span_years": max(years) - min(years) if years else 0,
            "avg_chronological_score": np.mean(chronological_scores),
            "avg_consistency_score": np.mean(consistency_scores),
            "temporal_density": len(all_timestamps) / len(paths),
            "most_common_year": max(set(years), key=years.count) if years else None
        }
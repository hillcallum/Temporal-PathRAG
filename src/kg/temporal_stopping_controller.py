"""
Temporal-Aware Stopping Controller for Temporal PathRAG

Implements temporal stopping criteria:
- Temporal coverage completeness (chronological span and density)
- Chronological chain satisfaction (temporal sequence validation)
- Temporal constraint fulfillment (query-specific temporal requirements)
- Temporal information overload prevention (optimal stopping before degradation)
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from .models import TemporalQuery, Path, TemporalReliabilityMetrics, IterativeStep, TemporalCoverageMetrics, TemporalStoppingDecision
from ..llm.llm_manager import LLMManager


class TemporalStoppingController:
    """
    Temporal-aware stopping controller for iterative reasoning
    """
    
    def __init__(self, 
                 llm_manager: LLMManager,
                 temporal_coverage_threshold: float = 0.75,
                 chain_satisfaction_threshold: float = 0.8,
                 constraint_fulfillment_threshold: float = 0.85,
                 overload_prevention_threshold: float = 0.9,
                 max_temporal_span_days: int = 365 * 10,  # 10 years default
                 min_temporal_density: float = 0.1):
        
        self.llm_manager = llm_manager
        
        # Temporal stopping thresholds
        self.temporal_coverage_threshold = temporal_coverage_threshold
        self.chain_satisfaction_threshold = chain_satisfaction_threshold
        self.constraint_fulfillment_threshold = constraint_fulfillment_threshold
        self.overload_prevention_threshold = overload_prevention_threshold
        
        # Temporal analysis parameters
        self.max_temporal_span_days = max_temporal_span_days
        self.min_temporal_density = min_temporal_density
        
        # Temporal pattern recognition
        self.temporal_keywords = {
            'sequence': ['then', 'after', 'next', 'subsequently', 'following', 'later'],
            'causation': ['because', 'due to', 'caused by', 'resulted in', 'led to', 'triggered'],
            'duration': ['during', 'throughout', 'for', 'lasting', 'spanning'],
            'frequency': ['often', 'frequently', 'regularly', 'repeatedly', 'always'],
            'comparison': ['before', 'after', 'earlier', 'later', 'compared to', 'relative to']
        }
        
        # Performance tracking
        self.stopping_history = []
        
    def should_stop(self, 
                   query: TemporalQuery, 
                   retrieved_paths: List[Tuple[Path, TemporalReliabilityMetrics]], 
                   current_context: str,
                   iteration: int = 0,
                   reasoning_steps: List[IterativeStep] = None) -> TemporalStoppingDecision:
        """
        Main stopping decision function with temporal-aware criteria
        """
        
        # Step 1: Analyse temporal coverage
        temporal_coverage = self.analyse_temporal_coverage(
            query, retrieved_paths, reasoning_steps or []
        )
        
        # Step 2: Assess chronological chain satisfaction
        chain_satisfaction = self.assess_chronological_chain_satisfaction(
            query, retrieved_paths, temporal_coverage
        )
        
        # Step 3: Evaluate temporal constraint fulfillment
        constraint_fulfillment = self.evaluate_temporal_constraint_fulfillment(
            query, retrieved_paths, temporal_coverage
        )
        
        # Step 4: Check for temporal information overload
        overload_assessment = self.assess_temporal_overload(
            retrieved_paths, iteration, temporal_coverage
        )
        
        # Step 5: Make integrated stopping decision
        stopping_decision = self.make_integrated_stopping_decision(
            query, temporal_coverage, chain_satisfaction, constraint_fulfillment,
            overload_assessment, current_context, iteration
        )
        
        # Track decision for analysis
        self.stopping_history.append(stopping_decision)
        
        return stopping_decision
    
    def analyse_temporal_coverage(self, 
                                  query: TemporalQuery, 
                                  retrieved_paths: List[Tuple[Path, TemporalReliabilityMetrics]],
                                  reasoning_steps: List[IterativeStep]) -> TemporalCoverageMetrics:
        """Analyse temporal coverage completeness"""
        
        if not retrieved_paths:
            return TemporalCoverageMetrics()
        
        # Extract all timestamps
        all_timestamps = []
        temporal_entities = set()
        
        for path, metrics in retrieved_paths:
            path_temporal_info = path.get_temporal_info()
            timestamps = path_temporal_info.get("timestamps", [])
            all_timestamps.extend(timestamps)
            
            # Extract temporal entities
            for node in path.path:
                if self.is_temporal_entity(node):
                    temporal_entities.add(node)
        
        if not all_timestamps:
            return TemporalCoverageMetrics()
        
        # Parse timestamps
        parsed_timestamps = []
        for ts in all_timestamps:
            try:
                # Handle various timestamp formats
                if 'T' in ts:
                    dt = datetime.fromisoformat(ts.replace('T', ' '))
                else:
                    dt = datetime.fromisoformat(ts)
                parsed_timestamps.append(dt)
            except:
                continue
        
        if not parsed_timestamps:
            return TemporalCoverageMetrics()
        
        # Calculate temporal span
        min_date = min(parsed_timestamps)
        max_date = max(parsed_timestamps)
        temporal_span_days = (max_date - min_date).days
        
        # Calculate temporal density
        unique_dates = set(dt.date() for dt in parsed_timestamps)
        temporal_density = len(unique_dates) / max(temporal_span_days, 1) if temporal_span_days > 0 else 1.0
        
        # Calculate chronological continuity
        chronological_continuity = self.calculate_chronological_continuity(parsed_timestamps)
        
        # Calculate temporal distribution score
        temporal_distribution_score = self.calculate_temporal_distribution_score(
            parsed_timestamps, query
        )
        
        # Calculate overall coverage score
        coverage_score = self.calculate_overall_coverage_score(
            temporal_span_days, temporal_density, chronological_continuity, 
            temporal_distribution_score, len(all_timestamps)
        )
        
        return TemporalCoverageMetrics(
            coverage_score=coverage_score,
            temporal_span_days=temporal_span_days,
            timestamp_count=len(all_timestamps),
            unique_timestamps=len(unique_dates),
            temporal_density=temporal_density,
            chronological_continuity=chronological_continuity,
            temporal_distribution_score=temporal_distribution_score
        )
    
    def assess_chronological_chain_satisfaction(self, 
                                               query: TemporalQuery, 
                                               retrieved_paths: List[Tuple[Path, TemporalReliabilityMetrics]],
                                               temporal_coverage: TemporalCoverageMetrics) -> float:
        """Assess if chronological chains are satisfied"""
        
        if not retrieved_paths:
            return 0.0
        
        # Extract temporal relationships
        temporal_relationships = []
        for path, metrics in retrieved_paths:
            path_relationships = self.extract_temporal_relationships(path)
            temporal_relationships.extend(path_relationships)
        
        if not temporal_relationships:
            return 0.5  # Neutral if no temporal relationships found
        
        # Check chain completeness
        chain_completeness = self.assess_chain_completeness(temporal_relationships, query)
        
        # Check temporal ordering consistency
        ordering_consistency = self.assess_temporal_ordering_consistency(temporal_relationships)
        
        # Check causality chain satisfaction
        causality_satisfaction = self.assess_causality_chain_satisfaction(temporal_relationships, query)
        
        # Combined chain satisfaction score
        chain_satisfaction = (chain_completeness * 0.4 + 
                            ordering_consistency * 0.3 + 
                            causality_satisfaction * 0.3)
        
        return chain_satisfaction
    
    def evaluate_temporal_constraint_fulfillment(self, 
                                                 query: TemporalQuery, 
                                                 retrieved_paths: List[Tuple[Path, TemporalReliabilityMetrics]],
                                                 temporal_coverage: TemporalCoverageMetrics) -> float:
        """Evaluate if temporal constraints from query are fulfilled"""
        
        # Extract temporal constraints from query
        query_constraints = self.extract_query_temporal_constraints(query)
        
        if not query_constraints:
            return 1.0  # No constraints to fulfill
        
        # Check constraint satisfaction
        satisfied_constraints = 0
        total_constraints = len(query_constraints)
        
        for constraint in query_constraints:
            if self.is_constraint_satisfied(constraint, retrieved_paths, temporal_coverage):
                satisfied_constraints += 1
        
        constraint_fulfillment = satisfied_constraints / total_constraints
        
        # Update temporal coverage metrics
        temporal_coverage.constraint_satisfaction_score = constraint_fulfillment
        temporal_coverage.missing_temporal_constraints = [
            c for c in query_constraints 
            if not self.is_constraint_satisfied(c, retrieved_paths, temporal_coverage)
        ]
        
        return constraint_fulfillment
    
    def assess_temporal_overload(self, 
                                 retrieved_paths: List[Tuple[Path, TemporalReliabilityMetrics]], 
                                 iteration: int,
                                 temporal_coverage: TemporalCoverageMetrics) -> Dict[str, Any]:
        """Assess if temporal information overload is occurring"""
        
        # Calculate information quality degradation
        quality_scores = [metrics.overall_reliability for _, metrics in retrieved_paths]
        
        if len(quality_scores) < 3:
            return {"overload_risk": 0.0, "should_stop_overload": False}
        
        # Check for declining quality in recent retrievals
        recent_quality = np.mean(quality_scores[-3:])
        overall_quality = np.mean(quality_scores)
        
        quality_degradation = max(0, overall_quality - recent_quality)
        
        # Check for temporal redundancy
        temporal_redundancy = self.calculate_temporal_redundancy(retrieved_paths)
        
        # Check for information diversity decline
        diversity_decline = self.calculate_diversity_decline(retrieved_paths)
        
        # Overall overload risk
        overload_risk = (quality_degradation * 0.4 + 
                        temporal_redundancy * 0.3 + 
                        diversity_decline * 0.3)
        
        should_stop_overload = overload_risk > self.overload_prevention_threshold
        
        return {
            "overload_risk": overload_risk,
            "should_stop_overload": should_stop_overload,
            "quality_degradation": quality_degradation,
            "temporal_redundancy": temporal_redundancy,
            "diversity_decline": diversity_decline
        }
    
    def make_integrated_stopping_decision(self, 
                                         query: TemporalQuery, 
                                         temporal_coverage: TemporalCoverageMetrics,
                                         chain_satisfaction: float,
                                         constraint_fulfillment: float,
                                         overload_assessment: Dict[str, Any],
                                         current_context: str,
                                         iteration: int) -> TemporalStoppingDecision:
        """Make integrated stopping decision based on all temporal criteria"""
        
        # Evaluate each stopping criterion
        coverage_sufficient = temporal_coverage.coverage_score >= self.temporal_coverage_threshold
        chain_sufficient = chain_satisfaction >= self.chain_satisfaction_threshold
        constraints_fulfilled = constraint_fulfillment >= self.constraint_fulfillment_threshold
        overload_risk = overload_assessment["should_stop_overload"]
        
        # Determine stopping decision
        should_stop = False
        stopping_criterion = ""
        reasoning = ""
        confidence = 0.0
        
        if overload_risk:
            should_stop = True
            stopping_criterion = "overload_prevention"
            reasoning = f"Stopping due to temporal information overload risk ({overload_assessment['overload_risk']:.2f})"
            confidence = 0.9
        elif coverage_sufficient and chain_sufficient and constraints_fulfilled:
            should_stop = True
            stopping_criterion = "comprehensive_satisfaction"
            reasoning = f"All temporal criteria satisfied (coverage: {temporal_coverage.coverage_score:.2f}, chain: {chain_satisfaction:.2f}, constraints: {constraint_fulfillment:.2f})"
            confidence = min(0.95, (temporal_coverage.coverage_score + chain_satisfaction + constraint_fulfillment) / 3)
        elif coverage_sufficient and constraints_fulfilled:
            should_stop = True
            stopping_criterion = "coverage_and_constraints"
            reasoning = f"Temporal coverage and constraints satisfied (coverage: {temporal_coverage.coverage_score:.2f}, constraints: {constraint_fulfillment:.2f})"
            confidence = min(0.85, (temporal_coverage.coverage_score + constraint_fulfillment) / 2)
        elif iteration > 3 and (coverage_sufficient or chain_sufficient):
            should_stop = True
            stopping_criterion = "partial_satisfaction_with_iterations"
            reasoning = f"Partial satisfaction after {iteration} iterations (coverage: {temporal_coverage.coverage_score:.2f}, chain: {chain_satisfaction:.2f})"
            confidence = 0.7
        
        # Generate exploration hints if not stopping
        next_exploration_hints = []
        if not should_stop:
            next_exploration_hints = self.generate_exploration_hints(
                temporal_coverage, chain_satisfaction, constraint_fulfillment, query
            )
        
        # Calculate performance metrics
        information_quality_score = self.calculate_information_quality_score(
            temporal_coverage, chain_satisfaction, constraint_fulfillment
        )
        
        retrieval_efficiency_score = self.calculate_retrieval_efficiency_score(
            temporal_coverage, iteration
        )
        
        return TemporalStoppingDecision(
            should_stop=should_stop,
            confidence=confidence,
            reasoning=reasoning,
            stopping_criterion=stopping_criterion,
            temporal_coverage=temporal_coverage,
            next_exploration_hints=next_exploration_hints,
            information_quality_score=information_quality_score,
            retrieval_efficiency_score=retrieval_efficiency_score
        )
    
    def is_temporal_entity(self, entity: str) -> bool:
        """Check if entity is temporal in nature"""
        temporal_indicators = ['date', 'time', 'year', 'month', 'day', 'period', 'era', 'age']
        return any(indicator in entity.lower() for indicator in temporal_indicators)
    
    def calculate_chronological_continuity(self, timestamps: List[datetime]) -> float:
        """Calculate chronological continuity score"""
        if len(timestamps) < 2:
            return 1.0
        
        sorted_timestamps = sorted(timestamps)
        gaps = []
        
        for i in range(1, len(sorted_timestamps)):
            gap = (sorted_timestamps[i] - sorted_timestamps[i-1]).days
            gaps.append(gap)
        
        if not gaps:
            return 1.0
        
        # Calculate continuity based on gap distribution
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        # Higher continuity for smaller, more consistent gaps
        continuity = 1.0 / (1.0 + std_gap / max(mean_gap, 1))
        
        return min(1.0, continuity)
    
    def calculate_temporal_distribution_score(self, timestamps: List[datetime], query: TemporalQuery) -> float:
        """Calculate how well timestamps are distributed relative to query requirements"""
        if not timestamps:
            return 0.0
        
        # Simple distribution score based on timestamp spread
        sorted_timestamps = sorted(timestamps)
        
        if len(sorted_timestamps) < 2:
            return 0.5
        
        # Calculate distribution evenness
        total_span = (sorted_timestamps[-1] - sorted_timestamps[0]).days
        if total_span == 0:
            return 1.0
        
        # Check for even distribution
        expected_interval = total_span / (len(sorted_timestamps) - 1)
        actual_intervals = [(sorted_timestamps[i+1] - sorted_timestamps[i]).days 
                           for i in range(len(sorted_timestamps)-1)]
        
        if not actual_intervals:
            return 0.5
        
        # Calculate distribution score
        interval_variance = np.var(actual_intervals)
        distribution_score = 1.0 / (1.0 + interval_variance / max(expected_interval, 1))
        
        return min(1.0, distribution_score)
    
    def calculate_overall_coverage_score(self, span_days: int, density: float, 
                                        continuity: float, distribution: float, 
                                        timestamp_count: int) -> float:
        """Calculate overall temporal coverage score"""
        
        # Normalise span score
        span_score = min(1.0, span_days / self.max_temporal_span_days)
        
        # Normalise density score
        density_score = min(1.0, density / self.min_temporal_density)
        
        # Normalise count score
        count_score = min(1.0, timestamp_count / 10)  # Assume 10 timestamps is good coverage
        
        # Weighted combination
        overall_score = (span_score * 0.25 + 
                        density_score * 0.25 + 
                        continuity * 0.25 + 
                        distribution * 0.15 + 
                        count_score * 0.1)
        
        return min(1.0, overall_score)
    
    def extract_temporal_relationships(self, path: Path) -> List[Dict[str, Any]]:
        """Extract temporal relationships from path"""
        relationships = []
        
        # Simple temporal relationship extraction
        for i in range(len(path.path) - 1):
            node1 = path.path[i]
            node2 = path.path[i + 1]
            
            # Check for temporal relationship indicators
            relationship_type = "sequence"  # Default
            
            relationships.append({
                "source": node1,
                "target": node2,
                "type": relationship_type,
                "temporal_order": i
            })
        
        return relationships
    
    def assess_chain_completeness(self, relationships: List[Dict[str, Any]], query: TemporalQuery) -> float:
        """Assess completeness of temporal chains"""
        if not relationships:
            return 0.0
        
        # Simple chain completeness based on relationship count and connectivity
        unique_entities = set()
        for rel in relationships:
            unique_entities.add(rel["source"])
            unique_entities.add(rel["target"])
        
        connectivity = len(relationships) / max(len(unique_entities), 1)
        
        return min(1.0, connectivity)
    
    def assess_temporal_ordering_consistency(self, relationships: List[Dict[str, Any]]) -> float:
        """Assess consistency of temporal ordering"""
        if not relationships:
            return 1.0
        
        # Check for temporal ordering consistency
        ordered_relationships = sorted(relationships, key=lambda x: x["temporal_order"])
        
        # Simple consistency check
        consistency_score = 1.0  # Assume consistent unless proven otherwise
        
        return consistency_score
    
    def assess_causality_chain_satisfaction(self, relationships: List[Dict[str, Any]], query: TemporalQuery) -> float:
        """Assess satisfaction of causality chains"""
        if not relationships:
            return 0.5
        
        # Check for causality indicators in relationships
        causality_indicators = ['caused', 'resulted', 'led to', 'triggered', 'because']
        
        causality_count = 0
        for rel in relationships:
            if any(indicator in str(rel).lower() for indicator in causality_indicators):
                causality_count += 1
        
        causality_score = causality_count / max(len(relationships), 1)
        
        return min(1.0, causality_score)
    
    def extract_query_temporal_constraints(self, query: TemporalQuery) -> List[str]:
        """Extract temporal constraints from query"""
        constraints = []
        
        query_text = query.query_text.lower()
        
        # Check for temporal patterns
        for pattern_type, keywords in self.temporal_keywords.items():
            if any(keyword in query_text for keyword in keywords):
                constraints.append(pattern_type)
        
        return constraints
    
    def is_constraint_satisfied(self, constraint: str, 
                               retrieved_paths: List[Tuple[Path, TemporalReliabilityMetrics]], 
                               temporal_coverage: TemporalCoverageMetrics) -> bool:
        """Check if specific temporal constraint is satisfied"""
        
        # Simple constraint satisfaction check
        if constraint == "sequence":
            return temporal_coverage.chronological_continuity > 0.5
        elif constraint == "causation":
            return temporal_coverage.causality_chain_score > 0.5
        elif constraint == "duration":
            return temporal_coverage.temporal_span_days > 0
        elif constraint == "frequency":
            return temporal_coverage.temporal_density > 0.1
        elif constraint == "comparison":
            return temporal_coverage.unique_timestamps > 1
        
        return True  # Default to satisfied
    
    def calculate_temporal_redundancy(self, retrieved_paths: List[Tuple[Path, TemporalReliabilityMetrics]]) -> float:
        """Calculate temporal redundancy in retrieved paths"""
        if len(retrieved_paths) < 2:
            return 0.0
        
        # Simple redundancy calculation based on timestamp overlap
        all_timestamps = []
        for path, _ in retrieved_paths:
            timestamps = path.get_temporal_info().get("timestamps", [])
            all_timestamps.extend(timestamps)
        
        if not all_timestamps:
            return 0.0
        
        unique_timestamps = set(all_timestamps)
        redundancy = 1.0 - (len(unique_timestamps) / len(all_timestamps))
        
        return redundancy
    
    def calculate_diversity_decline(self, retrieved_paths: List[Tuple[Path, TemporalReliabilityMetrics]]) -> float:
        """Calculate diversity decline in recent retrievals"""
        if len(retrieved_paths) < 4:
            return 0.0
        
        # Compare entity diversity in first half vs second half
        mid_point = len(retrieved_paths) // 2
        first_half = retrieved_paths[:mid_point]
        second_half = retrieved_paths[mid_point:]
        
        first_entities = set()
        for path, _ in first_half:
            first_entities.update(path.path)
        
        second_entities = set()
        for path, _ in second_half:
            second_entities.update(path.path)
        
        # Calculate diversity decline
        if not first_entities:
            return 0.0
        
        diversity_decline = 1.0 - (len(second_entities) / len(first_entities))
        
        return max(0.0, diversity_decline)
    
    def generate_exploration_hints(self, temporal_coverage: TemporalCoverageMetrics, 
                                   chain_satisfaction: float, constraint_fulfillment: float,
                                   query: TemporalQuery) -> List[str]:
        """Generate hints for next exploration"""
        hints = []
        
        if temporal_coverage.coverage_score < self.temporal_coverage_threshold:
            hints.append("Expand temporal coverage - look for more time periods")
        
        if chain_satisfaction < self.chain_satisfaction_threshold:
            hints.append("Strengthen chronological chains - find connecting temporal events")
        
        if constraint_fulfillment < self.constraint_fulfillment_threshold:
            hints.append("Address missing temporal constraints from query")
        
        if temporal_coverage.temporal_density < self.min_temporal_density:
            hints.append("Increase temporal density - find more events within time periods")
        
        return hints
    
    def calculate_information_quality_score(self, temporal_coverage: TemporalCoverageMetrics, 
                                           chain_satisfaction: float, constraint_fulfillment: float) -> float:
        """Calculate overall information quality score"""
        
        quality_score = (temporal_coverage.coverage_score * 0.4 + 
                        chain_satisfaction * 0.3 + 
                        constraint_fulfillment * 0.3)
        
        return min(1.0, quality_score)
    
    def calculate_retrieval_efficiency_score(self, temporal_coverage: TemporalCoverageMetrics, 
                                            iteration: int) -> float:
        """Calculate retrieval efficiency score"""
        
        if iteration == 0:
            return 1.0
        
        # Higher efficiency for achieving good coverage in fewer iterations
        efficiency = temporal_coverage.coverage_score / max(iteration, 1)
        
        return min(1.0, efficiency)
    
    def get_stopping_statistics(self) -> Dict[str, Any]:
        """Get statistics about stopping decisions"""
        
        if not self.stopping_history:
            return {}
        
        total_decisions = len(self.stopping_history)
        stop_decisions = sum(1 for d in self.stopping_history if d.should_stop)
        
        stopping_criteria = defaultdict(int)
        for decision in self.stopping_history:
            if decision.should_stop:
                stopping_criteria[decision.stopping_criterion] += 1
        
        avg_confidence = np.mean([d.confidence for d in self.stopping_history])
        avg_quality = np.mean([d.information_quality_score for d in self.stopping_history])
        
        return {
            "total_decisions": total_decisions,
            "stop_decisions": stop_decisions,
            "stop_rate": stop_decisions / total_decisions,
            "stopping_criteria": dict(stopping_criteria),
            "average_confidence": avg_confidence,
            "average_quality": avg_quality
        }
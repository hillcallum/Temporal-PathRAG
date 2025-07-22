"""
Temporal Iterative Reasoner for Temporal PathRAG
Based on KG-IRAG methodology with TKG integration

We are aiming to implement iterative reasoning for complex temporal queries by:
- Decomposing temporal queries into sub-questions
- Iteratively retrieving relevant paths from TKG
- Using temporal context to guide next iteration
- Stopping when sufficient evidence is gathered
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..core.tkg_query_engine import TKGQueryEngine
from ..models import TemporalQuery, Path, QueryResult, TemporalReliabilityMetrics, IterativeStep, IterativeResult
from .temporal_stopping_controller import TemporalStoppingController
from ...llm.llm_manager import LLMManager


class TemporalQueryDecomposer:
    """Decomposes complex temporal queries into manageable sub-questions"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        
        # Temporal decomposition patterns
        self.temporal_patterns = {
            'sequence': ['then', 'after', 'next', 'subsequently', 'following'],
            'causation': ['because', 'due to', 'caused by', 'resulted in', 'led to'],
            'time_range': ['during', 'between', 'from', 'to', 'within'],
            'comparison': ['before', 'after', 'earlier', 'later', 'compared to'],
            'condition': ['if', 'when', 'while', 'unless', 'given that']
        }
    
    def decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """Decompose complex temporal query into sub-questions"""
        
        # Create decomposition prompt
        decomposition_prompt = f"""
        Analyse this temporal query and break it down into logical sub-questions that can be answered sequentially:
        
        Query: {query}
        
        Consider:
        1. What entities are involved?
        2. What temporal constraints exist?
        3. What relationships need to be explored?
        4. What is the logical sequence of information needed?
        
        Return a JSON list of sub-questions with their temporal constraints:
        [
            {{
                "sub_query": "specific question",
                "temporal_constraints": {{"time_range": "...", "temporal_direction": "..."}},
                "depends_on": ["list of previous sub-questions it depends on"],
                "exploration_focus": "entity/relationship/time"
            }}
        ]
        """
        

        try:
            response = self.llm_manager.generate(decomposition_prompt)
            
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            json_str = response[start_idx:end_idx]
            
            sub_questions = json.loads(json_str)
            return sub_questions
            
        except Exception as e:
            print(f"Error in query decomposition: {e}")
            # Fallback: simple decomposition
            return self.simple_decomposition(query)
    
    def simple_decomposition(self, query: str) -> List[Dict[str, Any]]:
        """Simple fallback decomposition based on patterns"""
        
        # Basic pattern matching for temporal queries
        sub_questions = []
        
        # Look for entities
        words = query.split()
        entities = [word for word in words if word[0].isupper()]
        
        if entities:
            sub_questions.append({
                "sub_query": f"What information is available about {', '.join(entities)}?",
                "temporal_constraints": {},
                "depends_on": [],
                "exploration_focus": "entity"
            })
        
        # Look for temporal patterns
        for pattern_type, keywords in self.temporal_patterns.items():
            if any(keyword in query.lower() for keyword in keywords):
                sub_questions.append({
                    "sub_query": f"What {pattern_type} relationships exist in this context?",
                    "temporal_constraints": {"pattern_type": pattern_type},
                    "depends_on": [0] if sub_questions else [],
                    "exploration_focus": "relationship"
                })
        
        return sub_questions if sub_questions else [{"sub_query": query, "temporal_constraints": {}, "depends_on": [], "exploration_focus": "general"}]


class TemporalIterativeReasoner:
    """
    Main iterative reasoning component that coordinates the iterative process
    Integrates with existing TKG system for temporal path retrieval
    """
    
    def __init__(self, 
                 tkg_query_engine: TKGQueryEngine, 
                 llm_manager: LLMManager,
                 max_iterations: int = 5,
                 convergence_threshold: float = 0.8,
                 temporal_coverage_threshold: float = 0.7):
        
        self.tkg_engine = tkg_query_engine
        self.llm_manager = llm_manager
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.temporal_coverage_threshold = temporal_coverage_threshold
        
        # Initialise query decomposer
        self.temporal_decomposer = TemporalQueryDecomposer(llm_manager)
        
        # Initialise temporal stopping controller
        self.temporal_stopping_controller = TemporalStoppingController(
            llm_manager=llm_manager,
            temporal_coverage_threshold=temporal_coverage_threshold,
            chain_satisfaction_threshold=0.8,
            constraint_fulfillment_threshold=0.85,
            overload_prevention_threshold=0.9
        )
        
        # Track reasoning state
        self.reasoning_history = []
        self.accumulated_context = ""
        
    def reason_iteratively(self, query: str, verbose: bool = False) -> IterativeResult:
        """
        Execute iterative reasoning process for complex temporal query
        
        Following KG-IRAG methodology:
        1. Decompose query into sub-questions
        2. Iteratively retrieve relevant paths
        3. Evaluate sufficiency and decide next steps
        4. Stop when sufficient evidence is gathered
        """
        
        start_time = datetime.now()
        
        if verbose:
            print(f"Starting iterative reasoning for: {query}")
        
        # Step 1: Decompose query
        sub_questions = self.temporal_decomposer.decompose_query(query)
        
        if verbose:
            print(f"Decomposed into {len(sub_questions)} sub-questions")
        
        # Step 2: Initialise reasoning state
        reasoning_steps = []
        total_paths_retrieved = 0
        accumulated_paths = []
        
        # Step 3: Iterative reasoning loop
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Determine current sub-question to explore
            current_sub_question = self.select_next_sub_question(
                sub_questions, reasoning_steps, iteration
            )
            
            if not current_sub_question:
                convergence_reason = "no_more_subquestions"
                break
            
            # Execute sub-question against TKG
            sub_query_result = self.tkg_engine.query(
                current_sub_question["sub_query"],
                verbose=False
            )
            
            # Create reasoning step
            step = IterativeStep(
                step_id=iteration,
                sub_query=current_sub_question["sub_query"],
                temporal_constraints=current_sub_question.get("temporal_constraints", {}),
                retrieved_paths=sub_query_result.paths,
                reasoning_context=self.build_reasoning_context(accumulated_paths, sub_query_result.paths)
            )
            
            # Update accumulated paths
            accumulated_paths.extend(sub_query_result.paths)
            total_paths_retrieved += len(sub_query_result.paths)
            
            # Evaluate sufficiency using temporal stopping controller
            temporal_query = TemporalQuery(query_text=query)
            stopping_decision = self.temporal_stopping_controller.should_stop(
                temporal_query, accumulated_paths, step.reasoning_context, iteration, reasoning_steps
            )
            
            step.is_sufficient = stopping_decision.should_stop
            step.next_exploration_hints = stopping_decision.next_exploration_hints
            
            # Update temporal coverage from stopping controller
            step.temporal_coverage = {
                "coverage_score": stopping_decision.temporal_coverage.coverage_score,
                "temporal_span_days": stopping_decision.temporal_coverage.temporal_span_days,
                "timestamp_count": stopping_decision.temporal_coverage.timestamp_count,
                "chronological_continuity": stopping_decision.temporal_coverage.chronological_continuity,
                "constraint_satisfaction": stopping_decision.temporal_coverage.constraint_satisfaction_score,
                "stopping_criterion": stopping_decision.stopping_criterion,
                "confidence": stopping_decision.confidence,
                "reasoning": stopping_decision.reasoning
            }
            
            # Temporal coverage already calculated by stopping controller above
            
            reasoning_steps.append(step)
            
            if verbose:
                print(f"Retrieved {len(sub_query_result.paths)} paths")
                print(f"Sufficient: {step.is_sufficient}")
                coverage_score = step.temporal_coverage.get('coverage_score', 0) if hasattr(step, 'temporal_coverage') and step.temporal_coverage else 0
                print(f"Temporal coverage: {coverage_score:.2f}")
            
            # Check convergence using temporal stopping controller decision
            if step.is_sufficient:
                convergence_reason = step.temporal_coverage.get('stopping_criterion', 'sufficient_evidence') if hasattr(step, 'temporal_coverage') and step.temporal_coverage else 'sufficient_evidence'
                break
            
            # Check if we're making progress
            if iteration > 0 and not self.making_progress(reasoning_steps):
                convergence_reason = "no_progress"
                break
        
        else:
            convergence_reason = "max_iterations"
        
        # Step 4: Generate final answer
        final_answer = self.generate_final_answer(query, reasoning_steps)
        
        # Calculate execution time
        total_execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create final result
        result = IterativeResult(
            original_query=query,
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            total_paths_retrieved=total_paths_retrieved,
            total_execution_time=total_execution_time,
            convergence_reason=convergence_reason,
            temporal_coverage=reasoning_steps[-1].temporal_coverage if reasoning_steps else {}
        )
        
        if verbose:
            print(f"Iterative reasoning completed in {total_execution_time:.2f}s")
            print(f"Convergence reason: {convergence_reason}")
            print(f"Total paths retrieved: {total_paths_retrieved}")
        
        return result
    
    def select_next_sub_question(self, 
                                 sub_questions: List[Dict[str, Any]], 
                                 reasoning_steps: List[IterativeStep],
                                 iteration: int) -> Optional[Dict[str, Any]]:
        """Select next sub-question to explore based on dependencies and current state"""
        
        completed_steps = set(range(len(reasoning_steps)))
        
        for i, sub_question in enumerate(sub_questions):
            # Check if already explored
            if i in completed_steps:
                continue
            
            # Check dependencies
            depends_on = sub_question.get("depends_on", [])
            if all(dep in completed_steps for dep in depends_on):
                return sub_question
        
        # If no more sub-questions available, return None
        return None
    
    def build_reasoning_context(self, 
                                accumulated_paths: List[Tuple[Path, TemporalReliabilityMetrics]],
                                new_paths: List[Tuple[Path, TemporalReliabilityMetrics]]) -> str:
        """Build reasoning context from accumulated and new paths"""
        
        context_parts = []
        
        # Add accumulated context
        if accumulated_paths:
            context_parts.append(f"Previous knowledge gathered:")
            for path, metrics in accumulated_paths[-5:]:  # Last 5 paths
                context_parts.append(f" - {path.path_text} (reliability: {metrics.overall_reliability:.2f})")
        
        # Add new findings
        if new_paths:
            context_parts.append(f"New findings:")
            for path, metrics in new_paths[:3]:  # Top 3 new paths
                context_parts.append(f" - {path.path_text} (reliability: {metrics.overall_reliability:.2f})")
        
        return "\n".join(context_parts)
    
    def evaluate_sufficiency(self, 
                             original_query: str, 
                             reasoning_context: str,
                             accumulated_paths: List[Tuple[Path, TemporalReliabilityMetrics]]) -> Tuple[bool, List[str]]:
        """Evaluate if current evidence is sufficient to answer the original query"""
        
        # Create evaluation prompt
        evaluation_prompt = f"""
        Original Query: {original_query}
        
        Current Evidence:
        {reasoning_context}
        
        Based on the evidence gathered, can the original query be answered sufficiently?
        
        Consider:
        1. Are all key entities and relationships covered?
        2. Are temporal constraints satisfied?
        3. Is there enough reliable information?
        
        Respond with JSON:
        {{
            "is_sufficient": true/false,
            "confidence": 0.0-1.0,
            "missing_information": ["list of what's still needed"],
            "exploration_hints": ["suggestions for next exploration"]
        }}
        """
        
        try:
            response = self.llm_manager.generate(evaluation_prompt)
            
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            
            evaluation = json.loads(json_str)
            
            is_sufficient = evaluation.get("is_sufficient", False)
            hints = evaluation.get("exploration_hints", [])
            
            return is_sufficient, hints
            
        except Exception as e:
            print(f"Error in sufficiency evaluation: {e}")
            # Fallback: simple heuristic
            return self.simple_sufficiency_check(accumulated_paths)
    
    def simple_sufficiency_check(self, accumulated_paths: List[Tuple[Path, TemporalReliabilityMetrics]]) -> Tuple[bool, List[str]]:
        """Simple fallback sufficiency check"""
        
        if not accumulated_paths:
            return False, ["Need to gather more information"]
        
        # Check average reliability
        avg_reliability = np.mean([metrics.overall_reliability for _, metrics in accumulated_paths])
        
        if avg_reliability >= self.convergence_threshold and len(accumulated_paths) >= 3:
            return True, []
        else:
            return False, ["Need more reliable information"]
    
    def calculate_temporal_coverage(self, accumulated_paths: List[Tuple[Path, TemporalReliabilityMetrics]]) -> Dict[str, Any]:
        """Calculate temporal coverage of accumulated paths"""
        
        if not accumulated_paths:
            return {"coverage_score": 0.0, "temporal_span": 0, "timestamp_count": 0}
        
        # Extract timestamps
        all_timestamps = []
        for path, _ in accumulated_paths:
            if hasattr(path, 'get_temporal_info'):
                temporal_info = path.get_temporal_info()
                if temporal_info and isinstance(temporal_info, dict):
                    path_timestamps = temporal_info.get("timestamps", [])
                    all_timestamps.extend(path_timestamps)
        
        if not all_timestamps:
            return {"coverage_score": 0.0, "temporal_span": 0, "timestamp_count": 0}
        
        # Calculate temporal span
        try:
            dates = [datetime.fromisoformat(ts.replace('T', ' ')) for ts in all_timestamps]
            temporal_span = (max(dates) - min(dates)).days
            
            # Calculate coverage score based on span and density
            coverage_score = min(1.0, (len(all_timestamps) / max(len(accumulated_paths), 1)) * 0.5 + 
                               min(temporal_span / 365, 1.0) * 0.5)
            
            return {
                "coverage_score": coverage_score,
                "temporal_span": temporal_span,
                "timestamp_count": len(all_timestamps),
                "unique_timestamps": len(set(all_timestamps))
            }
            
        except Exception as e:
            print(f"Error calculating temporal coverage: {e}")
            return {"coverage_score": 0.0, "temporal_span": 0, "timestamp_count": len(all_timestamps)}
    
    def making_progress(self, reasoning_steps: List[IterativeStep]) -> bool:
        """Check if we're making progress in recent iterations"""
        
        if len(reasoning_steps) < 2:
            return True
        
        # Check if we're getting new paths
        recent_paths = reasoning_steps[-1].retrieved_paths
        previous_paths = reasoning_steps[-2].retrieved_paths
        
        # Simple progress check - are we getting new information?
        return len(recent_paths) > 0 and len(recent_paths) != len(previous_paths)
    
    def generate_final_answer(self, original_query: str, reasoning_steps: List[IterativeStep]) -> str:
        """Generate final answer based on all reasoning steps"""
        
        # Compile all evidence
        all_evidence = []
        for step in reasoning_steps:
            if step.reasoning_context:
                all_evidence.append(f"Step {step.step_id + 1}: {step.reasoning_context}")
        
        # Create final answer prompt
        answer_prompt = f"""
        Original Query: {original_query}
        
        Evidence gathered through iterative reasoning:
        {chr(10).join(all_evidence)}
        
        Based on all the evidence gathered, provide a comprehensive answer to the original query.
        Focus on:
        1. Direct answer to the question
        2. Supporting temporal evidence
        3. Confidence level in the answer
        
        Answer:
        """
        
        try:
            final_answer = self.llm_manager.generate(answer_prompt)
            return final_answer.strip()
            
        except Exception as e:
            print(f"Error generating final answer: {e}")
            return "Unable to generate final answer due to processing error."
    
    def get_reasoning_summary(self, result: IterativeResult) -> Dict[str, Any]:
        """Get summary of reasoning process for analysis"""
        
        return {
            "query": result.original_query,
            "total_steps": len(result.reasoning_steps),
            "total_paths": result.total_paths_retrieved,
            "execution_time": result.total_execution_time,
            "convergence_reason": result.convergence_reason,
            "temporal_coverage": result.temporal_coverage,
            "step_details": [
                {
                    "step_id": step.step_id,
                    "sub_query": step.sub_query,
                    "paths_retrieved": len(step.retrieved_paths),
                    "is_sufficient": step.is_sufficient,
                    "temporal_coverage": step.temporal_coverage
                }
                for step in result.reasoning_steps
            ]
        }
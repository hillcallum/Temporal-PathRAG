"""
Reasoning Trace System for Temporal PathRAG
Captures detailed logs of the reasoning process for explainability
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import networkx as nx

logger = logging.getLogger(__name__)


class TraceLevel(Enum):
    """Levels of detail for reasoning traces"""
    MINIMAL = "minimal"     # Just key steps
    STANDARD = "standard"   # Standard detail
    DETAILED = "detailed"   # Full detail including scores
    DEBUG = "debug"        # Everything including internal states


@dataclass
class QueryUnderstanding:
    """Records query understanding phase"""
    original_query: str
    extracted_entities: List[str] = field(default_factory=list)
    resolved_entities: List[Dict[str, Any]] = field(default_factory=list)
    temporal_expressions: List[str] = field(default_factory=list)
    temporal_constraints: Dict[str, Any] = field(default_factory=dict)
    query_type: str = "unknown"  # point, range, union, intersection, decay
    confidence: float = 0.0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphSubsetSelection:
    """Records graph subset selection based on temporal constraints"""
    total_nodes: int = 0
    total_edges: int = 0
    temporal_filter_applied: bool = False
    time_range: Optional[Tuple[int, int]] = None
    filtered_nodes: int = 0
    filtered_edges: int = 0
    filter_reduction_percentage: float = 0.0
    sample_filtered_entities: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PathScore:
    """Individual path scoring details"""
    path: List[str]
    path_string: str
    semantic_score: float = 0.0
    temporal_score: float = 0.0
    combined_score: float = 0.0
    path_length: int = 0
    temporal_violations: List[str] = field(default_factory=list)
    edge_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PathRanking:
    """Records path discovery and ranking"""
    source_entities: List[str] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    total_paths_found: int = 0
    paths_after_filtering: int = 0
    top_k: int = 10
    path_scores: List[PathScore] = field(default_factory=list)
    scoring_method: str = "hybrid"  # traditional, embedding, hybrid
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert path scores
        data['path_scores'] = [ps.to_dict() for ps in self.path_scores]
        return data


@dataclass
class AnswerGeneration:
    """Records answer generation phase"""
    retrieved_paths: int = 0
    context_length: int = 0
    llm_model: str = ""
    prompt_template: str = ""
    raw_answer: str = ""
    extracted_answer: str = ""
    answer_confidence: float = 0.0
    answer_type: str = ""  # entity, date, number, yes/no, description
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query"""
    trace_id: str
    timestamp: str
    query: str
    dataset: str = ""
    trace_level: TraceLevel = TraceLevel.STANDARD
    
    # Phases
    query_understanding: Optional[QueryUnderstanding] = None
    graph_subset: Optional[GraphSubsetSelection] = None
    path_ranking: Optional[PathRanking] = None
    answer_generation: Optional[AnswerGeneration] = None
    
    # Overall metrics
    total_processing_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation"""
        data = {
            'trace_id': self.trace_id,
            'timestamp': self.timestamp,
            'query': self.query,
            'dataset': self.dataset,
            'trace_level': self.trace_level.value,
            'total_processing_time': self.total_processing_time,
            'success': self.success,
            'error_message': self.error_message
        }
        
        # Add phases if present
        if self.query_understanding:
            data['query_understanding'] = self.query_understanding.to_dict()
        if self.graph_subset:
            data['graph_subset'] = self.graph_subset.to_dict()
        if self.path_ranking:
            data['path_ranking'] = self.path_ranking.to_dict()
        if self.answer_generation:
            data['answer_generation'] = self.answer_generation.to_dict()
            
        return data
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """Save trace to file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    def get_summary(self) -> str:
        """Get human-readable summary of the trace"""
        summary = f"Reasoning Trace Summary\n"
        summary += f"Query: {self.query}\n"
        summary += f"Success: {self.success}\n"
        summary += f"Total Time: {self.total_processing_time:.2f}s\n\n"
        
        if self.query_understanding:
            qu = self.query_understanding
            summary += "1. Query Understanding:\n"
            summary += f" - Entities: {qu.extracted_entities} → {[e['resolved'] for e in qu.resolved_entities]}\n"
            summary += f" - Temporal: {qu.temporal_expressions}\n"
            summary += f" - Query Type: {qu.query_type}\n"
            summary += f" - Confidence: {qu.confidence:.2f}\n\n"
        
        if self.graph_subset:
            gs = self.graph_subset
            summary += "2. Graph Subset Selection:\n"
            summary += f" - Original: {gs.total_nodes} nodes, {gs.total_edges} edges\n"
            if gs.temporal_filter_applied:
                summary += f" - Time Range: {gs.time_range}\n"
                summary += f" - Filtered: {gs.filtered_nodes} nodes, {gs.filtered_edges} edges\n"
                summary += f" - Reduction: {gs.filter_reduction_percentage:.1f}%\n\n"
        
        if self.path_ranking:
            pr = self.path_ranking
            summary += "3. Path Discovery & Ranking:\n"
            summary += f" - Source → Target: {pr.source_entities} → {pr.target_entities}\n"
            summary += f" - Paths: {pr.total_paths_found} found, {pr.paths_after_filtering} after filtering\n"
            summary += f" - Scoring: {pr.scoring_method}\n"
            if pr.path_scores:
                summary += f" - Top Path: {pr.path_scores[0].path_string} (score: {pr.path_scores[0].combined_score:.3f})\n\n"
        
        if self.answer_generation:
            ag = self.answer_generation
            summary += "4. Answer Generation:\n"
            summary += f" - Context: {ag.retrieved_paths} paths, {ag.context_length} chars\n"
            summary += f" - Model: {ag.llm_model}\n"
            summary += f" - Answer: {ag.extracted_answer}\n"
            summary += f" - Type: {ag.answer_type}\n"
            summary += f" - Confidence: {ag.answer_confidence:.2f}\n"
        
        if self.error_message:
            summary += f"\nError: {self.error_message}\n"
            
        return summary


class ReasoningTracer:
    """Manager for creating and handling reasoning traces"""
    
    def __init__(self, trace_level: TraceLevel = TraceLevel.STANDARD):
        self.trace_level = trace_level
        self.current_trace: Optional[ReasoningTrace] = None
        self.traces: List[ReasoningTrace] = []
        
    def start_trace(self, query: str, dataset: str = "") -> ReasoningTrace:
        """Start a new reasoning trace"""
        trace_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.current_trace = ReasoningTrace(
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            query=query,
            dataset=dataset,
            trace_level=self.trace_level
        )
        return self.current_trace
    
    def end_trace(self, success: bool = True, error_message: Optional[str] = None):
        """End the current trace"""
        if self.current_trace:
            self.current_trace.success = success
            self.current_trace.error_message = error_message
            self.traces.append(self.current_trace)
            
            # Log summary
            logger.info(f"Trace completed: {self.current_trace.trace_id}")
            if self.trace_level in [TraceLevel.DETAILED, TraceLevel.DEBUG]:
                logger.info(self.current_trace.get_summary())
                
            current = self.current_trace
            self.current_trace = None
            return current
    
    def get_current_trace(self) -> Optional[ReasoningTrace]:
        """Get the current active trace"""
        return self.current_trace
    
    def save_traces(self, filepath: str):
        """Save all traces to a file"""
        data = {
            'trace_level': self.trace_level.value,
            'num_traces': len(self.traces),
            'traces': [trace.to_dict() for trace in self.traces]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics across all traces"""
        if not self.traces:
            return {}
            
        stats = {
            'total_queries': len(self.traces),
            'successful_queries': sum(1 for t in self.traces if t.success),
            'average_processing_time': sum(t.total_processing_time for t in self.traces) / len(self.traces),
            'query_types': {},
            'answer_types': {},
            'average_path_reduction': []
        }
        
        for trace in self.traces:
            # Query types
            if trace.query_understanding:
                qt = trace.query_understanding.query_type
                stats['query_types'][qt] = stats['query_types'].get(qt, 0) + 1
            
            # Answer types
            if trace.answer_generation:
                at = trace.answer_generation.answer_type
                stats['answer_types'][at] = stats['answer_types'].get(at, 0) + 1
            
            # Path reduction
            if trace.graph_subset and trace.graph_subset.temporal_filter_applied:
                stats['average_path_reduction'].append(trace.graph_subset.filter_reduction_percentage)
        
        if stats['average_path_reduction']:
            stats['average_path_reduction'] = sum(stats['average_path_reduction']) / len(stats['average_path_reduction'])
        else:
            stats['average_path_reduction'] = 0
            
        return stats


# Global tracer instance
global_tracer = None


def get_tracer(trace_level: Optional[TraceLevel] = None) -> ReasoningTracer:
    """Get or create global tracer instance"""
    global global_tracer
    if global_tracer is None or (trace_level and global_tracer.trace_level != trace_level):
        global_tracer = ReasoningTracer(trace_level or TraceLevel.STANDARD)
    return global_tracer


def trace_query_understanding(
    original_query: str,
    extracted_entities: List[str],
    resolved_entities: List[Dict[str, Any]],
    temporal_expressions: List[str],
    temporal_constraints: Dict[str, Any],
    query_type: str,
    confidence: float,
    processing_time: float
):
    """Helper to trace query understanding phase"""
    tracer = get_tracer()
    if tracer.current_trace:
        tracer.current_trace.query_understanding = QueryUnderstanding(
            original_query=original_query,
            extracted_entities=extracted_entities,
            resolved_entities=resolved_entities,
            temporal_expressions=temporal_expressions,
            temporal_constraints=temporal_constraints,
            query_type=query_type,
            confidence=confidence,
            processing_time=processing_time
        )


def trace_graph_subset(
    graph: nx.MultiDiGraph,
    temporal_filter_applied: bool,
    time_range: Optional[Tuple[int, int]],
    filtered_graph: Optional[nx.MultiDiGraph],
    processing_time: float
):
    """Helper to trace graph subset selection"""
    tracer = get_tracer()
    if tracer.current_trace:
        gs = GraphSubsetSelection(
            total_nodes=len(graph.nodes()),
            total_edges=len(graph.edges()),
            temporal_filter_applied=temporal_filter_applied,
            time_range=time_range,
            processing_time=processing_time
        )
        
        if filtered_graph and temporal_filter_applied:
            gs.filtered_nodes = len(filtered_graph.nodes())
            gs.filtered_edges = len(filtered_graph.edges())
            gs.filter_reduction_percentage = (1 - gs.filtered_edges / gs.total_edges) * 100 if gs.total_edges > 0 else 0
            
            # Sample some filtered entities
            gs.sample_filtered_entities = list(filtered_graph.nodes())[:10]
            
        tracer.current_trace.graph_subset = gs


def trace_path_ranking(
    source_entities: List[str],
    target_entities: List[str],
    paths: List[Any],
    path_scores: List[Tuple[Any, float]],
    scoring_method: str,
    processing_time: float
):
    """Helper to trace path ranking"""
    tracer = get_tracer()
    if tracer.current_trace and tracer.trace_level != TraceLevel.MINIMAL:
        pr = PathRanking(
            source_entities=source_entities,
            target_entities=target_entities,
            total_paths_found=len(paths),
            paths_after_filtering=len(path_scores),
            scoring_method=scoring_method,
            processing_time=processing_time
        )
        
        # Add detailed path scores if detailed level
        if tracer.trace_level in [TraceLevel.DETAILED, TraceLevel.DEBUG]:
            for path, score in path_scores[:10]:  # Top 10 paths
                ps = PathScore(
                    path=path if isinstance(path, list) else list(path),
                    path_string=" -> ".join(str(n) for n in path),
                    combined_score=score,
                    path_length=len(path)
                )
                pr.path_scores.append(ps)
                
        tracer.current_trace.path_ranking = pr


def trace_answer_generation(
    retrieved_paths: int,
    context_length: int,
    llm_model: str,
    raw_answer: str,
    extracted_answer: str,
    answer_type: str,
    answer_confidence: float,
    processing_time: float
):
    """Helper to trace answer generation"""
    tracer = get_tracer()
    if tracer.current_trace:
        tracer.current_trace.answer_generation = AnswerGeneration(
            retrieved_paths=retrieved_paths,
            context_length=context_length,
            llm_model=llm_model,
            raw_answer=raw_answer[:200] if tracer.trace_level != TraceLevel.DEBUG else raw_answer,
            extracted_answer=extracted_answer,
            answer_type=answer_type,
            answer_confidence=answer_confidence,
            processing_time=processing_time
        )
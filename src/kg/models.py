"""
PathRAG data models for representing nodes, edges, and paths in knowledge graphs
Following 'PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths' 
by B. Chen et al. (2025)

Enhanced with temporal query processing and result models for TKG operations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

@dataclass
class TemporalPathRAGNode:
    """
    Node representation with identifier and textual chunk (tv)
    Follows PathRAG paper's node representation for semantic content
    """
    id: str
    entity_type: str  # e.g., "Person", "Object", "Event"
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # PathRAG specific properties
    tv: str = ""  # textual chunk - semantic description of the node
    embedding: Optional[List[float]] = None
    description: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.name:
            self.name = self.id
        # Create textual chunk if not provided
        if not self.tv:
            self.tv = f"{self.name}: {self.description}" if self.description else self.name

@dataclass 
class TemporalPathRAGEdge:
    """
    Edge representation with relationship type and descriptive textual chunk (te)
    Follows PathRAG paper's edge representation for semantic relationships
    Enhanced with temporal information for temporal weighting
    """
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    description: str = ""
    
    # PathRAG specific properties
    te: str = ""  # textual chunk - semantic description of the relationship
    flow_capacity: float = 1.0  # for future flow-based pruning
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal properties for enhanced scoring
    timestamp: Optional[str] = None  # ISO format timestamp
    temporal_weight: float = 1.0  # temporal relevance weight
    temporal_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_temporal_info(self) -> Dict[str, Any]:
        """Extract temporal information for scoring"""
        return {
            'timestamp': self.timestamp,
            'temporal_weight': self.temporal_weight,
            'metadata': self.temporal_metadata
        }
    
    def __post_init__(self):
        # Create textual chunk if not provided
        if not self.te:
            self.te = self.description if self.description else f"{self.source_id} {self.relation_type} {self.target_id}"
    
@dataclass
class Path:
    """
    Represents a path through the knowledge graph with PathRAG textual chunks
    Combines node textual chunks (tv) and edge textual chunks (te) for semantic reasoning
    """
    nodes: List[TemporalPathRAGNode] = field(default_factory=list)
    edges: List[TemporalPathRAGEdge] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    path_text: str = ""  # Combined textual representation
    
    def __post_init__(self):
        # Generate path text if not provided
        if not self.path_text and self.nodes:
            self.path_text = self.generate_path_text()
    
    def add_node(self, node: TemporalPathRAGNode):
        """Add a node to the path"""
        self.nodes.append(node)
        self.path_text = self.generate_path_text()  # Regenerate text
    
    def add_edge(self, edge: TemporalPathRAGEdge):
        """Add an edge to the path"""
        self.edges.append(edge)
        self.path_text = self.generate_path_text()  # Regenerate text
    
    def get_length(self) -> int:
        """Get the number of hops in the path"""
        return len(self.edges)
    
    def get_node_ids(self) -> List[str]:
        """Get list of node IDs in the path"""
        return [node.id for node in self.nodes]
    
    def get_edge_relations(self) -> List[str]:
        """Get list of edge relation types in the path"""
        return [edge.relation_type for edge in self.edges]
    
    def generate_path_text(self) -> str:
        """Generate textual representation using tv and te chunks"""
        if not self.nodes:
            return ""
        
        text_parts = [self.nodes[0].tv]
        
        for i, edge in enumerate(self.edges):
            if i + 1 < len(self.nodes):
                text_parts.append(f" --[{edge.relation_type}: {edge.te}]--> ")
                text_parts.append(self.nodes[i + 1].tv)
        
        return "".join(text_parts)
    
    def get_temporal_info(self) -> Dict[str, Any]:
        """Extract temporal information from path for temporal scoring"""
        timestamps = []
        temporal_edges = []
        
        for edge in self.edges:
            if hasattr(edge, 'timestamp') and edge.timestamp:
                timestamps.append(edge.timestamp)
                temporal_edges.append({
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'relation': edge.relation_type,
                    'timestamp': edge.timestamp
                })
        
        return {
            'timestamps': timestamps,
            'temporal_edges': temporal_edges,
            'temporal_density': len(timestamps) / max(len(self.edges), 1)
        }

    def __str__(self) -> str:
        if not self.nodes:
            return "Empty path"
        
        path_str = self.nodes[0].name
        for i, edge in enumerate(self.edges):
            if i + 1 < len(self.nodes):
                path_str += f" --[{edge.relation_type}]--> {self.nodes[i + 1].name}"
        
        return f"Path: {path_str} (score: {self.score:.3f})"

# TEMPORAL QUERY PROCESSING MODELS

@dataclass
class TemporalQuery:
    """Represents a temporal query with constraints and context"""
    
    query_text: str
    source_entities: List[str] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    temporal_constraints: Dict[str, Any] = field(default_factory=dict)
    query_time: str = ""
    max_hops: int = 3
    top_k: int = 10
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.query_time:
            self.query_time = datetime.now().isoformat()
        
        # Extract temporal patterns from query text if not provided
        if not self.temporal_patterns:
            self.temporal_patterns = self.extract_temporal_patterns()
    
    def extract_temporal_patterns(self) -> Dict[str, Any]:
        """Extract temporal patterns from query text"""
        patterns = {
            'before_keywords': ['before', 'prior to', 'earlier than', 'preceding'],
            'after_keywords': ['after', 'following', 'later than', 'subsequent to'],
            'during_keywords': ['during', 'in', 'within', 'throughout'],
            'temporal_direction': None,
            'time_references': []
        }
        
        query_lower = self.query_text.lower()
        
        # Determine temporal direction
        if any(keyword in query_lower for keyword in patterns['before_keywords']):
            patterns['temporal_direction'] = 'before'
        elif any(keyword in query_lower for keyword in patterns['after_keywords']):
            patterns['temporal_direction'] = 'after'
        elif any(keyword in query_lower for keyword in patterns['during_keywords']):
            patterns['temporal_direction'] = 'during'
            
        return patterns


@dataclass
class TemporalReliabilityMetrics:
    """Container for temporal reliability metrics"""
    
    temporal_consistency: float = 0.0
    chronological_coherence: float = 0.0
    source_credibility: float = 0.0
    cross_validation_score: float = 0.0
    temporal_pattern_strength: float = 0.0
    flow_reliability: float = 0.0
    semantic_coherence: float = 0.0
    overall_reliability: float = 0.0
    
    # Detailed breakdowns
    temporal_violations: List[str] = field(default_factory=list)
    credibility_factors: Dict[str, Any] = field(default_factory=dict)
    pattern_matches: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Container for query result with metadata"""
    
    query_text: str
    paths: List[Tuple[Path, TemporalReliabilityMetrics]]
    execution_time: float
    total_paths_discovered: int
    total_paths_after_pruning: int
    query_metadata: Dict[str, Any]
    reliability_threshold: float


@dataclass  
class PathExplanation:
    """Explanation for why a path was selected"""
    
    path_summary: str
    reliability_breakdown: Dict[str, float]
    temporal_highlights: List[str]
    confidence_level: str
    supporting_evidence: List[str]

# TEMPORAL FLOW PRUNING MODELS


@dataclass
class FlowPruningConfig:
    """Configuration for temporal flow pruning"""
    
    alpha: float = 0.01  # Temporal decay rate (optimized)
    base_theta: float = 0.1  # Base pruning threshold (optimized)
    diversity_threshold: float = 0.7  # Diversity constraint
    reliability_threshold: float = 0.6  # Reliability threshold
    enable_cross_validation: bool = True
    enable_gpu_acceleration: bool = True
    temporal_window: int = 365  # Days
    
    def __post_init__(self):
        # Validate parameters
        if not 0.001 <= self.alpha <= 1.0:
            raise ValueError(f"Alpha must be between 0.001 and 1.0, got {self.alpha}")
        if not 0.1 <= self.base_theta <= 10.0:
            raise ValueError(f"Base theta must be between 0.1 and 10.0, got {self.base_theta}")
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError(f"Diversity threshold must be between 0.0 and 1.0, got {self.diversity_threshold}")
        if not 0.0 <= self.reliability_threshold <= 1.0:
            raise ValueError(f"Reliability threshold must be between 0.0 and 1.0, got {self.reliability_threshold}")


@dataclass
class PerformanceMetrics:
    """Performance metrics for TKG operations"""
    
    total_queries: int = 0
    avg_retrieval_time: float = 0.0
    avg_paths_discovered: int = 0
    avg_paths_after_pruning: int = 0
    avg_reliability_score: float = 0.0
    success_rate: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilisation_percent: float = 0.0


# GRAPH STATISTICS MODELS

@dataclass
class GraphStatistics:
    """Statistics about the TKG for credibility assessment"""
    
    total_nodes: int = 0
    total_edges: int = 0
    entity_frequencies: Dict[str, int] = field(default_factory=dict)
    relation_frequencies: Dict[str, int] = field(default_factory=dict)
    max_entity_frequency: int = 0
    max_relation_frequency: int = 0
    temporal_coverage: Dict[str, Any] = field(default_factory=dict)
    
    def get_entity_credibility_score(self, entity_id: str) -> float:
        """Get credibility score for an entity based on frequency"""
        frequency = self.entity_frequencies.get(entity_id, 0)
        if frequency == 0 or self.max_entity_frequency == 0:
            return 0.5  # Neutral credibility
        
        # Log-scale normalisation to prevent dominance
        import math
        frequency_score = min(math.log(frequency + 1) / math.log(self.max_entity_frequency + 1), 1.0)
        return 0.4 + 0.6 * frequency_score
    
    def get_relation_credibility_score(self, relation_type: str) -> float:
        """Get credibility score for a relation type"""
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
            return 0.8
        elif relation_type in medium_credibility_relations:
            return 0.6
        else:
            # Use frequency-based scoring
            frequency = self.relation_frequencies.get(relation_type, 0)
            if frequency == 0 or self.max_relation_frequency == 0:
                return 0.5
            
            frequency_score = min(frequency / self.max_relation_frequency, 1.0)
            return 0.3 + 0.4 * frequency_score
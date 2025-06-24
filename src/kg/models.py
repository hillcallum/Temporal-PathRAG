"""
PathRAG data models for representing nodes, edges, and paths in knowledge graphs
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class PathRAGNode:
    """Basic node representation for PathRAG algorithm"""
    id: str
    entity_type: str  # e.g., "Person", "Object", "Event"
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # PathRAG specific properties
    embedding: Optional[List[float]] = None
    description: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.name:
            self.name = self.id

@dataclass 
class PathRAGEdge:
    """Basic edge representation for PathRAG algorithm"""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    description: str = ""
    
    # PathRAG specific properties for flow-based pruning
    flow_capacity: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class Path:
    """Represents a path through the knowledge graph"""
    nodes: List[PathRAGNode] = field(default_factory=list)
    edges: List[PathRAGEdge] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
    def add_node(self, node: PathRAGNode):
        """Add a node to the path"""
        self.nodes.append(node)
    
    def add_edge(self, edge: PathRAGEdge):
        """Add an edge to the path"""
        self.edges.append(edge)
    
    def get_length(self) -> int:
        """Get the number of hops in the path"""
        return len(self.edges)
    
    def get_node_ids(self) -> List[str]:
        """Get list of node IDs in the path"""
        return [node.id for node in self.nodes]

    def __str__(self) -> List[str]:
        if not self.nodes:
            return "Empty path"
        
        path_str = self.nodes[0].name
        for i, edge in enumerate(self.edges):
            if i + 1 < len(self.nodes):
                path_str += f" --[{edge.relation_type}] --> {self.nodes[i + 1].name}"
        
        return f"Path: {path_str} (score: {self.score:.3f})"

# Additional entity types, purely for toy implementation, will remove in the future
@dataclass
class Person(PathRAGNode):
    """Represents a person entity"""
    birth_date: Optional[str] = None
    nationality: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.entity_type = "Person"

@dataclass
class Event(PathRAGNode):
    """Represents an event entity"""
    date: Optional[str] = None
    location: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.entity_type = "Event"
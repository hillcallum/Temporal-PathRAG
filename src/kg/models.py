from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
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
    
@dataclass
class Path:
    """Represents a path through the knowledge graph"""
    nodes: List[PathRAGNode] = field(default_factory=list)
    edges: List[PathRAGEdge] = field(default_factory=list)
    score: float = 0.0
    
    def __len__(self):
        return len(self.nodes)
    
    def add_node(self, node: PathRAGNode):
        """Add a node to the path"""
        self.nodes.append(node)
    
    def add_edge(self, edge: PathRAGEdge):
        """Add an edge to the path"""
        self.edges.append(edge)
    
    def get_node_ids(self) -> List[str]:
        """Get list of node IDs in the path"""
        return [node.id for node in self.nodes]

# Simple data models for toy graph
@dataclass
class Person:
    """Simple person entity for toy graph"""
    id: str
    name: str
    year_born: int 

@dataclass
class Event:
    """Simple event entity for toy graph"""
    id: str
    name: str
    year: int
    participants: List[str]  # list of person IDs
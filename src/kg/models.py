"""
PathRAG data models for representing nodes, edges, and paths in knowledge graphs.
Following 'PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths' 
by B. Chen et al. (2025)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class PathRAGNode:
    """
    Node representation with identifier and textual chunk (tv).
    Follows PathRAG paper's node representation for semantic content.
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
class PathRAGEdge:
    """
    Edge representation with relationship type and descriptive textual chunk (te).
    Follows PathRAG paper's edge representation for semantic relationships.
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
    
    def __post_init__(self):
        # Create textual chunk if not provided
        if not self.te:
            self.te = self.description if self.description else f"{self.source_id} {self.relation_type} {self.target_id}"
    
@dataclass
class Path:
    """
    Represents a path through the knowledge graph with PathRAG textual chunks.
    Combines node textual chunks (tv) and edge textual chunks (te) for semantic reasoning.
    """
    nodes: List[PathRAGNode] = field(default_factory=list)
    edges: List[PathRAGEdge] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    path_text: str = ""  # Combined textual representation
    
    def __post_init__(self):
        # Generate path text if not provided
        if not self.path_text and self.nodes:
            self.path_text = self.generate_path_text()
    
    def add_node(self, node: PathRAGNode):
        """Add a node to the path"""
        self.nodes.append(node)
        self.path_text = self.generate_path_text()  # Regenerate text
    
    def add_edge(self, edge: PathRAGEdge):
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

    def __str__(self) -> str:
        if not self.nodes:
            return "Empty path"
        
        path_str = self.nodes[0].name
        for i, edge in enumerate(self.edges):
            if i + 1 < len(self.nodes):
                path_str += f" --[{edge.relation_type}]--> {self.nodes[i + 1].name}"
        
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
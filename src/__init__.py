"""
PathRAG implementation for multi-hop question answering
"""

from .kg.models import PathRAGNode, PathRAGEdge, Path, Person, Event
from .kg.path_traversal import BasicPathTraversal

__version__ = "0.1.0"

__all__ = [
    'PathRAGNode',
    'PathRAGEdge',
    'Path', 
    'Person',
    'Event',
    'BasicPathTraversal'
]
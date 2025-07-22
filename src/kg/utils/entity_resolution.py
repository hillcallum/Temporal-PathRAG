"""
Entity resolution and graph enhancement utilities for Temporal PathRAG

Addresses key issues:
1. Entity recognition and mapping between natural language and graph entities
2. Adding textual representations (tv) to nodes and edges for PathRAG
3. Fuzzy matching for better entity resolution
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx
import logging

# Try to import fuzzy matching library
try:
    from rapidfuzz import fuzz, process
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
    except ImportError:
        # Fallback to basic string matching
        fuzz = None
        process = None
        logging.warning("No fuzzy matching library found - need to install rapidfuzz or fuzzywuzzy")

logger = logging.getLogger(__name__)


class EntityResolver:
    """
    Resolves natural language entity mentions to graph entity IDs
    Handles various entity formats and naming conventions
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.entity_index = self.build_entity_index()
        self.alias_map = self.build_alias_map()
        
    def build_entity_index(self) -> Dict[str, Set[str]]:
        """Build an index of entity variations for fast look-up"""
        entity_index = {}
        
        for node_id in self.graph.nodes:
            # Extract variations from node ID
            variations = self.extract_variations(node_id)
            
            for var in variations:
                var_lower = var.lower()
                if var_lower not in entity_index:
                    entity_index[var_lower] = set()
                entity_index[var_lower].add(node_id)
        
        return entity_index
    
    def extract_variations(self, entity_id: str) -> List[str]:
        """Extract different variations of an entity name"""
        variations = [entity_id]
        
        # Handle format: "Entity_Name(Context)"
        if '(' in entity_id and ')' in entity_id:
            # Extract main name
            main_name = entity_id.split('(')[0].strip('_')
            variations.append(main_name)
            
            # Extract context
            context_match = re.search(r'\(([^)]+)\)', entity_id)
            if context_match:
                context = context_match.group(1)
                variations.append(context)
        
        # Handle underscores and slashes
        if '_' in entity_id:
            # Replace underscores with spaces
            spaced_version = entity_id.replace('_', ' ')
            variations.append(spaced_version)
            
            # Split by underscore
            parts = entity_id.split('_')
            variations.extend(parts)
        
        if '/' in entity_id:
            # Split by slash
            parts = entity_id.split('/')
            variations.extend(parts)
            
            # Replace slashes with spaces
            spaced_version = entity_id.replace('/', ' ')
            variations.append(spaced_version)
        
        # Clean variations
        cleaned_variations = []
        for var in variations:
            cleaned = var.strip().strip('_/')
            if cleaned and len(cleaned) > 1:
                cleaned_variations.append(cleaned)
        
        return list(set(cleaned_variations))
    
    def build_alias_map(self) -> Dict[str, str]:
        """Build common aliases and abbreviations"""
        alias_map = {
            # Country aliases
            'usa': 'United_States',
            'us': 'United_States',
            'uk': 'United_Kingdom',
            'ussr': 'Soviet_Union',
            
            # Organisation aliases
            'un': 'United_Nations',
            'eu': 'European_Union',
            'nato': 'NATO',
            
            # Common variations
            'ministry': ['Ministry', 'Department', 'Office'],
            'council': ['Council', 'Committee', 'Board'],
            'cabinet': ['Cabinet', 'Government', 'Administration'],
        }
        
        return alias_map
    
    def resolve(self, entity_mention: str, threshold: float = 0.7) -> Optional[str]:
        """
        Resolve an entity mention to a graph entity ID
        """
        if not entity_mention:
            return None
        
        # Try exact match first
        if entity_mention in self.graph.nodes:
            return entity_mention
        
        # Try case-insensitive exact match
        mention_lower = entity_mention.lower()
        if mention_lower in self.entity_index:
            candidates = list(self.entity_index[mention_lower])
            if len(candidates) == 1:
                return candidates[0]
        
        # Try alias resolution
        resolved_alias = self.resolve_alias(entity_mention)
        if resolved_alias and resolved_alias in self.graph.nodes:
            return resolved_alias
        
        # Fuzzy matching
        best_match = self.fuzzy_match(entity_mention, threshold)
        if best_match:
            return best_match
        
        # Partial matching
        partial_match = self.partial_match(entity_mention)
        if partial_match:
            return partial_match
        
        return None
    
    def resolve_multiple(self, entity_mentions: List[str], threshold: float = 0.7) -> List[Tuple[str, Optional[str]]]:
        """Resolve multiple entity mentions"""
        results = []
        for mention in entity_mentions:
            resolved = self.resolve(mention, threshold)
            results.append((mention, resolved))
        return results
    
    def resolve_alias(self, mention: str) -> Optional[str]:
        """Resolve common aliases"""
        mention_lower = mention.lower()
        
        # Direct alias lookup
        if mention_lower in self.alias_map:
            alias = self.alias_map[mention_lower]
            if isinstance(alias, str) and alias in self.graph.nodes:
                return alias
        
        return None
    
    def fuzzy_match(self, mention: str, threshold: float) -> Optional[str]:
        """Use fuzzy string matching to find best match"""
        best_score = 0
        best_match = None
        
        # Get all node IDs
        all_nodes = list(self.graph.nodes)
        
        if fuzz and process:
            # Use fuzzy matching library for efficient matching
            matches = process.extract(mention, all_nodes, scorer=fuzz.token_sort_ratio, limit=5)
            
            for match_data in matches:
                # Handle different return formats (rapidfuzz vs fuzzywuzzy)
                if isinstance(match_data, tuple) and len(match_data) >= 2:
                    match = match_data[0]
                    score = match_data[1]
                else:
                    continue
                    
                normalised_score = score / 100.0
                if normalised_score >= threshold and normalised_score > best_score:
                    best_score = normalised_score
                    best_match = match
        
        # Also check against variations (with or without fuzzy lib)
        for node_id in all_nodes[:1000]:  # Limit for performance
            variations = self.extract_variations(node_id)
            for var in variations:
                if fuzz:
                    score = fuzz.token_sort_ratio(mention.lower(), var.lower()) / 100.0
                else:
                    # Basic similarity without fuzzy library
                    score = self.basic_similarity(mention.lower(), var.lower())
                
                if score >= threshold and score > best_score:
                    best_score = score
                    best_match = node_id
        
        return best_match
    
    def basic_similarity(self, str1: str, str2: str) -> float:
        """Basic string similarity without external libraries"""
        if str1 == str2:
            return 1.0
        
        # Convert to sets of words
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def partial_match(self, mention: str) -> Optional[str]:
        """Find entities that contain the mention as a substring"""
        mention_lower = mention.lower()
        candidates = []
        
        for node_id in self.graph.nodes:
            node_lower = node_id.lower()
            if mention_lower in node_lower or node_lower in mention_lower:
                candidates.append((node_id, self.calculate_overlap_score(mention_lower, node_lower)))
        
        if candidates:
            # Sort by overlap score
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def calculate_overlap_score(self, str1: str, str2: str) -> float:
        """Calculate overlap score between two strings"""
        if str1 == str2:
            return 1.0
        
        # Calculate based on common substring length
        longer = max(len(str1), len(str2))
        shorter = min(len(str1), len(str2))
        
        return shorter / longer
    
    def get_entity_info(self, entity_id: str) -> Dict[str, Any]:
        """Get information about an entity"""
        if entity_id not in self.graph.nodes:
            return {"found": False}
        
        node_data = self.graph.nodes[entity_id]
        neighbours = list(self.graph.neighbors(entity_id))
        in_neighbours = list(self.graph.predecessors(entity_id))
        
        return {
            "found": True,
            "entity_id": entity_id,
            "node_data": dict(node_data),
            "variations": self.extract_variations(entity_id),
            "out_degree": len(neighbours),
            "in_degree": len(in_neighbours),
            "total_connections": len(neighbours) + len(in_neighbours)
        }


def enhance_graph_with_textual_representations(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Add textual representations (tv) to nodes and edges
    """
    logger.info("Enhancing graph with textual representations")
    
    # Enhance nodes
    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]
        
        # Skip if already has tv
        if 'tv' in node_data:
            continue
        
        # Generate textual representation
        tv = generate_node_textual_value(node_id, node_data)
        graph.nodes[node_id]['tv'] = tv
    
    # Enhance edges
    for u, v, key, data in graph.edges(keys=True, data=True):
        # Skip if already has te
        if 'te' in data:
            continue
        
        # Generate textual representation
        te = generate_edge_textual_value(u, v, data)
        graph.edges[u, v, key]['te'] = te
    
    logger.info(f"Enhanced {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph


def generate_node_textual_value(node_id: str, node_data: Dict[str, Any]) -> str:
    """
    Generate textual representation for a node
    """
    # Start with cleaned node ID
    text_parts = []
    
    # Clean node ID
    cleaned_id = node_id.replace('_', ' ').replace('/', ' or ')
    
    # Handle special formats
    if '(' in cleaned_id and ')' in cleaned_id:
        # Format: "Entity Name(Context)"
        main_part = cleaned_id.split('(')[0].strip()
        context_match = re.search(r'\(([^)]+)\)', cleaned_id)
        if context_match:
            context = context_match.group(1)
            text_parts.append(f"{main_part} of {context}")
        else:
            text_parts.append(main_part)
    else:
        text_parts.append(cleaned_id)
    
    # Add node type if available
    if 'node_type' in node_data and node_data['node_type'] != 'entity':
        text_parts.append(f"({node_data['node_type']})")
    
    # Add description if available
    if 'description' in node_data:
        text_parts.append(f"- {node_data['description']}")
    
    return " ".join(text_parts).strip()


def generate_edge_textual_value(source_id: str, target_id: str, edge_data: Dict[str, Any]) -> str:
    """
    Generate textual representation for an edge
    """
    relation = edge_data.get('relation', edge_data.get('relation_type', 'related to'))
    
    # Clean relation type
    cleaned_relation = relation.replace('_', ' ').lower()
    
    # Add temporal information if available
    if 'timestamp' in edge_data:
        timestamp = edge_data['timestamp']
        return f"{cleaned_relation} in {timestamp}"
    
    return cleaned_relation


def create_entity_name_map(graph: nx.DiGraph) -> Dict[str, List[str]]:
    """
    Create a mapping from simplified names to entity IDs
    """
    name_map = {}
    
    for node_id in graph.nodes:
        # Extract all variations
        variations = extract_entity_variations(node_id)
        
        for var in variations:
            var_lower = var.lower()
            if var_lower not in name_map:
                name_map[var_lower] = []
            name_map[var_lower].append(node_id)
    
    return name_map


def extract_entity_variations(entity_id: str) -> List[str]:
    """Extract all reasonable variations of an entity name"""
    variations = [entity_id]
    
    # Basic cleaning
    cleaned = entity_id.replace('_', ' ').replace('/', ' ')
    variations.append(cleaned)
    
    # Handle parentheses
    if '(' in entity_id:
        # Extract parts
        main_part = entity_id.split('(')[0].strip('_')
        variations.append(main_part)
        variations.append(main_part.replace('_', ' '))
        
        # Extract context
        context_match = re.search(r'\(([^)]+)\)', entity_id)
        if context_match:
            context = context_match.group(1)
            variations.append(context)
    
    # Split compound names
    if '_' in entity_id:
        parts = entity_id.split('_')
        variations.extend(parts)
        
        # Try last part as primary name
        if len(parts) > 1:
            variations.append(parts[-1])
    
    # Remove empty and duplicate variations
    cleaned_variations = []
    seen = set()
    for var in variations:
        var = var.strip().strip('_/')
        if var and len(var) > 1 and var.lower() not in seen:
            cleaned_variations.append(var)
            seen.add(var.lower())
    
    return cleaned_variations
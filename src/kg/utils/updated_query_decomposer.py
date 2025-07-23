"""
Updated query decomposer with entity resolution capabilities
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
from .entity_resolution import EntityResolver
import logging

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """
    Query decomposer that uses entity resolution for better entity matching
    """
    
    def __init__(self, graph: nx.DiGraph, llm_manager: Any):
        self.graph = graph
        self.llm_manager = llm_manager
        self.entity_resolver = EntityResolver(graph)
        
    def decompose_query(self, query: str, iteration: int = 0, 
                       previous_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Decompose query with entity resolution
        """
        # First, use LLM to extract entities from the query
        entities = self.extract_entities_from_query(query)
        
        # Resolve entities to graph IDs
        resolved_entities = []
        unresolved_entities = []
        
        for entity in entities:
            resolved = self.entity_resolver.resolve(entity)
            if resolved:
                resolved_entities.append({
                    'mention': entity,
                    'resolved_id': resolved,
                    'info': self.entity_resolver.get_entity_info(resolved)
                })
                logger.info(f"Resolved '{entity}' -> '{resolved}'")
            else:
                unresolved_entities.append(entity)
                logger.warning(f"Could not resolve entity: '{entity}'")
        
        # Create sub-query based on resolved entities
        sub_query = self.create_sub_query(query, resolved_entities, unresolved_entities, iteration, previous_context)
        
        # Extract temporal constraints
        temporal_constraints = self.extract_temporal_constraints(query)
        
        return {
            'original_query': query,
            'sub_query': sub_query,
            'resolved_entities': resolved_entities,
            'unresolved_entities': unresolved_entities,
            'temporal_constraints': temporal_constraints,
            'source_entities': [e['resolved_id'] for e in resolved_entities],
            'target_entities': []  # Will be populated based on query type
        }
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entity mentions from query"""
        
        prompt = f"""
        Extract all entity mentions from the following query. Include:
        - Person names
        - Organisation names
        - Place names
        - Government bodies
        - Any other proper nouns
        
        Query: {query}
        
        Return as JSON list of entity mentions:
        """
        
        try:
            response = self.llm_manager.generate(prompt)
            
            # Extract JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
                return entities
            else:
                # Fallback: extract capitalised words
                return self.extract_entities_fallback(query)
                
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return self.extract_entities_fallback(query)
    
    def extract_entities_fallback(self, query: str) -> List[str]:
        """Fallback entity extraction using patterns"""
        entities = []
        
        # Extract capitalised sequences
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(cap_pattern, query)
        entities.extend(matches)
        
        # Extract quoted entities
        quote_pattern = r'["\']([^"\']+)["\']'
        matches = re.findall(quote_pattern, query)
        entities.extend(matches)
        
        # Remove duplicates
        return list(set(entities))
    
    def create_sub_query(self, original_query: str, 
                         resolved_entities: List[Dict[str, Any]], 
                         unresolved_entities: List[str],
                         iteration: int,
                         previous_context: Optional[str]) -> str:
        """Create sub-query based on resolved entities"""
        
        if iteration == 0:
            # First iteration - focus on finding connections
            if resolved_entities:
                entity_names = [e['mention'] for e in resolved_entities]
                entity_ids = [e['resolved_id'] for e in resolved_entities]
                
                sub_query = f"Find connections and relationships involving: {', '.join(entity_names)}"
                sub_query += f" (Graph IDs: {', '.join(entity_ids)})"
                
                if unresolved_entities:
                    sub_query += f" Also look for entities similar to: {', '.join(unresolved_entities)}"
            else:
                sub_query = original_query  # Fallback to original
                
        else:
            # Subsequent iterations - refine based on context
            if previous_context:
                sub_query = f"Based on previous findings, explore further connections for {original_query}"
            else:
                sub_query = f"Continue searching for information about {original_query}"
        
        return sub_query
    
    def extract_temporal_constraints(self, query: str) -> Dict[str, Any]:
        """Extract temporal constraints from query"""
        constraints = {
            'has_temporal': False,
            'years': [],
            'time_expressions': [],
            'temporal_relations': []
        }
        
        # Extract years
        year_pattern = r'\b(1\d{3}|2\d{3})\b'
        years = re.findall(year_pattern, query)
        if years:
            constraints['has_temporal'] = True
            constraints['years'] = years
        
        # Extract temporal keywords
        temporal_keywords = {
            'before': ['before', 'prior to', 'earlier than'],
            'after': ['after', 'following', 'later than'],
            'during': ['during', 'in', 'within'],
            'between': ['between', 'from.*to']
        }
        
        query_lower = query.lower()
        for relation, keywords in temporal_keywords.items():
            for keyword in keywords:
                if re.search(keyword, query_lower):
                    constraints['temporal_relations'].append(relation)
                    constraints['has_temporal'] = True
        
        return constraints
    
    def resolve_entities_in_paths(self, paths: List[Any]) -> List[Any]:
        """
        Ensure all entities in paths have proper textual representations
        """
        enhanced_paths = []
        
        for path in paths:
            # Ensure nodes have tv attribute
            for node in path.nodes:
                if not hasattr(node, 'tv') or not node.tv:
                    # Generate tv from node ID
                    node.tv = self.generate_node_tv(node.id)
            
            # Ensure edges have te attribute
            for edge in path.edges:
                if not hasattr(edge, 'te') or not edge.te:
                    # Generate te from edge data
                    edge.te = self.generate_edge_te(edge)
            
            enhanced_paths.append(path)
        
        return enhanced_paths
    
    def generate_node_tv(self, node_id: str) -> str:
        """Generate textual value for a node"""
        # Clean the node ID
        tv = node_id.replace('_', ' ').replace('/', ' or ')
        
        # Handle parentheses
        if '(' in tv and ')' in tv:
            main_part = tv.split('(')[0].strip()
            context = re.search(r'\(([^)]+)\)', tv)
            if context:
                tv = f"{main_part} of {context.group(1)}"
        
        return tv
    
    def generate_edge_te(self, edge: Any) -> str:
        """Generate textual value for an edge"""
        relation = getattr(edge, 'relation_type', 'related to')
        te = relation.replace('_', ' ').lower()
        
        if hasattr(edge, 'timestamp') and edge.timestamp:
            te += f" in {edge.timestamp}"
        
        return te


def integrate_with_reasoner(reasoner: Any) -> None:
    """
    Integrate query decomposer with temporal iterative reasoner
    """
    # Create decomposer
    decomposer = QueryDecomposer(reasoner.tkg_engine.graph, reasoner.llm_manager)
    
    # Monkey-patch the decompose_query method
    # The method is actually at reasoner.temporal_decomposer.decompose_query
    if hasattr(reasoner, 'temporal_decomposer') and hasattr(reasoner.temporal_decomposer, 'decompose_query'):
        original_decompose = reasoner.temporal_decomposer.decompose_query
        
        def decompose_query(query: str, iteration: int = 0, 
                                    previous_findings: Optional[List] = None) -> Dict[str, Any]:
            # Use decomposer
            result = decomposer.decompose_query(query, iteration, 
                                               previous_findings[-1] if previous_findings else None)
            
            # Maintain compatibility with original format
            return {
                'sub_query': result['sub_query'],
                'temporal_constraints': result['temporal_constraints'],
                'entities': result['source_entities']
            }
        
        reasoner.temporal_decomposer.decompose_query = decompose_query
    else:
        # Fallback to original location if structure is different
        original_decompose = reasoner.decompose_query
        
        def decompose_query(query: str, iteration: int = 0, 
                                    previous_findings: Optional[List] = None) -> Dict[str, Any]:
            # Use decomposer
            result = decomposer.decompose_query(query, iteration, 
                                               previous_findings[-1] if previous_findings else None)
            
            # Maintain compatibility with original format
            return {
                'sub_query': result['sub_query'],
                'temporal_constraints': result['temporal_constraints'],
                'entities': result['source_entities']
            }
        
        reasoner.decompose_query = decompose_query
    logger.info("Query decomposer integrated with reasoner")
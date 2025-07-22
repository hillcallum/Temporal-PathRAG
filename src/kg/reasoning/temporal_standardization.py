"""
Module for standardizing temporal data formats across the PathRAG system.
Ensures consistent handling of temporal information in nodes, edges, and queries.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TemporalStandardizer:
    """Standardize temporal data formats across components."""
    
    # Standard temporal attribute names
    TEMPORAL_EDGE_ATTR = 'te'  # temporal edge validity
    TEMPORAL_NODE_ATTR = 'tv'  # temporal node validity
    TIMESTAMP_ATTR = 'timestamp'
    RELATION_ATTR = 'relation'
    WEIGHT_ATTR = 'weight'
    
    # Alternative names mapping
    TEMPORAL_MAPPINGS = {
        'te': ['temporal_edge', 'time_validity', 'edge_time', 'temporal_validity'],
        'tv': ['temporal_validity', 'node_time', 'time_valid'],
        'timestamp': ['time', 'date', 'year', 't', 'ts'],
        'relation': ['predicate', 'rel', 'type', 'edge_type', 'relationship'],
        'weight': ['score', 'confidence', 'strength', 'w']
    }
    
    @classmethod
    def standardize_edge_data(cls, edge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize edge data to use consistent attribute names.
        
        Args:
            edge_data: Raw edge data dictionary
            
        Returns:
            Standardized edge data
        """
        if not edge_data:
            return {}
            
        standardized = edge_data.copy()
        
        # Standardize attribute names
        for standard_name, alternatives in cls.TEMPORAL_MAPPINGS.items():
            if standard_name not in standardized:
                for alt in alternatives:
                    if alt in edge_data:
                        standardized[standard_name] = edge_data[alt]
                        # Optionally remove the alternative name
                        # del standardized[alt]
                        break
                        
        # Standardize temporal formats
        if cls.TEMPORAL_EDGE_ATTR in standardized:
            standardized[cls.TEMPORAL_EDGE_ATTR] = cls.standardize_temporal_value(
                standardized[cls.TEMPORAL_EDGE_ATTR]
            )
            
        if cls.TIMESTAMP_ATTR in standardized:
            standardized[cls.TIMESTAMP_ATTR] = cls.standardize_temporal_value(
                standardized[cls.TIMESTAMP_ATTR]
            )
            
        return standardized
    
    @classmethod
    def standardize_node_data(cls, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize node data to use consistent attribute names.
        
        Args:
            node_data: Raw node data dictionary
            
        Returns:
            Standardized node data
        """
        if not node_data:
            return {}
            
        standardized = node_data.copy()
        
        # Standardize attribute names
        for standard_name, alternatives in cls.TEMPORAL_MAPPINGS.items():
            if standard_name not in standardized:
                for alt in alternatives:
                    if alt in node_data:
                        standardized[standard_name] = node_data[alt]
                        break
                        
        # Standardize temporal formats
        if cls.TEMPORAL_NODE_ATTR in standardized:
            standardized[cls.TEMPORAL_NODE_ATTR] = cls.standardize_temporal_value(
                standardized[cls.TEMPORAL_NODE_ATTR]
            )
            
        if cls.TIMESTAMP_ATTR in standardized:
            standardized[cls.TIMESTAMP_ATTR] = cls.standardize_temporal_value(
                standardized[cls.TIMESTAMP_ATTR]
            )
            
        return standardized
    
    @classmethod
    def standardize_temporal_value(cls, value: Any) -> Union[Tuple[Optional[str], Optional[str]], str, None]:
        """
        Standardize a temporal value to consistent format.
        
        Outputs:
        - Single time point: "YYYY" or "YYYY-MM-DD"
        - Time range: (start, end) tuple
        - None if invalid
        
        Args:
            value: Raw temporal value
            
        Returns:
            Standardized temporal value
        """
        if value is None:
            return None
            
        # Handle tuple/list (time range)
        if isinstance(value, (list, tuple)):
            if len(value) == 2:
                start = cls._parse_single_time(value[0])
                end = cls._parse_single_time(value[1])
                return (start, end)
            elif len(value) == 1:
                return cls._parse_single_time(value[0])
                
        # Handle dict format
        elif isinstance(value, dict):
            start = value.get('start') or value.get('from') or value.get('begin')
            end = value.get('end') or value.get('to') or value.get('until')
            
            if start or end:
                return (cls._parse_single_time(start), cls._parse_single_time(end))
            elif 'time' in value:
                return cls._parse_single_time(value['time'])
            elif 'year' in value:
                return cls._parse_single_time(value['year'])
                
        # Handle single value
        else:
            return cls._parse_single_time(value)
            
        return None
    
    @classmethod
    def _parse_single_time(cls, time_value: Any) -> Optional[str]:
        """Parse a single time value to standard format."""
        if time_value is None:
            return None
            
        # Convert to string
        time_str = str(time_value).strip()
        
        if not time_str:
            return None
            
        # Try to extract year
        year_match = re.search(r'(\d{4})', time_str)
        if year_match:
            year = year_match.group(1)
            
            # Check for full date
            date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', time_str)
            if date_match:
                return f"{date_match.group(1)}-{date_match.group(2).zfill(2)}-{date_match.group(3).zfill(2)}"
            else:
                return year
                
        # Try to parse as integer year
        try:
            year_int = int(float(time_str))
            if 1000 <= year_int <= 3000:  # Reasonable year range
                return str(year_int)
        except:
            pass
            
        # Return original if can't parse
        return time_str
    
    @classmethod
    def extract_temporal_context(cls, query: str) -> Dict[str, Any]:
        """
        Extract temporal context from a query string.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with temporal context
        """
        context = {
            'temporal_type': 'any',
            'query_time': None,
            'start_time': None,
            'end_time': None
        }
        
        query_lower = query.lower()
        
        # Detect temporal type
        if 'before' in query_lower:
            context['temporal_type'] = 'before'
        elif 'after' in query_lower:
            context['temporal_type'] = 'after'
        elif 'between' in query_lower or ('from' in query_lower and 'to' in query_lower):
            context['temporal_type'] = 'range'
        elif 'during' in query_lower or 'in' in query_lower:
            context['temporal_type'] = 'point'
            
        # Extract years/dates
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years = re.findall(year_pattern, query)
        
        if years:
            if context['temporal_type'] == 'range' and len(years) >= 2:
                context['start_time'] = years[0]
                context['end_time'] = years[1]
            elif context['temporal_type'] == 'before' and years:
                context['query_time'] = years[0]
            elif context['temporal_type'] == 'after' and years:
                context['query_time'] = years[0]
            else:
                context['query_time'] = years[0]
                
        return context
    
    @classmethod
    def format_temporal_for_display(cls, temporal_value: Any) -> str:
        """
        Format temporal value for human-readable display.
        
        Args:
            temporal_value: Standardized temporal value
            
        Returns:
            Human-readable string
        """
        if temporal_value is None:
            return "unknown time"
            
        if isinstance(temporal_value, (list, tuple)) and len(temporal_value) == 2:
            start, end = temporal_value
            if start and end:
                return f"from {start} to {end}"
            elif start:
                return f"since {start}"
            elif end:
                return f"until {end}"
                
        return str(temporal_value)
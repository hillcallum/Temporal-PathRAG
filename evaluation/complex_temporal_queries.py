"""
Complex Temporal Query Dataset
Challenging queries that require sophisticated temporal reasoning
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class ComplexTemporalQuery:
    """Represents a complex temporal query with metadata"""
    query: str
    query_type: str
    temporal_complexity: str  # simple, moderate, complex, extreme
    required_capabilities: List[str]
    expected_reasoning_steps: List[str]
    example_answer: str
    evaluation_notes: str

class ComplexQueryDataset:
    """Collection of complex temporal queries for testing"""
    
    def __init__(self):
        self.queries = self.create_queries()
    
    def create_queries(self) -> List[ComplexTemporalQuery]:
        """Create a set of complex temporal queries"""
        
        queries = [
            # Geopolitical Events
            ComplexTemporalQuery(
                query="Who were the first European leaders that Joe Biden met after the Ukraine War started?",
                query_type="sequence_after_event",
                temporal_complexity="complex",
                required_capabilities=[
                    "event_detection",
                    "temporal_ordering",
                    "entity_classification",
                    "multi_hop_reasoning"
                ],
                expected_reasoning_steps=[
                    "Identify when Ukraine War started (Feb 24, 2022)",
                    "Find all Biden meetings with leaders after this date",
                    "Filter for European leaders only",
                    "Order by date to find 'first'"
                ],
                example_answer="Emmanuel Macron and Olaf Scholz at NATO summit March 2022",
                evaluation_notes="Requires understanding 'European', temporal ordering, and event causality"
            ),
            
            ComplexTemporalQuery(
                query="Which sanctions were imposed on Russia between the annexation of Crimea and the start of the Ukraine War?",
                query_type="range_between_events",
                temporal_complexity="complex",
                required_capabilities=[
                    "event_boundary_detection",
                    "temporal_range_filtering",
                    "entity_relation_tracking"
                ],
                expected_reasoning_steps=[
                    "Identify Crimea annexation date (March 2014)",
                    "Identify Ukraine War start (February 2022)",
                    "Find all sanctions in 2014-2022 range",
                    "Filter for Russia as target"
                ],
                example_answer="Multiple rounds including SWIFT restrictions, asset freezes, travel bans",
                evaluation_notes="Requires precise event dating and range queries"
            ),
            
            # Scientific/Medical Events
            ComplexTemporalQuery(
                query="Which COVID vaccines were approved before any variant of concern was identified?",
                query_type="sequence_before_event",
                temporal_complexity="complex",
                required_capabilities=[
                    "temporal_precedence",
                    "entity_subtype_understanding",
                    "negative_temporal_reasoning"
                ],
                expected_reasoning_steps=[
                    "Identify first variant of concern date (Alpha, Dec 2020)",
                    "Find vaccine approvals before this date",
                    "Distinguish approvals from development/trials"
                ],
                example_answer="None - first VOC identified before first approval",
                evaluation_notes="Requires understanding 'variant of concern' vs regular variants"
            ),
            
            # Business/Economic Events
            ComplexTemporalQuery(
                query="How did tech company layoffs change in frequency during the 6 months after the Fed's first 2022 rate hike?",
                query_type="temporal_trend_analysis",
                temporal_complexity="extreme",
                required_capabilities=[
                    "event_frequency_analysis",
                    "temporal_window_calculation",
                    "trend_detection",
                    "causal_reasoning"
                ],
                expected_reasoning_steps=[
                    "Identify Fed's first 2022 rate hike (March 16, 2022)",
                    "Define 6-month window (March-September 2022)",
                    "Count tech layoffs in this period",
                    "Compare to baseline frequency",
                    "Describe trend"
                ],
                example_answer="Increased 3x from baseline, with major layoffs at Meta, Twitter, Amazon",
                evaluation_notes="Requires frequency analysis and trend description"
            ),
            
            # Multi-Event Intersection
            ComplexTemporalQuery(
                query="Which world leaders attended both the Queen's funeral and COP27 in 2022?",
                query_type="event_intersection",
                temporal_complexity="moderate",
                required_capabilities=[
                    "event_participant_tracking",
                    "set_intersection",
                    "temporal_proximity"
                ],
                expected_reasoning_steps=[
                    "Identify Queen's funeral attendees (Sept 19, 2022)",
                    "Identify COP27 attendees (Nov 6-18, 2022)",
                    "Find intersection of attendee lists",
                    "Verify both events in 2022"
                ],
                example_answer="Joe Biden, Emmanuel Macron, Justin Trudeau",
                evaluation_notes="Requires maintaining attendance lists and set operations"
            ),
            
            # Temporal Duration Queries
            ComplexTemporalQuery(
                query="Which company had the longest continuous period as the world's most valuable between 2010 and 2023?",
                query_type="duration_supremacy",
                temporal_complexity="extreme",
                required_capabilities=[
                    "continuous_period_tracking",
                    "value_comparison",
                    "temporal_aggregation",
                    "superlative_reasoning"
                ],
                expected_reasoning_steps=[
                    "Track market cap rankings 2010-2023",
                    "Identify #1 company each quarter",
                    "Find continuous periods for each company",
                    "Compare period lengths"
                ],
                example_answer="Apple (2011-2022 with brief interruptions)",
                evaluation_notes="Requires tracking continuous periods and handling interruptions"
            ),
            
            # Causal Chain Queries
            ComplexTemporalQuery(
                query="What major events happened within 30 days after each Fed rate hike in 2022?",
                query_type="causal_window_multiple",
                temporal_complexity="extreme",
                required_capabilities=[
                    "multiple_event_tracking",
                    "temporal_window_application",
                    "event_significance_filtering",
                    "causal_correlation"
                ],
                expected_reasoning_steps=[
                    "Identify all Fed rate hikes in 2022",
                    "Create 30-day window after each",
                    "Find major events in each window",
                    "Filter for significance",
                    "Group by rate hike"
                ],
                example_answer="March: Russian oil sanctions; May: Luna crash; June: Celsius bankruptcy",
                evaluation_notes="Requires multiple temporal windows and event significance assessment"
            ),
            
            # Negative Temporal Queries
            ComplexTemporalQuery(
                query="Which G7 countries have never had a female head of government as of 2023?",
                query_type="temporal_negation",
                temporal_complexity="moderate",
                required_capabilities=[
                    "historical_scanning",
                    "negative_existence",
                    "role_understanding"
                ],
                expected_reasoning_steps=[
                    "Identify G7 members",
                    "Scan all historical leaders",
                    "Check gender of each",
                    "Identify countries with no female leaders"
                ],
                example_answer="Japan and the United States",
                evaluation_notes="Requires exhaustive historical search and negation"
            ),
            
            # Relative Temporal Queries
            ComplexTemporalQuery(
                query="Who was the UK Prime Minister when the previous monarch before Elizabeth II died?",
                query_type="relative_temporal_reference",
                temporal_complexity="complex",
                required_capabilities=[
                    "relative_reference_resolution",
                    "predecessor_identification",
                    "temporal_alignment"
                ],
                expected_reasoning_steps=[
                    "Identify Elizabeth II's predecessor (George VI)",
                    "Find George VI's death date (Feb 6, 1952)",
                    "Identify UK PM on that date"
                ],
                example_answer="Winston Churchill",
                evaluation_notes="Requires resolving 'previous monarch' and temporal alignment"
            ),
            
            # Periodic Event Queries
            ComplexTemporalQuery(
                query="How many times did the Fed reverse its rate decision within 6 months between 2000-2023?",
                query_type="periodic_reversal_count",
                temporal_complexity="extreme",
                required_capabilities=[
                    "decision_tracking",
                    "reversal_detection",
                    "temporal_window_counting",
                    "long_range_analysis"
                ],
                expected_reasoning_steps=[
                    "Track all Fed rate decisions 2000-2023",
                    "For each decision, check next 6 months",
                    "Identify reversals (hike→cut or cut→hike)",
                    "Count total reversals"
                ],
                example_answer="3 times: 2001, 2007, 2019",
                evaluation_notes="Requires tracking decisions and detecting reversals over 23 years"
            )
        ]
        
        return queries
    
    def get_queries_by_complexity(self, complexity: str) -> List[ComplexTemporalQuery]:
        """Get queries filtered by complexity level"""
        return [q for q in self.queries if q.temporal_complexity == complexity]
    
    def get_queries_by_type(self, query_type: str) -> List[ComplexTemporalQuery]:
        """Get queries filtered by type"""
        return [q for q in self.queries if q.query_type == query_type]
    
    def get_queries_requiring_capability(self, capability: str) -> List[ComplexTemporalQuery]:
        """Get queries that require a specific capability"""
        return [q for q in self.queries if capability in q.required_capabilities]
    
    def export_for_evaluation(self, output_path: Path):
        """Export queries in format suitable for evaluation"""
        export_data = []
        
        for i, query in enumerate(self.queries):
            export_data.append({
                'id': f'complex_{i:03d}',
                'query': query.query,
                'type': query.query_type,
                'complexity': query.temporal_complexity,
                'answer': query.example_answer,
                'metadata': {
                    'required_capabilities': query.required_capabilities,
                    'reasoning_steps': query.expected_reasoning_steps,
                    'notes': query.evaluation_notes
                }
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def get_capability_distribution(self) -> Dict[str, int]:
        """Analyse distribution of required capabilities"""
        capability_counts = {}
        
        for query in self.queries:
            for capability in query.required_capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
        
        return dict(sorted(capability_counts.items(), key=lambda x: x[1], reverse=True))


def create_evaluation_challenge():
    """Create a challenging evaluation set"""
    dataset = ComplexQueryDataset()
    
    print("Complex Temporal Query Dataset Statistics:")
    print(f"Total queries: {len(dataset.queries)}")
    print(f"\nComplexity distribution:")
    for complexity in ['simple', 'moderate', 'complex', 'extreme']:
        count = len(dataset.get_queries_by_complexity(complexity))
        print(f"  {complexity}: {count}")
    
    print(f"\nRequired capabilities:")
    for capability, count in dataset.get_capability_distribution().items():
        print(f"  {capability}: {count} queries")
    
    # Export for evaluation
    output_path = Path("evaluation/complex_temporal_queries.json")
    dataset.export_for_evaluation(output_path)
    print(f"\nExported to {output_path}")


if __name__ == "__main__":
    create_evaluation_challenge()
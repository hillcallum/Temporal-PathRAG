#!/usr/bin/env python3
"""
Demo script for Temporal Iterative Reasoning
Tests the new TemporalIterativeReasoner with sample temporal queries
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import pickle
from datetime import datetime
from src.kg.temporal_iterative_reasoner import TemporalIterativeReasoner
from src.kg.tkg_query_engine import TKGQueryEngine
from src.llm.llm_manager import LLMManager
from src.utils.config import get_config
from src.utils.device import get_device
from src.llm.config import llm_config


def load_sample_graph():
    """Load a sample temporal knowledge graph for testing"""
    
    # Try to load from saved analysis results
    graph_path = "analysis_results/temporal_graph_db/main_graph.pkl"
    
    if os.path.exists(graph_path):
        print(f"Loading graph from {graph_path}")
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        return graph
    
    # Fallback: create a simple test graph
    print("Creating simple test graph")
    import networkx as nx
    from src.kg.models import TemporalPathRAGNode, TemporalPathRAGEdge
    
    graph = nx.DiGraph()
    
    # Add sample nodes
    nodes = [
        TemporalPathRAGNode(id="barack_obama", name="Barack Obama", entity_type="Person", 
                           description="44th President of the United States"),
        TemporalPathRAGNode(id="michelle_obama", name="Michelle Obama", entity_type="Person", 
                           description="Former First Lady of the United States"),
        TemporalPathRAGNode(id="white_house", name="White House", entity_type="Location", 
                           description="Official residence of the US President"),
        TemporalPathRAGNode(id="harvard_law", name="Harvard Law School", entity_type="Institution", 
                           description="Law school at Harvard University"),
        TemporalPathRAGNode(id="chicago", name="Chicago", entity_type="Location", 
                           description="City in Illinois"),
    ]
    
    for node in nodes:
        graph.add_node(node.id, **node.__dict__)
    
    # Add sample edges with temporal information
    edges = [
        TemporalPathRAGEdge(source_id="barack_obama", target_id="michelle_obama", 
                           relation_type="married_to", timestamp="1992-10-03T00:00:00",
                           description="Barack Obama married Michelle Obama"),
        TemporalPathRAGEdge(source_id="barack_obama", target_id="white_house", 
                           relation_type="resided_at", timestamp="2009-01-20T00:00:00",
                           description="Barack Obama moved to White House as President"),
        TemporalPathRAGEdge(source_id="barack_obama", target_id="harvard_law", 
                           relation_type="graduated_from", timestamp="1991-06-15T00:00:00",
                           description="Barack Obama graduated from Harvard Law School"),
        TemporalPathRAGEdge(source_id="michelle_obama", target_id="chicago", 
                           relation_type="born_in", timestamp="1964-01-17T00:00:00",
                           description="Michelle Obama was born in Chicago"),
    ]
    
    for edge in edges:
        graph.add_edge(edge.source_id, edge.target_id, **edge.__dict__)
    
    return graph


def run_iterative_reasoning_demo():
    """Run demonstration of iterative reasoning"""
    
    print("=== Temporal Iterative Reasoning Demo ===")
    print()
    
    # Load configuration
    config = get_config()
    device = get_device()
    
    # Load sample graph
    graph = load_sample_graph()
    print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    print()
    
    # Initialise LLM manager
    llm_manager = LLMManager(llm_config)
    
    # Initialise TKG Query Engine
    tkg_engine = TKGQueryEngine(
        graph=graph,
        alpha=0.01,
        base_theta=0.1,
        device=device
    )
    
    # Initialise Iterative Reasoner
    iterative_reasoner = TemporalIterativeReasoner(
        tkg_query_engine=tkg_engine,
        llm_manager=llm_manager,
        max_iterations=3,  # Limit for demo
        convergence_threshold=0.7,
        temporal_coverage_threshold=0.6
    )
    
    # Test queries
    test_queries = [
        "What was Barack Obama's educational and career progression before becoming President?",
        "How are Barack Obama and Michelle Obama connected through their life events?",
        "What temporal relationships exist between the Obama family and Chicago?",
        "When did Barack Obama's presidential career begin and what led to it?"
    ]
    
    print("Testing iterative reasoning on sample queries")
    print()
    
    for i, query in enumerate(test_queries):
        print(f"Query {i+1}: {query}")
        print("-" * 80)
        
        try:
            # Run iterative reasoning
            result = iterative_reasoner.reason_iteratively(query, verbose=True)
            
            print(f"Final Answer: {result.final_answer}")
            print(f"Reasoning Steps: {len(result.reasoning_steps)}")
            print(f"Total Paths Retrieved: {result.total_paths_retrieved}")
            print(f"Execution Time: {result.total_execution_time:.2f}s")
            print(f"Convergence Reason: {result.convergence_reason}")
            
            # Show reasoning steps
            print("\nReasoning Steps:")
            for step in result.reasoning_steps:
                print(f"Step {step.step_id + 1}: {step.sub_query}")
                print(f"Paths Retrieved: {len(step.retrieved_paths)}")
                print(f"Sufficient: {step.is_sufficient}")
                print(f"Temporal Coverage: {step.temporal_coverage.get('coverage_score', 0):.2f}")
                
                if step.next_exploration_hints:
                    print(f"Next Hints: {', '.join(step.next_exploration_hints)}")
            
            # Get reasoning summary
            summary = iterative_reasoner.get_reasoning_summary(result)
            print(f"\nSummary: {json.dumps(summary, indent=2, default=str)}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        print("=" * 80)
        print()


def test_query_decomposition():
    """Test query decomposition component separately"""
    
    print("=== Testing Query Decomposition ===")
    print()
    
    # Initialise LLM manager
    llm_manager = LLMManager(llm_config)
    
    from src.kg.temporal_iterative_reasoner import TemporalQueryDecomposer
    decomposer = TemporalQueryDecomposer(llm_manager)
    
    test_queries = [
        "What happened to Barack Obama after he graduated from Harvard Law School?",
        "How did Michelle Obama's early life in Chicago influence her later career?",
        "What sequence of events led to Barack Obama becoming President?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("Decomposition:")
        
        try:
            sub_questions = decomposer.decompose_query(query)
            for i, sub_q in enumerate(sub_questions):
                print(f"{i+1}. {sub_q['sub_query']}")
                print(f"Focus: {sub_q.get('exploration_focus', 'N/A')}")
                print(f"Depends on: {sub_q.get('depends_on', [])}")
                print(f"Temporal constraints: {sub_q.get('temporal_constraints', {})}")
                print()
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 60)
        print()


def main():
    """Main demo function"""
    
    print("Temporal Iterative Reasoning Demo")
    print("=" * 50)
    
    # Check if LLM is available
    try:
        llm_manager = LLMManager(llm_config)
        
        # Test basic LLM functionality
        test_response = llm_manager.generate_response("Hello, this is a test.")
        print(f"LLM available: {len(test_response) > 0}")
        
    except Exception as e:
        print(f"LLM not available: {e}")
        print("Proceeding with limited functionality")
    
    print()
    
    # Run tests
    try:
        # Test query decomposition
        test_query_decomposition()
        
        # Run full iterative reasoning demo
        run_iterative_reasoning_demo()
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
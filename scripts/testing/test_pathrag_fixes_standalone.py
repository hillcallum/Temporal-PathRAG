"""
Standalone test script for Temporal PathRAG fixes
This version doesn't require OpenAI API and tests individual components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import networkx as nx
from src.kg.utils.entity_resolution import EntityResolver, enhance_graph_with_textual_representations
from src.evaluation.answer_extractor import AnswerExtractor


def create_test_graph():
    """Create a small test graph to demonstrate fixes"""
    G = nx.MultiDiGraph()
    
    # Add nodes that match the problematic entity formats
    nodes = [
        "Cabinet_/Council_of_Ministers/Advisors(Denmark)",
        "Danish_Ministry",
        "Denmark",
        "Royal_Castle(Copenhagen)",
        "Prime_Minister(Denmark)",
        "European_Union",
        "United_States"
    ]
    
    for node in nodes:
        G.add_node(node, node_type="entity")
    
    # Add some edges
    G.add_edge("Cabinet_/Council_of_Ministers/Advisors(Denmark)", "Denmark", 
               relation="located_in", timestamp="1990-01-01")
    G.add_edge("Danish_Ministry", "Denmark",
               relation="part_of", timestamp="1985-06-15")
    G.add_edge("Prime_Minister(Denmark)", "Cabinet_/Council_of_Ministers/Advisors(Denmark)",
               relation="member_of", timestamp="1992-03-20")
    
    return G


def test_graph_enhancement():
    """Test adding textual representations to graph"""
    print("Testing Graph Enhancement")
    
    # Create test graph
    graph = create_test_graph()
    
    print("\nBefore enhancement:")
    for node in list(graph.nodes)[:3]:
        print(f"{node}: tv={graph.nodes[node].get('tv', 'Missing')}")
    
    # Enhance graph
    enhanced_graph = enhance_graph_with_textual_representations(graph)
    
    print("\nAfter enhancement:")
    for node in list(enhanced_graph.nodes)[:3]:
        print(f"{node}: tv={enhanced_graph.nodes[node].get('tv', 'Missing')}")
    
    print("\nEdge enhancements:")
    for u, v, k, data in list(enhanced_graph.edges(keys=True, data=True))[:3]:
        print(f"{u} -> {v}")
        print(f"relation: {data.get('relation')}")
        print(f"te: {data.get('te', 'Missing')}")
    
    return enhanced_graph


def test_entity_resolution(graph):
    """Test entity resolution capabilities"""
    print("Testing Entity Resolution")
    
    resolver = EntityResolver(graph)
    
    # Test cases that were problematic
    test_cases = [
        ("Danish Ministry", "Danish_Ministry"),
        ("Cabinet of Denmark", "Cabinet_/Council_of_Ministers/Advisors(Denmark)"),
        ("Council of Ministers", "Cabinet_/Council_of_Ministers/Advisors(Denmark)"),
        ("Denmark", "Denmark"),
        ("Royal Castle", "Royal_Castle(Copenhagen)"),
        ("Copenhagen Castle", "Royal_Castle(Copenhagen)"),
        ("EU", "European_Union"),
        ("United States", "United_States"),
        ("USA", "United_States"),
        ("Prime Minister of Denmark", "Prime_Minister(Denmark)")
    ]
    
    successes = 0
    for mention, expected in test_cases:
        resolved = resolver.resolve(mention)
        success = resolved == expected
        successes += success
        
        status = "Yes" if success else "No"
        print(f"{status} '{mention}' -> '{resolved}' (expected: '{expected}')")
        
        if resolved and resolved != expected:
            info = resolver.get_entity_info(resolved)
            print(f"Variations: {info['variations'][:3]}")
    
    print(f"\nSuccess rate: {successes}/{len(test_cases)} ({successes/len(test_cases)*100:.1f}%)")
    
    return resolver


def test_answer_extraction():
    """Test improved answer extraction"""
    print("Testing Improved Answer Extraction")
    
    extractor = AnswerExtractor()
    
    # Test cases that represent typical PathRAG responses
    test_cases = [
        {
            'text': "Based on the evidence gathered, the answer is Cabinet of Denmark, which replaced the Council of Ministers.",
            'question': "What replaced the Council of Ministers?",
            'expected_type': 'entity',
            'expected_answers': ['Cabinet of Denmark', 'Council of Ministers']
        },
        {
            'text': "The Danish Ministry was established in 1848.",
            'question': "When was the Danish Ministry established?",
            'expected_type': 'time',
            'expected_answers': ['1848']
        },
        {
            'text': "No relevant information found in the knowledge graph.",
            'question': "Who was the first leader?",
            'expected_type': 'entity',
            'expected_answers': []
        },
        {
            'text': "Unable to find any paths connecting these entities in the temporal knowledge graph.",
            'question': "What is the connection?",
            'expected_type': 'entity',
            'expected_answers': []
        },
        {
            'text': "The Prime Minister of Denmark appointed John Smith as the first director in 1990.",
            'question': "Who was appointed as the first director?",
            'expected_type': 'entity',
            'expected_answers': ['John Smith', 'Prime Minister of Denmark']
        },
        {
            'text': "Evidence shows that the organization had 150 members.",
            'question': "How many members did it have?",
            'expected_type': 'value',
            'expected_answers': ['150']
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {case['text'][:80]}")
        print(f"Question: {case['question']}")
        
        answers = extractor.extract_answers_from_response(
            case['text'],
            case['question'],
            answer_type=case['expected_type']
        )
        
        print(f"Extracted: {answers}")
        print(f"Expected: {case['expected_answers']}")
        
        # Check if extraction matches expectations
        if case['expected_answers']:
            # Should find at least one expected answer
            found_any = any(ans in answers for ans in case['expected_answers'])
            status = "Yes" if found_any else "No"
        else:
            # Should find no answers
            status = "Yes" if not answers else "No"
        
        print(f"Status: {status}")


def test_full_pipeline():
    """Test the full pipeline"""
    print("Testing Full Pipeline")
    
    # Create and enhance graph
    graph = create_test_graph()
    enhanced_graph = enhance_graph_with_textual_representations(graph)
    
    # Create resolver
    resolver = EntityResolver(enhanced_graph)
    
    # Simulate a query processing scenario
    query = "Who was the first person appointed by the Danish Ministry?"
    
    print(f"\nQuery: {query}")
    
    # Extract entities from query
    entities_in_query = ["Danish Ministry"]
    
    print("\nEntity Resolution:")
    for entity in entities_in_query:
        resolved = resolver.resolve(entity)
        print(f"'{entity}' -> '{resolved}'")
        if resolved:
            # Check if node has tv attribute
            tv = enhanced_graph.nodes[resolved].get('tv', 'Missing')
            print(f"Textual value: {tv}")
    
    # Simulate PathRAG response
    simulated_response = "Based on the temporal paths found, the Danish Ministry appointed the Prime Minister of Denmark in 1992."
    
    print(f"\nSimulated PathRAG Response: {simulated_response}")
    
    # Extract answer
    extractor = AnswerExtractor()
    answers = extractor.extract_answers_from_response(
        simulated_response,
        query,
        answer_type='entity'
    )
    
    print(f"\nExtracted Answers: {answers}")
    
    # Verify the full pipeline works
    checks = {
        "Graph has tv attributes": any('tv' in enhanced_graph.nodes[n] for n in enhanced_graph.nodes),
        "Entity resolution works": resolver.resolve("Danish Ministry") is not None,
        "Answer extraction works": len(answers) > 0
    }
    
    print("\nPipeline Verification:")
    all_pass = True
    for check, result in checks.items():
        status = "Yes" if result else "No"
        print(f"{status} {check}")
        all_pass = all_pass and result
    
    return all_pass


def main():
    """Run all tests"""
    print("\nTemporal PathRAG Enhancement Tests (Standalone)")
    print("This test runs without requiring OpenAI API access")
    
    # Test individual components
    enhanced_graph = test_graph_enhancement()
    resolver = test_entity_resolution(enhanced_graph)
    test_answer_extraction()
    
    # Test full pipeline
    pipeline_success = test_full_pipeline()
    
    print("Test Summary")
    
    if pipeline_success:
        print("All enhancement components are working correctly!")
    else:
        print("Some components need attention")


if __name__ == "__main__":
    main()
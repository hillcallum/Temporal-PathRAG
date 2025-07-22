"""
Test script to verify Temporal PathRAG fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.temporal_qa_benchmarks import MultiTQBenchmark
from evaluation.baseline_runners import TemporalPathRAGBaseline
from evaluation.temporal_qa_benchmarks import TemporalQAQuestion
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_specific_questions():
    """Test specific failing questions"""
    
    # Create baseline
    baseline = TemporalPathRAGBaseline()
    
    # Test questions that were failing
    test_questions = [
        {
            'qid': 'test1',
            'question': 'Who was the first person appointed by the Danish Ministry?',
            'answer': ['Person Name'],  # Placeholder
            'answer_type': 'entity',
            'dataset': 'MultiTQ'
        },
        {
            'qid': 'test2', 
            'question': 'What year was the Cabinet of Denmark established?',
            'answer': ['1848'],  # Placeholder
            'answer_type': 'time',
            'dataset': 'MultiTQ'
        },
        {
            'qid': 'test3',
            'question': 'Which organization replaced the Council of Ministers in Denmark?',
            'answer': ['Cabinet'],  # Placeholder
            'answer_type': 'entity',
            'dataset': 'MultiTQ'
        }
    ]
    
    print("Testing Temporal PathRAG with Enhanced Components")
    
    for q_data in test_questions:
        print(f"Testing Question: {q_data['question']}")
        print(f"Expected Answer Type: {q_data['answer_type']}")
        
        # Create question object
        question = TemporalQAQuestion(
            qid=q_data['qid'],
            question=q_data['question'],
            answer=q_data['answer'],
            answer_type=q_data['answer_type']
        )
        
        try:
            # Run prediction
            prediction = baseline.predict(question, q_data['dataset'])
            
            print(f"\nPredicted Answers: {prediction.predicted_answers}")
            print(f"Confidence: {prediction.confidence}")
            print(f"Reasoning Path: {prediction.reasoning_path}")
            
            # Check if we got any answers
            if prediction.predicted_answers:
                print("\nSuccess: Got answers from the system")
            else:
                print("\nFailure: No answers extracted")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.exception("Error during prediction")


def test_entity_resolution():
    """Test entity resolution capabilities"""
    from src.kg.storage.updated_tkg_loader import load_enhanced_dataset, create_enhanced_entity_resolver
    
    print("Testing Entity Resolution")
    
    # Load enhanced graph
    graph = load_enhanced_dataset("MultiTQ", use_cache=True)
    resolver = create_enhanced_entity_resolver(graph)
    
    # Test entity mentions
    test_mentions = [
        "Danish Ministry",
        "Cabinet of Denmark",
        "Council of Ministers",
        "Denmark",
        "European Union",
        "United States"
    ]
    
    for mention in test_mentions:
        resolved = resolver.resolve(mention)
        if resolved:
            info = resolver.get_entity_info(resolved)
            print(f"\n'{mention}' -> '{resolved}'")
            print(f" Variations: {info['variations'][:3]}...")  # Show first 3
            print(f" Connections: {info['total_connections']}")
        else:
            print(f"\n'{mention}' -> Not Found")


def test_answer_extraction():
    """Test improved answer extraction"""
    from src.evaluation.answer_extractor import ImprovedAnswerExtractor
    
    print("\n" + "="*80)
    print("Testing Improved Answer Extraction")
    print("="*80)
    
    extractor = ImprovedAnswerExtractor()
    
    # Test cases
    test_cases = [
        {
            'text': "The answer is Cabinet of Denmark, which was established in 1848.",
            'question': "What replaced the Council of Ministers?",
            'expected_type': 'entity'
        },
        {
            'text': "Based on the evidence, the year was 1990 when this occurred.",
            'question': "When did this happen?",
            'expected_type': 'time'
        },
        {
            'text': "No relevant information found in the knowledge graph.",
            'question': "Who was the leader?",
            'expected_type': 'entity'
        },
        {
            'text': "The Danish Ministry appointed John Smith as the first director.",
            'question': "Who was appointed first?",
            'expected_type': 'entity'
        }
    ]
    
    for case in test_cases:
        print(f"\nText: {case['text']}")
        print(f"Question: {case['question']}")
        
        answers = extractor.extract_answers_from_response(
            case['text'],
            case['question'],
            answer_type=case['expected_type']
        )
        
        print(f"Extracted: {answers}")


def test_graph_enhancement():
    """Test that graph nodes have textual representations"""
    from src.kg.storage.updated_tkg_loader import load_enhanced_dataset
    
    print("\n" + "="*80)
    print("Testing Graph Enhancement")
    print("="*80)
    
    # Load enhanced graph
    graph = load_enhanced_dataset("MultiTQ", use_cache=True)
    
    # Sample some nodes
    sample_nodes = list(graph.nodes)[:10]
    
    print("\nSample Enhanced Nodes:")
    for node_id in sample_nodes:
        node_data = graph.nodes[node_id]
        tv = node_data.get('tv', 'MISSING')
        print(f" {node_id[:50]}... -> tv: {tv[:50]}...")
    
    # Check edges
    sample_edges = list(graph.edges(data=True))[:5]
    
    print("\nSample Enhanced Edges:")
    for u, v, data in sample_edges:
        te = data.get('te', 'MISSING')
        relation = data.get('relation', 'unknown')
        print(f"  {u[:30]}... -> {v[:30]}...")
        print(f"    relation: {relation}, te: {te}")


def main():
    """Run all tests"""
    print("\nStarting Temporal PathRAG Enhancement Tests")
    
    # Test individual components
    test_graph_enhancement()
    test_entity_resolution()
    test_answer_extraction()
    
    # Test full pipeline
    test_specific_questions()

if __name__ == "__main__":
    main()
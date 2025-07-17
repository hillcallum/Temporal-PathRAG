#!/usr/bin/env python3
"""
Test a single question to debug the answer extraction issue
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.baseline_runners import create_baseline_runner
from evaluation.temporal_qa_benchmarks import MultiTQBenchmark

def test_single_question():
    """Test on a single question for debugging"""
    
    print("Testing Single Question Debug")
    
    # Create benchmark
    benchmark = MultiTQBenchmark(split="test")
    
    # Get just the first question
    question = benchmark.questions[0]
    print(f"\nQuestion: {question.question}")
    print(f"Gold answers: {question.answers}")
    print(f"Answer type: {question.answer_type if hasattr(question, 'answer_type') else 'auto'}")
    
    # Create temporal PathRAG baseline
    runner = create_baseline_runner('temporal_pathrag')
    
    # Run prediction
    print("\nRunning prediction")
    pred = runner.predict(question, benchmark.dataset_type.value)
    
    print(f"\nPrediction results:")
    print(f"Predicted answers: {pred.predicted_answers}")
    print(f"Confidence: {pred.confidence}")
    print(f"Metadata: {pred.metadata}")

if __name__ == "__main__":
    test_single_question()
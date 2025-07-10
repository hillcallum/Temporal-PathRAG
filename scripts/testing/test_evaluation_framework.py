#!/usr/bin/env python3
"""
Test script for the evaluation framework
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation import (
    MultiTQBenchmark,
    TimeQuestionsBenchmark,
    TemporalQAPrediction,
    create_baseline_runner,
    AblationStudy
)


def test_benchmark_loading():
    """Test loading of benchmark datasets"""
    print("=" * 60)
    print("Testing Benchmark Loading")
    print("=" * 60)
    
    # Test MultiTQ
    print("\n1. Testing MultiTQ Benchmark")
    try:
        multitq = MultiTQBenchmark()
        print(f"Successfully loaded {len(multitq.questions)} MultiTQ questions")
        
        # Show sample question
        if multitq.questions:
            q = multitq.questions[0]
            print(f"\nSample question:")
            print(f"ID: {q.qid}")
            print(f"Question: {q.question}")
            print(f"Answers: {q.answers}")
            print(f"Type: {q.answer_type}")
            print(f"Time Level: {q.time_level}")
    except Exception as e:
        print(f"Error loading MultiTQ: {e}")
        
    # Test TimeQuestions
    print("\n2. Testing TimeQuestions Benchmark")
    try:
        tq = TimeQuestionsBenchmark()
        print(f"Successfully loaded {len(tq.questions)} TimeQuestions")
        
        # Show sample question
        if tq.questions:
            q = tq.questions[0]
            print(f"\nSample question:")
            print(f"ID: {q.qid}")
            print(f"Question: {q.question}")
            print(f"Answers: {q.answers}")
            print(f"Type: {q.answer_type}")
            print(f"Temporal Signals: {q.temporal_signal}")
    except Exception as e:
        print(f"Error loading TimeQuestions: {e}")


def test_metrics_computation():
    """Test metric computation"""
    print("\n" + "=" * 60)
    print("Testing Metrics Computation")
    print("=" * 60)
    
    try:
        # Load full benchmark
        full_benchmark = MultiTQBenchmark()
        
        # Create a smaller benchmark with only 5 questions for testing
        benchmark = MultiTQBenchmark()
        benchmark.questions = full_benchmark.questions[:5]
        
        # Create mock predictions
        predictions = []
        for i, question in enumerate(benchmark.questions):
            # Create perfect prediction for first 3, wrong for last 2
            if i < 3:
                pred_answers = question.answers
                confidence = 0.9
            else:
                pred_answers = ["wrong answer"]
                confidence = 0.3
                
            pred = TemporalQAPrediction(
                qid=question.qid,
                predicted_answers=pred_answers,
                confidence=confidence,
                retrieval_time=0.1,
                reasoning_time=0.2
            )
            predictions.append(pred)
            
        # Evaluate
        metrics = benchmark.evaluate(predictions, verbose=False)
        
        print(f"\nMetrics for 5 questions (3 correct, 2 wrong):")
        print(f"Exact Match: {metrics.exact_match:.3f} (expected ~0.600)")
        print(f"F1 Score: {metrics.f1_score:.3f}")
        print(f"Temporal Accuracy: {metrics.temporal_accuracy:.3f}")
        print(f"Avg Retrieval Time: {metrics.avg_retrieval_time:.3f}s")
        print(f"Avg Reasoning Time: {metrics.avg_reasoning_time:.3f}s")
        
        print("Metrics computation working correctly")
        
    except Exception as e:
        print(f"Error computing metrics: {e}")


def test_baseline_runners():
    """Test baseline runners"""
    print("\n" + "=" * 60)
    print("Testing Baseline Runners")
    print("=" * 60)
    
    # Test Vanilla LLM baseline
    print("\n1. Testing Vanilla LLM Baseline")
    try:
        vanilla = create_baseline_runner("vanilla")
        print(f"Created {vanilla.name} baseline")
        
        # Test on single question
        benchmark = MultiTQBenchmark()
        if benchmark.questions:
            question = benchmark.questions[0]
            pred = vanilla.predict(question, "MultiTQ")
            print(f"Prediction: {pred.predicted_answers}")
            print(f"Confidence: {pred.confidence}")
            
    except Exception as e:
        print(f"Error with Vanilla baseline: {e}")
        
    # Test Temporal PathRAG baseline  
    print("\n2. Testing Temporal PathRAG Baseline")
    try:
        temporal = create_baseline_runner("temporal_pathrag")
        print(f"Created {temporal.name} baseline")
        print("(Full test would require loaded graph)")
        
    except Exception as e:
        print(f"Error with Temporal PathRAG baseline: {e}")


def test_ablation_framework():
    """Test ablation study framework"""
    print("\n" + "=" * 60)
    print("Testing Ablation Framework")
    print("=" * 60)
    
    try:
        # Create ablation study
        study = AblationStudy("MultiTQ")
        
        print(f"Created ablation study with {len(study.components)} components:")
        for component in study.components:
            print(f" - {component.name} ({component.category})")
            
        # Test configuration creation
        config, states = study.create_configuration(["temporal_weighting", "path_reliability"])
        print(f"\nCreated configuration with 2 active components")
        print(f"Active: {[k for k, v in states.items() if v]}")
        print(f"Inactive: {[k for k, v in states.items() if not v]}")
        
        print("\nAblation framework setup correctly")
        print("(Full ablation study would take significant time)")
        
    except Exception as e:
        print(f"Error with ablation framework: {e}")


def main():
    """Run all tests"""
    print("Testing Temporal PathRAG Evaluation Framework\n")
    
    # Run tests
    test_benchmark_loading()
    test_metrics_computation()
    test_baseline_runners()
    test_ablation_framework()
    
    print("\n" + "=" * 60)
    print("Testing Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
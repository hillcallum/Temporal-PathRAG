"""
Baseline system runners for comparison with Temporal PathRAG.
"""

import os
import sys
import time
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import subprocess
from dataclasses import dataclass

from .temporal_qa_benchmarks import (
    TemporalQAQuestion, 
    TemporalQAPrediction,
    TemporalQABenchmark,
    MultiTQBenchmark,
    TimeQuestionsBenchmark
)
from src.utils.config import get_config
from src.llm.llm_manager import LLMManager


class BaselineRunner(ABC):
    """Abstract base class for baseline system runners"""
    
    def __init__(self, name: str):
        """
        Initialise baseline runner
        """
        self.name = name
        self.config = get_config()
        
    @abstractmethod
    def predict(self, question: TemporalQAQuestion, 
                dataset_name: str) -> TemporalQAPrediction:
        """
        Generate prediction for a single question
        """
        pass
    
    def run_benchmark(self, benchmark: TemporalQABenchmark, 
                     max_questions: Optional[int] = None,
                     verbose: bool = True) -> List[TemporalQAPrediction]:
        """
        Run baseline on entire benchmark
        """
        predictions = []
        questions = benchmark.questions[:max_questions] if max_questions else benchmark.questions
        
        for i, question in enumerate(questions):
            if verbose and i % 10 == 0:
                print(f"Processing question {i+1}/{len(questions)}")
                
            try:
                pred = self.predict(question, benchmark.dataset_type.value)
                predictions.append(pred)
            except Exception as e:
                print(f"Error processing question {question.qid}: {e}")
                # Add empty prediction
                predictions.append(TemporalQAPrediction(
                    qid=question.qid,
                    predicted_answers=[],
                    confidence=0.0,
                    metadata={'error': str(e)}
                ))
                
        return predictions


class VanillaLLMBaseline(BaselineRunner):
    """Vanilla LLM baseline - no retrieval, just direct QA"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialise vanilla LLM baseline
        """
        super().__init__(f"VanillaLLM-{model_name}")
        self.llm_manager = LLMManager(model_preference=[model_name])
        
    def predict(self, question: TemporalQAQuestion, 
                dataset_name: str) -> TemporalQAPrediction:
        """Generate prediction using vanilla LLM"""
        start_time = time.time()
        
        # Create prompt based on question type
        if question.answer_type in ['time', 'value']:
            prompt = f"""Answer the following temporal question with a date or time period.
            Question: {question.question}
            Answer with just the date/time, no explanation."""
        else:
            prompt = f"""Answer the following question with entity names.
            Question: {question.question}
            Answer with just the entity name(s), no explanation."""
            
        # Get LLM response
        response = self.llm_manager.generate(prompt, max_tokens=50)
        
        # Parse response into answer list
        if response:
            # Simple parsing - split by common delimiters
            answers = [ans.strip() for ans in response.replace('\n', ',').split(',')]
            answers = [ans for ans in answers if ans]  # Remove empty strings
        else:
            answers = []
            
        reasoning_time = time.time() - start_time
        
        return TemporalQAPrediction(
            qid=question.qid,
            predicted_answers=answers,
            confidence=0.7 if answers else 0.0,
            reasoning_time=reasoning_time,
            metadata={'baseline': 'vanilla_llm', 'model': self.llm_manager.get_current_model()}
        )


class DirectLLMBaseline(BaselineRunner):
    """Direct LLM baseline - different models without RAG"""
    
    def __init__(self, model_name: str = "llama2-7b"):
        """
        Initialise direct LLM baseline.
        """
        super().__init__(f"DirectLLM-{model_name}")
        self.model_name = model_name
        self.llm_manager = LLMManager(model_preference=[model_name])
        
    def predict(self, question: TemporalQAQuestion, 
                dataset_name: str) -> TemporalQAPrediction:
        """Generate prediction using direct LLM without any retrieval"""
        start_time = time.time()
        
        # Simple direct prompt without any RAG context
        prompt = f"Answer this question concisely: {question.question}"
        
        # Get LLM response
        response = self.llm_manager.generate(prompt, max_tokens=50)
        
        # Parse response
        if response:
            answers = [ans.strip() for ans in response.replace('\n', ',').split(',')]
            answers = [ans for ans in answers if ans]
        else:
            answers = []
            
        reasoning_time = time.time() - start_time
        
        return TemporalQAPrediction(
            qid=question.qid,
            predicted_answers=answers,
            confidence=0.5 if answers else 0.0,
            reasoning_time=reasoning_time,
            metadata={'baseline': 'direct_llm', 'model': self.model_name}
        )


class VanillaRAGBaseline(BaselineRunner):
    """Standard RAG baseline with simple retrieval"""
    
    def __init__(self, retriever_type: str = "dense"):
        """
        Initialise vanilla RAG baseline.
        """
        super().__init__(f"VanillaRAG-{retriever_type}")
        self.retriever_type = retriever_type
        self.llm_manager = LLMManager()
        
    def predict(self, question: TemporalQAQuestion, 
                dataset_name: str) -> TemporalQAPrediction:
        """Generate prediction using vanilla RAG"""
        start_retrieval = time.time()
        
        # Simple retrieval simulation (in future, will use actual retriever)
        # For now, we'll simulate with a mock retrieval
        retrieved_facts = self._retrieve_facts(question.question, dataset_name)
        retrieval_time = time.time() - start_retrieval
        
        # Generate answer with retrieved context
        start_reasoning = time.time()
        context = "\n".join(retrieved_facts[:5])  # Use top 5 facts
        prompt = f"""Given the following context, answer the question.
        
        Context:
        {context}

        Question: {question.question}
        Answer:"""
        
        response = self.llm_manager.generate(prompt, max_tokens=50)
        reasoning_time = time.time() - start_reasoning
        
        # Parse response
        if response:
            answers = [ans.strip() for ans in response.replace('\n', ',').split(',')]
            answers = [ans for ans in answers if ans]
        else:
            answers = []
            
        return TemporalQAPrediction(
            qid=question.qid,
            predicted_answers=answers,
            confidence=0.75 if answers else 0.0,
            retrieval_time=retrieval_time,
            reasoning_time=reasoning_time,
            metadata={'baseline': 'vanilla_rag', 'retriever': self.retriever_type}
        )
        
    def _retrieve_facts(self, query: str, dataset_name: str) -> List[str]:
        """Mock retrieval function - in future, will use actual retriever"""
        # Simulate retrieval with dummy facts
        return [
            f"Fact 1 related to: {query[:30]}",
            f"Fact 2 about temporal information",
            f"Fact 3 from {dataset_name} dataset",
            f"Fact 4 with entity information",
            f"Fact 5 containing dates and times"
        ]


class HyDEBaseline(BaselineRunner):
    """Hypothetical Document Embeddings (HyDE) baseline"""
    
    def __init__(self):
        """Initialise HyDE baseline"""
        super().__init__("HyDE")
        self.llm_manager = LLMManager()
        
    def predict(self, question: TemporalQAQuestion, 
                dataset_name: str) -> TemporalQAPrediction:
        """Generate prediction using HyDE approach"""
        start_time = time.time()
        
        # Step 1: Generate hypothetical answer
        hypo_prompt = f"""Write a detailed answer to this question as if you had perfect knowledge:
        {question.question}

        Detailed answer:"""
        
        hypothetical_answer = self.llm_manager.generate(hypo_prompt, max_tokens=150)
        
        # Step 2: Use hypothetical answer for retrieval (simulated)
        start_retrieval = time.time()
        retrieved_facts = self.retrieve_with_hypothesis(
            question.question, hypothetical_answer, dataset_name
        )
        retrieval_time = time.time() - start_retrieval
        
        # Step 3: Generate final answer with retrieved facts
        context = "\n".join(retrieved_facts[:5])
        final_prompt = f"""Based on the following facts, answer the question accurately.
        
        Facts:
        {context}

        Question: {question.question}
        Answer:"""
        
        response = self.llm_manager.generate(final_prompt, max_tokens=50)
        total_time = time.time() - start_time
        reasoning_time = total_time - retrieval_time
        
        # Parse response
        if response:
            answers = [ans.strip() for ans in response.replace('\n', ',').split(',')]
            answers = [ans for ans in answers if ans]
        else:
            answers = []
            
        return TemporalQAPrediction(
            qid=question.qid,
            predicted_answers=answers,
            confidence=0.8 if answers else 0.0,
            retrieval_time=retrieval_time,
            reasoning_time=reasoning_time,
            metadata={
                'baseline': 'hyde',
                'hypothetical_answer_length': len(hypothetical_answer) if hypothetical_answer else 0
            }
        )
        
    def retrieve_with_hypothesis(self, query: str, hypothesis: str, 
                                 dataset_name: str) -> List[str]:
        """Mock retrieval using hypothetical answer"""
        # In future, will use hypothesis for enhanced retrieval
        return [
            f"Fact retrieved using hypothesis about: {query[:30]}",
            f"Temporal fact from hypothesis matching",
            f"Entity fact aligned with hypothetical answer",
            f"Date/time information from {dataset_name}",
            f"Additional context from hypothesis-based retrieval"
        ]


class PathRAGBaseline(BaselineRunner):
    """PathRAG baseline adapter"""
    
    def __init__(self, pathrag_dir: Optional[Path] = None):
        """
        Initialise PathRAG baseline
        """
        super().__init__("PathRAG")
        
        # Use PathRAG from home directory if not specified
        if pathrag_dir is None:
            home = Path.home()
            pathrag_dir = home / "PathRAG"
            
        self.pathrag_dir = pathrag_dir
        if not self.pathrag_dir.exists():
            raise FileNotFoundError(f"PathRAG directory not found: {self.pathrag_dir}")
            
        # Add PathRAG to Python path
        sys.path.insert(0, str(self.pathrag_dir))
        
    def predict(self, question: TemporalQAQuestion, 
                dataset_name: str) -> TemporalQAPrediction:
        """Generate prediction using PathRAG"""
        start_retrieval = time.time()
        
        try:
            # Import PathRAG components - will fix 
            from pathrag import PathRAG  # type: ignore
            
            # Initialise PathRAG for the dataset
            pathrag = PathRAG(dataset=dataset_name.lower())
            
            # Retrieve paths
            paths = pathrag.retrieve(question.question, top_k=10)
            retrieval_time = time.time() - start_retrieval
            
            # Generate answer using retrieved paths
            start_reasoning = time.time()
            answer = pathrag.answer(question.question, paths)
            reasoning_time = time.time() - start_reasoning
            
            # Parse answer
            if isinstance(answer, str):
                answers = [answer]
            elif isinstance(answer, list):
                answers = answer
            else:
                answers = []
                
        except ImportError:
            # Fallback if PathRAG import fails
            print(f"Warning: Could not import PathRAG, using mock implementation")
            retrieval_time = 0.1
            reasoning_time = 0.1
            answers = ["PathRAG baseline not available"]
            
        return TemporalQAPrediction(
            qid=question.qid,
            predicted_answers=answers,
            confidence=0.8 if answers else 0.0,
            retrieval_time=retrieval_time,
            reasoning_time=reasoning_time,
            metadata={'baseline': 'pathrag'}
        )


class TimeR4Baseline(BaselineRunner):
    """TimeR4 baseline adapter"""
    
    def __init__(self, timer4_dir: Optional[Path] = None):
        """
        Initialise TimeR4 baseline
        """
        super().__init__("TimeR4")
        
        # Use TimeR4 from home directory if not specified
        if timer4_dir is None:
            home = Path.home()
            timer4_dir = home / "TimeR4"
            
        self.timer4_dir = timer4_dir
        if not self.timer4_dir.exists():
            raise FileNotFoundError(f"TimeR4 directory not found: {self.timer4_dir}")
            
        # Add TimeR4 to Python path
        sys.path.insert(0, str(self.timer4_dir))
        
    def predict(self, question: TemporalQAQuestion, 
                dataset_name: str) -> TemporalQAPrediction:
        """Generate prediction using TimeR4"""
        start_time = time.time()
        
        try:
            # Import TimeR4 components - will fix
            from timer4 import TimeR4  # type: ignore
            
            # Initialise TimeR4
            timer4 = TimeR4(dataset=dataset_name.lower())
            
            # Retrieve-Rewrite phase
            rewritten_question = timer4.retrieve_rewrite(question.question)
            
            # Retrieve-Rerank phase
            facts = timer4.retrieve_rerank(rewritten_question)
            retrieval_time = time.time() - start_time
            
            # Generate answer
            start_reasoning = time.time()
            answer = timer4.answer(rewritten_question, facts)
            reasoning_time = time.time() - start_reasoning
            
            # Parse answer
            if isinstance(answer, str):
                answers = [answer]
            elif isinstance(answer, list):
                answers = answer
            else:
                answers = []
                
        except ImportError:
            # Fallback if TimeR4 import fails
            print(f"Warning: Could not import TimeR4, using mock implementation")
            retrieval_time = 0.1
            reasoning_time = 0.1
            answers = ["TimeR4 baseline not available"]
            
        return TemporalQAPrediction(
            qid=question.qid,
            predicted_answers=answers,
            confidence=0.85 if answers else 0.0,
            retrieval_time=retrieval_time,
            reasoning_time=reasoning_time,
            metadata={'baseline': 'timer4'}
        )


class TemporalPathRAGBaseline(BaselineRunner):
    """Our Temporal PathRAG system as a baseline for ablation studies"""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialise Temporal PathRAG baseline
        """
        super().__init__("TemporalPathRAG")
        self.config_override = config_override or {}
        
        # Import our system components
        from src.kg.tkg_query_engine import TKGQueryEngine
        from src.kg.temporal_iterative_reasoner import TemporalIterativeReasoner
        
        self.query_engine = TKGQueryEngine()
        self.reasoner = TemporalIterativeReasoner(
            query_engine=self.query_engine,
            **self.config_override
        )
        
    def predict(self, question: TemporalQAQuestion, 
                dataset_name: str) -> TemporalQAPrediction:
        """Generate prediction using Temporal PathRAG"""
        start_time = time.time()
        
        # Load the appropriate dataset if not already loaded
        if not hasattr(self, f'_{dataset_name}_loaded'):
            from src.utils.dataset_loader import load_dataset
            graph = load_dataset(dataset_name)
            self.query_engine.set_graph(graph)
            setattr(self, f'_{dataset_name}_loaded', True)
            
        # Run temporal iterative reasoning
        result = self.reasoner.answer_question(
            question.question,
            max_iterations=5
        )
        
        total_time = time.time() - start_time
        
        return TemporalQAPrediction(
            qid=question.qid,
            predicted_answers=result.get('answers', []),
            confidence=result.get('confidence', 0.0),
            reasoning_path=result.get('reasoning_path', []),
            retrieval_time=result.get('retrieval_time', 0.0),
            reasoning_time=total_time - result.get('retrieval_time', 0.0),
            metadata={
                'baseline': 'temporal_pathrag',
                'iterations': result.get('iterations', 0),
                'paths_retrieved': result.get('paths_retrieved', 0)
            }
        )


def create_baseline_runner(baseline_name: str, **kwargs) -> BaselineRunner:
    """
    Factory function to create baseline runners
    """
    baseline_map = {
        'vanilla': VanillaLLMBaseline,
        'vanilla_llm': VanillaLLMBaseline,  # alias
        'direct_llm': DirectLLMBaseline,
        'llama2': lambda: DirectLLMBaseline('llama2-7b'),
        'llama2-13b': lambda: DirectLLMBaseline('llama2-13b'),
        'gpt3.5': lambda: DirectLLMBaseline('gpt-3.5-turbo'),
        'gpt4': lambda: DirectLLMBaseline('gpt-4'),
        'vanilla_rag': VanillaRAGBaseline,
        'hyde': HyDEBaseline,
        'pathrag': PathRAGBaseline,
        'timer4': TimeR4Baseline,
        'temporal_pathrag': TemporalPathRAGBaseline
    }
    
    if baseline_name.lower() not in baseline_map:
        raise ValueError(f"Unknown baseline: {baseline_name}. "
                        f"Available: {list(baseline_map.keys())}")
                        
    baseline_class = baseline_map[baseline_name.lower()]
    if callable(baseline_class) and not isinstance(baseline_class, type):
        # Handle lambda functions
        return baseline_class()
    else:
        return baseline_class(**kwargs)


def run_baseline_comparison(dataset_name: str, 
                           baselines: List[str],
                           max_questions: Optional[int] = None,
                           output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run multiple baselines and compare results
    """
    # Create benchmark
    if dataset_name.lower() == "multitq":
        benchmark = MultiTQBenchmark()
    elif dataset_name.lower() == "timequestions":
        benchmark = TimeQuestionsBenchmark()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Set output directory
    if output_dir is None:
        config = get_config()
        output_dir = config.get_output_dir("baseline_comparison")
        
    results = {}
    
    for baseline_name in baselines:
        print(f"\n{'='*60}")
        print(f"Running {baseline_name} baseline on {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Create baseline runner
            runner = create_baseline_runner(baseline_name)
            
            # Run predictions
            predictions = runner.run_benchmark(
                benchmark, 
                max_questions=max_questions,
                verbose=True
            )
            
            # Evaluate
            metrics = benchmark.evaluate(predictions, verbose=True)
            
            # Save results
            result_path = output_dir / f"{baseline_name}_{dataset_name}_results.json"
            benchmark.save_results(metrics, result_path)
            
            results[baseline_name] = {
                'metrics': metrics,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"Error running {baseline_name}: {e}")
            results[baseline_name] = {'error': str(e)}
            
    # Save comparison summary
    summary_path = output_dir / f"baseline_comparison_{dataset_name}_summary.json"
    summary = {
        'dataset': dataset_name,
        'baselines': baselines,
        'num_questions': len(benchmark.questions[:max_questions] if max_questions else benchmark.questions),
        'results': {
            name: {
                'exact_match': res['metrics'].exact_match if 'metrics' in res else 0.0,
                'f1_score': res['metrics'].f1_score if 'metrics' in res else 0.0,
                'temporal_accuracy': res['metrics'].temporal_accuracy if 'metrics' in res else 0.0,
                'avg_time': (res['metrics'].avg_retrieval_time + res['metrics'].avg_reasoning_time) if 'metrics' in res else 0.0,
                'error': res.get('error')
            }
            for name, res in results.items()
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nComparison summary saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Running baseline comparison")
    
    # Test on a small subset
    results = run_baseline_comparison(
        dataset_name="MultiTQ",
        baselines=["vanilla", "temporal_pathrag"],
        max_questions=10  # Just test on 10 questions
    )
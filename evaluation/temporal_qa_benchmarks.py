"""
Temporal Question Answering Benchmarks for Temporal PathRAG
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from collections import defaultdict
import numpy as np
from datetime import datetime
import re

from src.utils.config import get_config, DatasetType
from src.utils.dataset_loader import load_dataset


@dataclass
class TemporalQAQuestion:
    """Unified representation of a temporal QA question"""
    qid: str
    question: str
    answers: List[str]
    answer_type: str  # 'entity', 'time', 'value'
    temporal_signal: Optional[List[str]] = None
    temporal_question_type: Optional[List[str]] = None
    time_level: Optional[str] = None  # 'day', 'year', etc.
    qtype: Optional[str] = None  # Question type classification
    qlabel: Optional[str] = None  # 'Single' or 'Multiple'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalQAPrediction:
    """Prediction for a temporal QA question"""
    qid: str
    predicted_answers: List[str]
    confidence: float = 1.0
    reasoning_path: Optional[List[Dict[str, Any]]] = None
    retrieval_time: float = 0.0
    reasoning_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalMetrics:
    """Comprehensive metrics for temporal QA evaluation"""
    exact_match: float = 0.0
    f1_score: float = 0.0
    temporal_accuracy: float = 0.0
    entity_accuracy: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    hits_at_1: float = 0.0
    hits_at_3: float = 0.0
    hits_at_10: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_reasoning_time: float = 0.0
    total_questions: int = 0
    correct_predictions: int = 0
    
    # Breakdown by question type
    metrics_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Breakdown by temporal granularity
    metrics_by_granularity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Error analysis
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class TemporalQABenchmark(ABC):
    """Abstract base class for temporal QA benchmarks"""
    
    def __init__(self, dataset_type: DatasetType, split: str = "test"):
        """
        Initialise benchmark
        """
        self.dataset_type = dataset_type
        self.split = split
        self.config = get_config()
        self.dataset_config = self.config.get_dataset_config(dataset_type)
        self.questions: List[TemporalQAQuestion] = []
        self.load_questions()
        
    @abstractmethod
    def load_questions(self) -> None:
        """Load questions from the dataset"""
        pass
    
    @abstractmethod
    def normalise_answer(self, answer: str) -> str:
        """Normalise answer for comparison"""
        pass
    
    def compute_exact_match(self, predicted: List[str], gold: List[str]) -> float:
        """Compute exact match scores"""
        pred_set = {self.normalise_answer(p) for p in predicted}
        gold_set = {self.normalise_answer(g) for g in gold}
        return float(pred_set == gold_set)
    
    def compute_f1_score(self, predicted: List[str], gold: List[str]) -> float:
        """Compute F1 score between predicted and gold answers"""
        pred_set = {self.normalise_answer(p) for p in predicted}
        gold_set = {self.normalise_answer(g) for g in gold}
        
        if not pred_set and not gold_set:
            return 1.0
        if not pred_set or not gold_set:
            return 0.0
            
        precision = len(pred_set & gold_set) / len(pred_set)
        recall = len(pred_set & gold_set) / len(gold_set)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def compute_temporal_accuracy(self, predicted: List[str], gold: List[str], 
                                  question: TemporalQAQuestion) -> float:
        """
        Compute temporal-specific accuracy based on time granularity
        """
        if question.answer_type not in ['time', 'value']:
            return self.compute_exact_match(predicted, gold)
            
        # For temporal answers, allow some flexibility based on granularity
        pred_dates = self.extract_dates(predicted)
        gold_dates = self.extract_dates(gold)
        
        if not pred_dates or not gold_dates:
            return 0.0
            
        # Check if any predicted date matches gold within tolerance
        for pred_date in pred_dates:
            for gold_date in gold_dates:
                if self.dates_match(pred_date, gold_date, question.time_level):
                    return 1.0
        return 0.0
    
    def extract_dates(self, answers: List[str]) -> List[datetime]:
        """Extract datetime objects from answer strings"""
        dates = []
        for answer in answers:
            # Try various date formats
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',  # ISO format
                r'(\d{4}-\d{1,2}-\d{1,2})',  # Flexible ISO
                r'(\d{1,2}/\d{1,2}/\d{4})',  # US format
                r'(\d{4})',  # Year only
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, answer)
                for match in matches:
                    try:
                        if len(match) == 4:  # Year only
                            dates.append(datetime(int(match), 1, 1))
                        else:
                            # Try parsing with various formats
                            for fmt in ['%Y-%m-%d', '%Y-%m-%d', '%m/%d/%Y']:
                                try:
                                    dates.append(datetime.strptime(match, fmt))
                                    break
                                except ValueError:
                                    continue
                    except (ValueError, AttributeError):
                        continue
        return dates
    
    def dates_match(self, date1: datetime, date2: datetime, 
                     granularity: Optional[str]) -> bool:
        """Check if two dates match given the granularity level"""
        if granularity == 'year':
            return date1.year == date2.year
        elif granularity == 'month':
            return date1.year == date2.year and date1.month == date2.month
        elif granularity == 'day':
            return (date1.year == date2.year and 
                    date1.month == date2.month and 
                    date1.day == date2.day)
        else:
            # Default to exact match
            return date1 == date2
    
    def evaluate(self, predictions: List[TemporalQAPrediction], 
                 verbose: bool = True) -> TemporalMetrics:
        """
        Evaluate predictions against gold answers
        """
        metrics = TemporalMetrics(total_questions=len(self.questions))
        
        # Create prediction lookup
        pred_dict = {p.qid: p for p in predictions}
        
        # Metrics accumulators
        exact_matches = []
        f1_scores = []
        temporal_accuracies = []
        entity_accuracies = []
        retrieval_times = []
        reasoning_times = []
        
        # Type-specific accumulators
        type_metrics = defaultdict(lambda: {'count': 0, 'exact_match': 0, 'f1': 0})
        granularity_metrics = defaultdict(lambda: {'count': 0, 'accuracy': 0})
        
        for question in self.questions:
            if question.qid not in pred_dict:
                # Missing prediction
                metrics.error_types['missing_prediction'] += 1
                exact_matches.append(0)
                f1_scores.append(0)
                temporal_accuracies.append(0)
                if question.answer_type == 'entity':
                    entity_accuracies.append(0)
                continue
                
            pred = pred_dict[question.qid]
            
            # Compute metrics
            em = self.compute_exact_match(pred.predicted_answers, question.answers)
            f1 = self.compute_f1_score(pred.predicted_answers, question.answers)
            temp_acc = self.compute_temporal_accuracy(
                pred.predicted_answers, question.answers, question
            )
            
            exact_matches.append(em)
            f1_scores.append(f1)
            temporal_accuracies.append(temp_acc)
            
            if question.answer_type == 'entity':
                entity_accuracies.append(em)
                
            # Track times
            retrieval_times.append(pred.retrieval_time)
            reasoning_times.append(pred.reasoning_time)
            
            # Update type-specific metrics
            if question.qtype:
                type_metrics[question.qtype]['count'] += 1
                type_metrics[question.qtype]['exact_match'] += em
                type_metrics[question.qtype]['f1'] += f1
                
            # Update granularity-specific metrics  
            if question.time_level:
                granularity_metrics[question.time_level]['count'] += 1
                granularity_metrics[question.time_level]['accuracy'] += temp_acc
                
            # Error analysis
            if em == 0:
                if not pred.predicted_answers:
                    metrics.error_types['no_answer'] += 1
                elif question.answer_type in ['time', 'value']:
                    metrics.error_types['temporal_error'] += 1
                else:
                    metrics.error_types['entity_error'] += 1
                    
        # Compute aggregate metrics
        metrics.exact_match = np.mean(exact_matches) if exact_matches else 0.0
        metrics.f1_score = np.mean(f1_scores) if f1_scores else 0.0
        metrics.temporal_accuracy = np.mean(temporal_accuracies) if temporal_accuracies else 0.0
        metrics.entity_accuracy = np.mean(entity_accuracies) if entity_accuracies else 0.0
        metrics.correct_predictions = sum(exact_matches)
        
        # Compute MRR and Hits@K (simplified for now)
        metrics.hits_at_1 = metrics.exact_match
        metrics.hits_at_3 = metrics.f1_score  # Approximation
        metrics.hits_at_10 = min(metrics.f1_score * 1.2, 1.0)  # Approximation
        metrics.mrr = metrics.exact_match  # Simplified
        
        # Average times
        metrics.avg_retrieval_time = np.mean(retrieval_times) if retrieval_times else 0.0
        metrics.avg_reasoning_time = np.mean(reasoning_times) if reasoning_times else 0.0
        
        # Compute type-specific metrics
        for qtype, counts in type_metrics.items():
            if counts['count'] > 0:
                metrics.metrics_by_type[qtype] = {
                    'exact_match': counts['exact_match'] / counts['count'],
                    'f1_score': counts['f1'] / counts['count'],
                    'count': counts['count']
                }
                
        # Compute granularity-specific metrics
        for gran, counts in granularity_metrics.items():
            if counts['count'] > 0:
                metrics.metrics_by_granularity[gran] = {
                    'accuracy': counts['accuracy'] / counts['count'],
                    'count': counts['count']
                }
        
        if verbose:
            self.print_results(metrics)
            
        return metrics
    
    def print_results(self, metrics: TemporalMetrics) -> None:
        """Print detailed evaluation results"""
        print(f"\n{'='*60}")
        print(f"Temporal QA Evaluation Results - {self.dataset_type.value}")
        print(f"{'='*60}")
        print(f"Total Questions: {metrics.total_questions}")
        print(f"Correct Predictions: {metrics.correct_predictions}")
        print(f"\nOverall Metrics:")
        print(f"Exact Match: {metrics.exact_match:.3f}")
        print(f"F1 Score: {metrics.f1_score:.3f}")
        print(f"Temporal Accuracy: {metrics.temporal_accuracy:.3f}")
        print(f"Entity Accuracy: {metrics.entity_accuracy:.3f}")
        print(f"\nRanking Metrics:")
        print(f"MRR: {metrics.mrr:.3f}")
        print(f"Hits@1: {metrics.hits_at_1:.3f}")
        print(f"Hits@3: {metrics.hits_at_3:.3f}")
        print(f"Hits@10: {metrics.hits_at_10:.3f}")
        print(f"\nEfficiency Metrics:")
        print(f"Avg Retrieval Time: {metrics.avg_retrieval_time:.3f}s")
        print(f"Avg Reasoning Time: {metrics.avg_reasoning_time:.3f}s")
        
        if metrics.metrics_by_type:
            print(f"\nMetrics by Question Type:")
            for qtype, type_metrics in metrics.metrics_by_type.items():
                print(f"{qtype}:")
                print(f"EM: {type_metrics['exact_match']:.3f}, "
                      f"F1: {type_metrics['f1_score']:.3f} "
                      f"(n={type_metrics['count']})")
                      
        if metrics.metrics_by_granularity:
            print(f"\nMetrics by Temporal Granularity:")
            for gran, gran_metrics in metrics.metrics_by_granularity.items():
                print(f"{gran}: {gran_metrics['accuracy']:.3f} "
                      f"(n={gran_metrics['count']})")
                      
        if metrics.error_types:
            print(f"\nError Analysis:")
            for error_type, count in metrics.error_types.items():
                print(f"{error_type}: {count}")
        print(f"{'='*60}\n")
        
    def save_results(self, metrics: TemporalMetrics, output_path: Path) -> None:
        """Save evaluation results to JSON file"""
        results = {
            'dataset': self.dataset_type.value,
            'split': self.split,
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': {
                'exact_match': metrics.exact_match,
                'f1_score': metrics.f1_score,
                'temporal_accuracy': metrics.temporal_accuracy,
                'entity_accuracy': metrics.entity_accuracy,
                'mrr': metrics.mrr,
                'hits_at_1': metrics.hits_at_1,
                'hits_at_3': metrics.hits_at_3,
                'hits_at_10': metrics.hits_at_10,
                'total_questions': metrics.total_questions,
                'correct_predictions': metrics.correct_predictions,
            },
            'efficiency_metrics': {
                'avg_retrieval_time': metrics.avg_retrieval_time,
                'avg_reasoning_time': metrics.avg_reasoning_time,
            },
            'breakdown': {
                'by_type': metrics.metrics_by_type,
                'by_granularity': metrics.metrics_by_granularity,
            },
            'error_analysis': dict(metrics.error_types)
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")


class MultiTQBenchmark(TemporalQABenchmark):
    """Benchmark for MultiTQ dataset"""
    
    def __init__(self, split: str = "test"):
        super().__init__(DatasetType.MULTITQ, split)
        
    def load_questions(self) -> None:
        """Load MultiTQ questions"""
        questions_path = self.dataset_config.path / self.dataset_config.questions_dir / f"{self.split}.json"
        
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
            
        with open(questions_path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            question = TemporalQAQuestion(
                qid=str(item['quid']),
                question=item['question'],
                answers=item['answers'] if isinstance(item['answers'], list) else [item['answers']],
                answer_type=item['answer_type'],
                time_level=item.get('time_level'),
                qtype=item.get('qtype'),
                qlabel=item.get('qlabel'),
                metadata={'original': item}
            )
            self.questions.append(question)
            
        print(f"Loaded {len(self.questions)} questions from MultiTQ {self.split} split")
        
    def normalise_answer(self, answer: str) -> str:
        """Normalise MultiTQ answer"""
        # Convert to lowercase and strip whitespace
        normalised = answer.lower().strip()
        
        # Remove articles
        normalised = re.sub(r'\b(the|a|an)\b', '', normalised).strip()
        
        # Normalise date formats
        # Convert various date formats to ISO format
        date_patterns = [
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),  # MM/DD/YYYY to YYYY-MM-DD
            (r'(\d{1,2})-(\d{1,2})-(\d{4})', r'\3-\1-\2'),  # MM-DD-YYYY to YYYY-MM-DD
        ]
        
        for pattern, replacement in date_patterns:
            normalised = re.sub(pattern, replacement, normalised)
            
        return normalised


class TimeQuestionsBenchmark(TemporalQABenchmark):
    """Benchmark for TimeQuestions dataset"""
    
    def __init__(self, split: str = "test"):
        super().__init__(DatasetType.TIMEQUESTIONS, split)
        
    def load_questions(self) -> None:
        """Load TimeQuestions questions"""
        questions_path = self.dataset_config.path / self.dataset_config.questions_dir / f"{self.split}.json"
        
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")
            
        with open(questions_path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            # Extract answers based on type
            answers = []
            answer_type = 'entity'  # default
            
            if 'Answer' in item:
                for ans in item['Answer']:
                    if ans['AnswerType'] == 'Value':
                        # Temporal answer
                        answer_type = 'value'
                        answers.append(ans['AnswerArgument'])
                    else:
                        # Entity answer
                        answers.append(ans.get('WikidataLabel', ans.get('WikidataQid', '')))
                        
            question = TemporalQAQuestion(
                qid=str(item['Id']),
                question=item['Question'],
                answers=answers,
                answer_type=answer_type,
                temporal_signal=item.get('Temporal signal', []),
                temporal_question_type=item.get('Temporal question type', []),
                metadata={
                    'data_source': item.get('Data source'),
                    'dataset': item.get('Data set'),
                    'original': item
                }
            )
            self.questions.append(question)
            
        print(f"Loaded {len(self.questions)} questions from TimeQuestions {self.split} split")
        
    def normalise_answer(self, answer: str) -> str:
        """Normalise TimeQuestions answer."""
        # Convert to lowercase and strip whitespace
        normalised = answer.lower().strip()
        
        # Handle ISO 8601 timestamps
        iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?'
        iso_match = re.search(iso_pattern, answer)
        if iso_match:
            # Extract just the date part
            normalised = iso_match.group()[:10]
            
        # Remove articles and common prefixes
        normalised = re.sub(r'\b(the|a|an)\b', '', normalised).strip()
        
        # Handle Wikidata IDs
        if normalised.startswith('q') and normalised[1:].isdigit():
            # Keep Wikidata IDs as is
            return normalised.upper()
            
        return normalised


def create_benchmark(dataset_name: str, split: str = "test") -> TemporalQABenchmark:
    """
    Factory function to create appropriate benchmark instance
    """
    if dataset_name.lower() == "multitq":
        return MultiTQBenchmark(split)
    elif dataset_name.lower() == "timequestions":
        return TimeQuestionsBenchmark(split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Example usage
    print("Testing MultiTQ Benchmark")
    multitq_bench = MultiTQBenchmark()
    print(f"Loaded {len(multitq_bench.questions)} questions")
    
    print("\nTesting TimeQuestions Benchmark")
    tq_bench = TimeQuestionsBenchmark()
    print(f"Loaded {len(tq_bench.questions)} questions")
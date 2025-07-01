#!/usr/bin/env python3
"""
Generate prompts with GPU-accelerated embeddings
"""

import json
import random
import time
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Import embedding libraries
try:
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        DEVICE = "cuda"
    else:
        print("Using CPU processing")
        DEVICE = "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    DEVICE = "cpu"
    print("Using CPU processing (PyTorch not available)")

try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print(f"Sentence transformers available on {DEVICE}")
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"Sentence transformers not available: {e}")
    print("Falling back to string-based matching")

class EmbeddingGenerator:
    """Prompt generator with semantic embeddings"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_path.split('/')[-1]
        self.device = DEVICE
        
        # Initialise sentence transformer model
        if GPU_AVAILABLE:
            print("Loading sentence transformer model on GPU")
            self.model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
            print(f"Model loaded on {self.device}")
        else:
            print("Loading sentence transformer model on CPU")
            self.model = SentenceTransformer('all-mpnet-base-v2')
            print("Model loaded on CPU")
        
        # Cache for embeddings to avoid recomputation
        self.fact_embeddings = None
        self.facts_list = None
        
    def load_kg_facts(self) -> List[str]:
        """Load and format KG facts for embedding"""
        kg_file = self.dataset_path / "kg/full.txt"
        facts = []
        
        print(f"Loading knowledge graph from {kg_file}")
        
        with open(kg_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 4:
                    subject, predicate, obj, timestamp = parts[:4]
                    
                    if self.dataset_name == "MultiTQ":
                        # MultiTQ: "Subject Predicate Object Timestamp"
                        subject = subject.replace('_', ' ')
                        predicate = predicate.replace('_', ' ')
                        obj = obj.replace('_', ' ')
                        fact = f"{subject} {predicate} {obj} {timestamp}"
                    else:
                        # TimeQuestions: "Subject predicate Object from start to end."
                        subject = subject.replace('_', ' ')
                        predicate = predicate.replace('_', ' ')
                        obj = obj.replace('_', ' ')
                        
                        if len(parts) >= 5:
                            end_time = parts[4]
                            fact = f"{subject} {predicate} {obj} from {timestamp} to {end_time}."
                        else:
                            if timestamp in ['1', '2916']:
                                fact = f"{subject} {predicate} {obj} {timestamp}"
                            else:
                                fact = f"{subject} {predicate} {obj} from {timestamp} to {timestamp}."
                    
                    facts.append(fact)
                    
                if line_num % 100000 == 0 and line_num > 0:
                    print(f"Processed {line_num:,} facts")
        
        print(f"Loaded {len(facts):,} facts")
        
        # Store facts for embedding computation
        self.facts_list = facts
        return facts
    
    def compute_fact_embeddings(self, facts: List[str], batch_size: int = 512):
        """Compute embeddings for all facts"""
        if self.fact_embeddings is not None:
            return self.fact_embeddings
        
        print(f"Computing embeddings for {len(facts):,} facts using {self.device}")
        print(f"Batch size: {batch_size}")
        
        start_time = time.time()
        
        # Encode facts in batches for memory efficiency
        all_embeddings = []
        
        for i in range(0, len(facts), batch_size):
            batch_facts = facts[i:i+batch_size]
            
            if GPU_AVAILABLE:
                # Use GPU for encoding
                with torch.amp.autocast('cuda'):  # Mixed precision for faster computation
                    batch_embeddings = self.model.encode(
                        batch_facts,
                        batch_size=batch_size,
                        device=self.device,
                        show_progress_bar=False,
                        convert_to_tensor=True
                    )
            else:
                batch_embeddings = self.model.encode(
                    batch_facts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True
                )
            
            all_embeddings.append(batch_embeddings)
            
            if i % (batch_size * 10) == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(facts) - i) / rate if rate > 0 else 0
                print(f"{i:,}/{len(facts):,} facts | {rate:.1f} facts/sec | ETA: {eta/60:.1f}min")
        
        # Concatenate all embeddings
        self.fact_embeddings = torch.cat(all_embeddings, dim=0)
        
        elapsed = time.time() - start_time
        print(f"Embeddings computed in {elapsed/60:.1f} minutes")
        print(f"Shape: {self.fact_embeddings.shape}")
        print(f"Memory: {self.fact_embeddings.element_size() * self.fact_embeddings.nelement() / 1e9:.2f} GB")
        
        return self.fact_embeddings
    
    def load_questions(self, split: str) -> List[Dict]:
        """Load questions for the split"""
        questions_file = self.dataset_path / f"questions/{split}.json"
        with open(questions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def semantic_fact_selection(self, question: str, answers: List[str], all_facts: List[str], max_facts: int = 11) -> List[str]:
        """Semantic fact selection using embeddings"""
        # Ensure fact embeddings are computed
        if self.fact_embeddings is None:
            self.compute_fact_embeddings(all_facts)
        
        # Create query from question and answers
        query_text = question + " " + " ".join(str(ans) for ans in answers)
        
        # Encode query
        if GPU_AVAILABLE:
            with torch.amp.autocast('cuda'):
                query_embedding = self.model.encode(
                    [query_text], 
                    device=self.device,
                    convert_to_tensor=True
                )
        else:
            query_embedding = self.model.encode(
                [query_text], 
                convert_to_tensor=True
            )
        
        # Move to same device as fact embeddings
        query_embedding = query_embedding.to(self.fact_embeddings.device)
        
        # Compute cosine similarities using GPU
        with torch.no_grad():
            similarities = F.cosine_similarity(
                query_embedding.unsqueeze(0), 
                self.fact_embeddings.unsqueeze(1), 
                dim=2
            ).squeeze()
        
        # Get top-k most similar facts
        top_k = min(max_facts * 2, len(all_facts))  # Get more candidates for filtering
        top_indices = torch.topk(similarities, top_k).indices
        
        # Convert back to CPU for fact selection
        top_indices = top_indices.cpu().numpy()
        
        # Get top facts and apply additional filtering
        candidate_facts = [all_facts[idx] for idx in top_indices]
        
        # Secondary filtering: prefer facts containing answer entities
        answer_text = " ".join(str(ans).lower() for ans in answers)
        scored_facts = []
        
        for fact in candidate_facts[:max_facts * 2]:
            fact_lower = fact.lower()
            score = 0
            
            # Boost score if fact contains answer entities
            for answer in answers:
                answer_str = str(answer).lower()
                if len(answer_str) > 2 and answer_str in fact_lower:
                    score += 10
            
            scored_facts.append((fact, score))
        
        # Sort by boost score and take top facts
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        selected_facts = [fact for fact, score in scored_facts[:max_facts]]
        
        # Fill remaining with highest similarity if needed
        if len(selected_facts) < max_facts:
            used_facts = set(selected_facts)
            for fact in candidate_facts:
                if fact not in used_facts:
                    selected_facts.append(fact)
                    if len(selected_facts) >= max_facts:
                        break
        
        return selected_facts[:max_facts]
    
    def format_prompt(self, question: str, answers: List[str], facts: List[str], split: str) -> Dict[str, str]:
        """Format prompt in TimeR4 style matching original format"""
        # Facts formatting 
        facts_str = "', '".join(facts)
        facts_formatted = f"['{facts_str}']"
        
        # Text content - exact TimeR4 format
        text = (
            "Based on the historical facts, please answer the given question. "
            "Please keep the answer as simple as possible and return all the possible answers as a list.\n"
            f"Historical facts:{facts_formatted}\n"
            f"Question:\n {question}"
        )
        
        # Answer formatting 
        answers_str = str(answers)  # TimeR4 uses str() of the list directly
        
        # Format varies by dataset and split
        if self.dataset_name == "MultiTQ":
            if split == "train":
                return {
                    "instruction": text,
                    "output": answers_str,
                    "input": ""
                }
            else:  # test
                return {
                    "text": text,
                    "answers": answers_str,
                    "question": question
                }
        else:  # TimeQuestions
            if split == "train":
                return {
                    "instruction": text,
                    "output": answers_str,
                    "input": ""
                }
            else:  # test  
                # TimeQuestions test uses different format
                return {
                    "question": question,
                    "answer": answers_str,
                    "id": len(facts),  # Simple ID based on fact count
                    "type": "temporal"
                }
    
    def generate_prompts(self, split: str) -> List[Dict[str, str]]:
        """Generate prompts with optimised processing"""
        print(f"\nProcessing {self.dataset_name.upper()} - {split.upper()}")
        print("=" * 60)
        
        # Load data
        all_facts = self.load_kg_facts()
        questions = self.load_questions(split)
        
        print(f"Processing {len(questions):,} questions with {len(all_facts):,} facts")
        
        prompts = []
        start_time = time.time()
        processed = 0
        skipped = 0
        
        for i, q_data in enumerate(questions):
            # Progress tracking
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(questions) - i) / rate if rate > 0 else 0
                print(f"{i:,}/{len(questions):,} questions | {len(prompts):,} prompts | {rate:.1f} q/sec | ETA: {eta/60:.1f}min")
            
            # Extract question and answers
            if "question" in q_data:
                question = q_data["question"]
                answers = q_data.get("answers", [])
            elif "Question" in q_data:
                question = q_data["Question"]
                answer_data = q_data.get("Answer", [])
                if answer_data and isinstance(answer_data, list):
                    answers = []
                    for ans in answer_data:
                        if isinstance(ans, dict) and "AnswerArgument" in ans:
                            answers.append(ans["AnswerArgument"])
                        elif isinstance(ans, str):
                            answers.append(ans)
                else:
                    answers = []
            else:
                skipped += 1
                continue
            
            if not answers:
                skipped += 1
                continue
            
            # Semantic fact selection using GPU embeddings - reduce filtering to match TimeR4
            relevant_facts = self.semantic_fact_selection(question, answers, all_facts)
            
            # TimeR4 is less strict about fact count - only require 1+ facts
            if len(relevant_facts) >= 1:
                prompt = self.format_prompt(question, answers, relevant_facts, split)
                prompts.append(prompt)
                processed += 1
            else:
                skipped += 1
        
        elapsed = time.time() - start_time
        print(f"\nGenerated {len(prompts):,} prompts in {elapsed/60:.1f} minutes")
        print(f"Processed: {processed:,}, Skipped: {skipped:,}")
        
        return prompts
    
    def save_prompts(self, prompts: List[Dict[str, str]], split: str):
        """Save prompts in TimeR4 format"""
        output_path = self.dataset_path / f"prompt/{split}_prompt.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, separators=(',', ':'), ensure_ascii=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Saved to {output_path} ({file_size_mb:.1f}MB)")

def process_dataset(dataset_name: str):
    """Process a single dataset with GPU-accelerated embeddings"""
    generator = EmbeddingGenerator(f"datasets/{dataset_name}")
    
    # Process available splits
    for split in ["test", "dev"]:
        try:
            prompts = generator.generate_prompts(split)
            if split == "dev":
                # Save dev as train to match TimeR4 structure
                generator.save_prompts(prompts, "train")
            else:
                generator.save_prompts(prompts, split)
        except FileNotFoundError:
            print(f"No {split} split found for {dataset_name}")
        
        # Clean up GPU memory between splits
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()

def main():
    """Main execution with GPU-accelerated embeddings"""
    print("Prompt Generator with GPU Embeddings")
    print("Semantic similarity matching using sentence transformers")
    print(f"Device: {DEVICE}")
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    if GPU_AVAILABLE:
        torch.cuda.manual_seed(42)
    
    start_time = time.time()
    
    # Process both datasets
    process_dataset("MultiTQ")
    
    # Clean up GPU memory between datasets
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
    
    process_dataset("TimeQuestions")
    
    elapsed = time.time() - start_time
    print(f"\nGPU-accelerated generation complete in {elapsed/60:.1f} minutes")
    
    if GPU_AVAILABLE:
        # Final cleanup
        torch.cuda.empty_cache()
        print("GPU memory cleaned up")

if __name__ == "__main__":
    main()
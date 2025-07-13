"""
Train temporal embeddings using quadruplet loss with batch-hard negative mining
Supports memory-efficient training on 8GB GPUs with gradient accumulation and mixed precision
"""

import os
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# Imports for monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
try:
    import psutil
    import GPUtil
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training temporal embeddings"""
    # Model
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    embedding_dim: int = 2048  # LLaMA-3.2-1B hidden size
    
    # Training
    batch_size: int = 32
    grad_accum_steps: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Quadruplet loss
    quadruplet_margin: float = 0.5
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16  # rank
    lora_alpha: int = 32  # scaling parameter
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None  # Will be set to ["q_proj", "v_proj", "k_proj", "o_proj"]
    temporal_weight: float = 0.3
    hard_negative_ratio: float = 0.8
    
    # Memory optimisation
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Paths
    data_dir: str = "./data/training"
    output_dir: str = "./models"
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_every: int = 100
    save_every: int = 1000
    eval_every: int = 500
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = "temporal-pathrag"
    
    def to_dict(self) -> dict:
        return asdict(self)


class TemporalQuadrupletDataset(Dataset):
    """Dataset for temporal quadruplet training examples"""
    
    def __init__(self, data_path: Path, split: str = "train"):
        """Load training data from JSON files"""
        self.data_path = data_path
        self.split = split
        
        # Load data
        data_file = data_path / f"{split}.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        with open(data_file, 'r') as f:
            self.examples = json.load(f)
            
        logger.info(f"Loaded {len(self.examples)} {split} examples from {data_file}")
        
        # Index by type for efficient sampling
        self.quadruplet_examples = []
        self.contrastive_examples = []
        self.reconstruction_examples = []
        
        for ex in self.examples:
            if ex['example_type'] == 'quadruplet':
                self.quadruplet_examples.append(ex)
            elif ex['example_type'].startswith('contrastive'):
                self.contrastive_examples.append(ex)
            elif ex['example_type'].startswith('reconstruction'):
                self.reconstruction_examples.append(ex)
                
        logger.info(f"Dataset breakdown - Quadruplet: {len(self.quadruplet_examples)}, "
                   f"Contrastive: {len(self.contrastive_examples)}, "
                   f"Reconstruction: {len(self.reconstruction_examples)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def get_batch_by_type(self, batch_size: int, example_type: str = "quadruplet"):
        """Get a batch of examples of specific type"""
        if example_type == "quadruplet" and self.quadruplet_examples:
            examples = random.sample(self.quadruplet_examples, 
                                   min(batch_size, len(self.quadruplet_examples)))
        elif example_type == "contrastive" and self.contrastive_examples:
            examples = random.sample(self.contrastive_examples,
                                   min(batch_size, len(self.contrastive_examples)))
        elif example_type == "reconstruction" and self.reconstruction_examples:
            examples = random.sample(self.reconstruction_examples,
                                   min(batch_size, len(self.reconstruction_examples)))
        else:
            examples = random.sample(self.examples, min(batch_size, len(self.examples)))
            
        return examples


class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention layer that learns to attend to different time periods
    Uses sinusoidal positional encoding for temporal information
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for temporal features
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal position encoding
        self.temporal_pos_encoding = nn.Parameter(
            torch.zeros(1, 1000, hidden_dim)  # Support up to 1000 time steps
        )
        
        # Layer normalisation
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.init_temporal_encoding()
    
    def init_temporal_encoding(self):
        """Initialise temporal positional encoding with sinusoidal pattern"""
        position = torch.arange(0, 1000).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / self.hidden_dim))
        
        self.temporal_pos_encoding.data[0, :, 0::2] = torch.sin(position * div_term)
        self.temporal_pos_encoding.data[0, :, 1::2] = torch.cos(position * div_term)
    
    def forward(self, embeddings: torch.Tensor, temporal_indices: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention to embeddings
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Add temporal positional encoding
        temporal_pos = self.temporal_pos_encoding[:, temporal_indices.flatten(), :].reshape(
            batch_size, seq_len, self.hidden_dim
        )
        embeddings_with_time = embeddings + temporal_pos
        
        # Apply temporal attention
        attn_output, _ = self.temporal_attention(
            embeddings_with_time, 
            embeddings_with_time, 
            embeddings_with_time
        )
        embeddings = self.layer_norm1(embeddings + attn_output)
        
        # Apply feed-forward network
        ffn_output = self.ffn(embeddings)
        embeddings = self.layer_norm2(embeddings + ffn_output)
        
        return embeddings


class TemporalEmbeddingModel(nn.Module):
    """
    Temporal embedding model with fine-tunable LLaMA model
    Supports temporal encoding and projection layers
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained LLaMA model with quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.encoder = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Apply LoRA if enabled
        if config.use_lora:
            # Set default target modules for LLaMA
            if config.lora_target_modules is None:
                config.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            self.encoder = get_peft_model(self.encoder, lora_config)
            logger.info(f"Applied LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
            self.encoder.print_trainable_parameters()
        
        # Temporal attention layer
        self.temporal_attention = TemporalAttentionLayer(
            hidden_dim=config.embedding_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Temporal encoding layer (after attention)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        # Optional projection head for training (can be removed after training)
        self.projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        # Initialise temporal encoder
        self.init_temporal_encoder()
        
    def init_temporal_encoder(self):
        """Initialise temporal encoder weights"""
        for module in self.temporal_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_text(self, texts: List[str], normalise: bool = True) -> torch.Tensor:
        """Encode text using LLaMA model"""
        # Tokenise inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.encoder.device)
        
        # Get hidden states from LLaMA
        with torch.no_grad():
            outputs = self.encoder(**inputs, output_hidden_states=True)
            # Use last hidden state, averaged across sequence length
            hidden_states = outputs.hidden_states[-1]
            embeddings = hidden_states.mean(dim=1)
            
        if normalise:
            embeddings = F.normalise(embeddings, p=2, dim=1)
            
        return embeddings
    
    def encode_temporal(self, embeddings: torch.Tensor, 
                       temporal_info: Optional[torch.Tensor] = None,
                       temporal_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add temporal encoding to embeddings with attention"""
        # Apply temporal attention if indices provided
        if temporal_indices is not None:
            # Ensure embeddings have sequence dimension
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)  # [batch, 1, hidden_dim]
                temporal_indices = temporal_indices.unsqueeze(1)  # [batch, 1]
            
            embeddings = self.temporal_attention(embeddings, temporal_indices)
            
            # Squeeze back if needed
            if embeddings.size(1) == 1:
                embeddings = embeddings.squeeze(1)
        
        # Apply temporal encoding
        if temporal_info is not None:
            # Combine with temporal information
            temporal_emb = self.temporal_encoder(temporal_info)
            embeddings = embeddings + self.config.temporal_weight * temporal_emb
        else:
            # Apply temporal encoder to embeddings directly
            embeddings = self.temporal_encoder(embeddings)
            
        # Apply projection head during training
        if self.training:
            embeddings = self.projection(embeddings)
            
        # Normalise
        embeddings = F.normalise(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def forward(self, texts: List[str], temporal_texts: Optional[List[str]] = None,
                temporal_indices: Optional[torch.Tensor] = None):
        """Forward pass for training"""
        # Encode text
        embeddings = self.encode_text(texts, normalise=False)
        
        # Encode temporal information if provided
        temporal_embeddings = None
        if temporal_texts:
            temporal_embeddings = self.encode_text(temporal_texts, normalise=False)
            
        # Apply temporal encoding and projection with attention
        embeddings = self.encode_temporal(embeddings, temporal_embeddings, temporal_indices)
        
        return embeddings


class TemporalQuadrupletLoss(nn.Module):
    """
    Quadruplet loss with batch-hard negative mining for temporal embeddings
    Supports three types of negatives: wrong entity, wrong time, no time
    """
    
    def __init__(self, margin: float = 0.5, temporal_weight: float = 0.3):
        super().__init__()
        self.margin = margin
        self.temporal_weight = temporal_weight
        
    def batch_hard_negative_mining(self, embeddings: torch.Tensor, 
                                   labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform batch-hard negative mining
        """
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create masks for positive and negative pairs
        labels = labels.unsqueeze(0)
        mask_positive = labels == labels.T
        mask_negative = ~mask_positive
        
        # Remove diagonal (self-similarity)
        mask_positive.fill_diagonal_(False)
        
        # Find hardest positive (furthest positive)
        positive_distances = distances * mask_positive.float()
        positive_distances[~mask_positive] = 0
        hardest_positive_dist, _ = positive_distances.max(dim=1)
        
        # Find hardest negative (closest negative)
        negative_distances = distances + mask_positive.float() * 1e9  # Mask positives with large value
        hardest_negative_dist, _ = negative_distances.min(dim=1)
        
        return hardest_positive_dist, hardest_negative_dist
    
    def forward(self, anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor,
                use_hard_negatives: bool = True) -> torch.Tensor:
        """
        Compute quadruplet loss
        """
        if use_hard_negatives:
            # Concatenate all embeddings for hard negative mining
            all_embeddings = torch.cat([anchor_embeddings, positive_embeddings, negative_embeddings], dim=0)
            
            # Create labels (0 for anchors, 1 for positives, 2 for negatives)
            batch_size = anchor_embeddings.size(0)
            labels = torch.cat([
                torch.zeros(batch_size),
                torch.ones(batch_size),
                torch.ones(batch_size) * 2
            ]).to(anchor_embeddings.device).long()
            
            # Perform hard negative mining
            pos_dist, neg_dist = self.batch_hard_negative_mining(all_embeddings[:batch_size], labels)
            
            # Compute quadruplet loss
            loss = F.relu(pos_dist - neg_dist + self.margin)
        else:
            # Standard quadruplet loss
            pos_dist = F.pairwise_distance(anchor_embeddings, positive_embeddings, p=2)
            neg_dist = F.pairwise_distance(anchor_embeddings, negative_embeddings, p=2)
            loss = F.relu(pos_dist - neg_dist + self.margin)
            
        return loss.mean()


class TemporalEmbeddingTrainer:
    """Trainer for temporal embeddings with memory-efficient training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialise model
        logger.info(f"Initialising model: {config.model_name}")
        self.model = TemporalEmbeddingModel(config).to(self.device)
        
        # Loss function
        self.criterion = TemporalQuadrupletLoss(
            margin=config.quadruplet_margin,
            temporal_weight=config.temporal_weight
        )
        
        # Optimiser
        self.optimiser = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision and torch.cuda.is_available() else None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialise wandb if requested
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=f"temporal_embeddings_{time.strftime('%Y%m%d_%H%M%S')}",
                config=config.to_dict()
            )
            
    def timestamp_to_index(self, timestamp: str) -> int:
        """Convert timestamp to temporal index (0-999)"""
        # Parse year from timestamp
        try:
            if '-' in timestamp:
                year = int(timestamp.split('-')[0])
            else:
                year = int(timestamp)
        except:
            year = 2010  # Default to middle of range
            
        # Map to 0-999 range (assuming years 1000-3000)
        index = min(max((year - 1000) // 2, 0), 999)
        return index
    
    def prepare_batch(self, examples: List[dict]) -> Tuple[List[str], List[str], List[str], torch.Tensor]:
        """Prepare batch of examples for training"""
        anchor_texts = []
        positive_texts = []
        negative_texts = []
        temporal_indices = []
        
        for ex in examples:
            # Get anchor text
            anchor = ex['anchor']
            anchor_text = f"At {anchor['timestamp']}, {anchor['subject']} {anchor['relation'].replace('_', ' ')} {anchor['object']}"
            anchor_texts.append(anchor_text)
            temporal_indices.append(self.timestamp_to_index(anchor['timestamp']))
            
            # Get positive text
            if 'positive' in ex and ex['positive']:
                positive = ex['positive']
                positive_text = f"At {positive['timestamp']}, {positive['subject']} {positive['relation'].replace('_', ' ')} {positive['object']}"
            else:
                # Use anchor as positive (for reconstruction tasks)
                positive_text = anchor_text
            positive_texts.append(positive_text)
            
            # Get negative text
            if 'negative' in ex and ex['negative']:
                negative = ex['negative']
                negative_text = f"At {negative['timestamp']}, {negative['subject']} {negative['relation'].replace('_', ' ')} {negative['object']}"
            else:
                # Generate random negative
                negative_text = f"At random_time, random_subject random_relation random_object"
            negative_texts.append(negative_text)
            
        return anchor_texts, positive_texts, negative_texts, torch.tensor(temporal_indices)
    
    def train_step(self, batch: List[dict]) -> dict:
        """Single training step"""
        self.model.train()
        
        # Prepare batch
        anchor_texts, positive_texts, negative_texts, temporal_indices = self.prepare_batch(batch)
        
        # Move temporal indices to device
        temporal_indices = temporal_indices.to(self.device)
        
        # Forward pass with mixed precision
        if self.config.mixed_precision:
            with torch.amp.autocast('cuda'):
                anchor_emb = self.model(anchor_texts, temporal_indices=temporal_indices)
                positive_emb = self.model(positive_texts, temporal_indices=temporal_indices)
                negative_emb = self.model(negative_texts, temporal_indices=temporal_indices)
                
                loss = self.criterion(
                    anchor_emb, positive_emb, negative_emb,
                    use_hard_negatives=random.random() < self.config.hard_negative_ratio
                )
        else:
            anchor_emb = self.model(anchor_texts)
            positive_emb = self.model(positive_texts)
            negative_emb = self.model(negative_texts)
            
            loss = self.criterion(
                anchor_emb, positive_emb, negative_emb,
                use_hard_negatives=random.random() < self.config.hard_negative_ratio
            )
        
        # Backward pass with gradient accumulation
        if self.config.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Metrics
        with torch.no_grad():
            pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2).mean()
            neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2).mean()
            
        return {
            'loss': loss.item(),
            'pos_dist': pos_dist.item(),
            'neg_dist': neg_dist.item(),
            'margin': (neg_dist - pos_dist).item()
        }
    
    def validate(self, val_dataset: TemporalQuadrupletDataset) -> dict:
        """Validation step"""
        self.model.eval()
        
        val_losses = []
        val_margins = []
        
        with torch.no_grad():
            # Sample validation batches
            for _ in range(min(50, len(val_dataset) // self.config.batch_size)):
                batch = val_dataset.get_batch_by_type(self.config.batch_size, "quadruplet")
                
                # Prepare batch
                anchor_texts, positive_texts, negative_texts, temporal_indices = self.prepare_batch(batch)
                
                # Move temporal indices to device
                temporal_indices = temporal_indices.to(self.device)
                
                # Forward pass
                anchor_emb = self.model(anchor_texts, temporal_indices=temporal_indices)
                positive_emb = self.model(positive_texts, temporal_indices=temporal_indices)
                negative_emb = self.model(negative_texts, temporal_indices=temporal_indices)
                
                loss = self.criterion(anchor_emb, positive_emb, negative_emb, use_hard_negatives=False)
                
                # Metrics
                pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2).mean()
                neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2).mean()
                margin = (neg_dist - pos_dist).item()
                
                val_losses.append(loss.item())
                val_margins.append(margin)
                
        return {
            'val_loss': np.mean(val_losses),
            'val_margin': np.mean(val_margins)
        }
    
    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
            'metrics': metrics
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = Path(self.config.output_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        self.global_step = checkpoint['global_step']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        logger.info(f"Loaded checkpoint from {checkpoint_path} (step {self.global_step})")
        
    def train(self, train_dataset: TemporalQuadrupletDataset, 
              val_dataset: Optional[TemporalQuadrupletDataset] = None):
        """Main training loop"""
        logger.info("Starting training")
        
        # Calculate total steps
        steps_per_epoch = len(train_dataset) // (self.config.batch_size * self.config.grad_accum_steps)
        total_steps = steps_per_epoch * self.config.num_epochs
        
        # Learning rate scheduler
        scheduler = OneCycleLR(
            self.optimiser,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_steps / total_steps
        )
        
        # Training metrics
        train_losses = []
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            epoch_losses = []
            
            # Create progress bar
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}")
            
            for step in pbar:
                # Accumulate gradients
                step_losses = []
                
                for accum_step in range(self.config.grad_accum_steps):
                    # Get batch
                    batch = train_dataset.get_batch_by_type(self.config.batch_size, "quadruplet")
                    
                    # Train step
                    metrics = self.train_step(batch)
                    step_losses.append(metrics['loss'])
                    
                # Optimiser step
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimiser)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimiser.step()
                    
                self.optimiser.zero_grad()
                scheduler.step()
                
                # Update metrics
                avg_loss = np.mean(step_losses)
                epoch_losses.append(avg_loss)
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Log metrics
                if self.global_step % self.config.log_every == 0:
                    log_metrics = {
                        'train_loss': avg_loss,
                        'learning_rate': scheduler.get_last_lr()[0],
                        'epoch': epoch + 1,
                        'step': self.global_step
                    }
                    
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log(log_metrics, step=self.global_step)
                        
                    # Log GPU memory if available
                    if MONITORING_AVAILABLE and torch.cuda.is_available():
                        gpu = GPUtil.getGPUs()[0]
                        log_metrics['gpu_memory_used'] = gpu.memoryUsed
                        log_metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                        
                # Validation
                if val_dataset and self.global_step % self.config.eval_every == 0:
                    val_metrics = self.validate(val_dataset)
                    
                    logger.info(f"Step {self.global_step} - Validation loss: {val_metrics['val_loss']:.4f}, "
                               f"Margin: {val_metrics['val_margin']:.4f}")
                    
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log(val_metrics, step=self.global_step)
                        
                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint(val_metrics, is_best=True)
                        
                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint({'train_loss': avg_loss})
                    
            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1} completed - Average loss: {avg_epoch_loss:.4f}")
            
        # Save final model
        final_path = Path(self.config.output_dir) / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict()
        }, final_path)
        logger.info(f"Training completed - final model saved to {final_path}")


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train temporal embeddings")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="MultiTQ", help="Dataset name")
    parser.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for models")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model name")
    parser.add_argument("--embedding_dim", type=int, default=2048, help="Embedding dimension")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Quadruplet loss arguments
    parser.add_argument("--quadruplet_margin", type=float, default=0.5, help="Quadruplet loss margin")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--temporal_weight", type=float, default=0.3, help="Temporal encoding weight")
    parser.add_argument("--hard_negative_ratio", type=float, default=0.8, help="Ratio of hard negative mining")
    
    # Memory optimisation
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--resume_from", type=str, help="Resume from checkpoint")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="temporal-pathrag", help="Wandb project name")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        quadruplet_margin=args.quadruplet_margin,
        temporal_weight=args.temporal_weight,
        hard_negative_ratio=args.hard_negative_ratio,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Log configuration
    logger.info("Training configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"{key}: {value}")
        
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available, training on CPU")
        
    # Load datasets
    data_path = Path(args.data_dir) / args.dataset
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        logger.info("Please run the training data pipeline first to generate training data")
        return
        
    train_dataset = TemporalQuadrupletDataset(data_path, split="train")
    val_dataset = TemporalQuadrupletDataset(data_path, split="validation")
    
    # Create trainer
    trainer = TemporalEmbeddingTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        
    # Train
    trainer.train(train_dataset, val_dataset)
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
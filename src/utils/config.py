"""
Configuration management for Temporal PathRAG
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DatasetType(Enum):
    """Supported dataset types"""
    MULTITQ = "MultiTQ"
    TIMEQUESTIONS = "TimeQuestions" 
    TOY = "toy"


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset"""
    name: str
    path: Path
    kg_file: str = "kg/full.txt"
    questions_dir: str = "questions"
    prompts_dir: str = "prompt"
    metadata_file: str = "metadata.json"


class TemporalPathRAGConfig:
    """
    Central configuration manager for Temporal PathRAG
    
    Handles dataset paths, model parameters, and environment settings
    in a modular, environment-independent way
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialise configuration
        """
        if base_dir is None:
            # Auto-detect project root (assume we're in src/)
            current_file = Path(__file__).resolve()
            # Go up from src/utils/config.py -> src/utils -> src -> project_root
            self.base_dir = current_file.parent.parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.datasets_dir = self.base_dir / "datasets"
        self.raw_datasets_dir = self.base_dir / "raw_datasets"
        self.analysis_results_dir = self.base_dir / "analysis_results"
        self.scripts_dir = self.base_dir / "scripts"
        
        # Dataset configurations
        self._dataset_configs = self.initialise_dataset_configs()
        
        # Model parameters
        self.temporal_config = self.initialise_temporal_config()
        
        # GPU and device settings
        self.device_config = self.initialise_device_config()
    
    def initialise_dataset_configs(self) -> Dict[DatasetType, DatasetConfig]:
        """Initialise dataset configurations"""
        return {
            DatasetType.MULTITQ: DatasetConfig(
                name="MultiTQ",
                path=self.datasets_dir / "MultiTQ",
                kg_file="kg/full.txt",
                questions_dir="questions",
                prompts_dir="prompt"
            ),
            DatasetType.TIMEQUESTIONS: DatasetConfig(
                name="TimeQuestions", 
                path=self.datasets_dir / "TimeQuestions",
                kg_file="kg/full.txt",
                questions_dir="questions",
                prompts_dir="prompt"
            ),
            DatasetType.TOY: DatasetConfig(
                name="toy",
                path=self.datasets_dir / "toy",
                kg_file="expanded_toy_graph.py",
                questions_dir="",
                prompts_dir=""
            )
        }
    
    def initialise_temporal_config(self) -> Dict[str, Any]:
        """Initialise temporal scoring configuration"""
        return {
            "decay_rate": 0.01,
            "temporal_window_days": 365,
            "chronological_weight": 0.3,
            "proximity_weight": 0.4,
            "consistency_weight": 0.3,
            "default_mode": "exponential_decay"
        }
    
    def initialise_device_config(self) -> Dict[str, Any]:
        """Initialise device and GPU configuration"""
        return {
            "use_gpu": True,
            "gpu_memory_fraction": 0.8,
            "embedding_batch_size": 32,
            "semantic_model": "all-mpnet-base-v2"
        }
    
    def get_dataset_path(self, dataset: DatasetType) -> Path:
        """Get the full path to a dataset"""
        config = self._dataset_configs[dataset]
        return config.path
    
    def get_dataset_config(self, dataset: DatasetType) -> DatasetConfig:
        """Get the full configuration for a dataset"""
        return self._dataset_configs[dataset]
    
    def get_kg_file_path(self, dataset: DatasetType) -> Path:
        """Get the path to the knowledge graph file for a dataset"""
        config = self._dataset_configs[dataset]
        return config.path / config.kg_file
    
    def get_questions_dir(self, dataset: DatasetType) -> Path:
        """Get the questions directory for a dataset""" 
        config = self._dataset_configs[dataset]
        return config.path / config.questions_dir
    
    def get_prompts_dir(self, dataset: DatasetType) -> Path:
        """Get the prompts directory for a dataset"""
        config = self._dataset_configs[dataset]
        return config.path / config.prompts_dir
    
    def get_output_dir(self, subdir: str = "") -> Path:
        """Get output directory for results"""
        output_dir = self.analysis_results_dir
        if subdir:
            output_dir = output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def get_temporal_config(self) -> Dict[str, Any]:
        """Get temporal scoring configuration"""
        return self.temporal_config.copy()
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration"""
        return self.device_config.copy()
    
    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all configured paths exist"""
        validation_results = {}
        
        # Check base directories
        validation_results["base_dir"] = self.base_dir.exists()
        validation_results["datasets_dir"] = self.datasets_dir.exists()
        
        # Check dataset paths
        for dataset_type, config in self._dataset_configs.items():
            key = f"dataset_{dataset_type.value}"
            validation_results[key] = config.path.exists()
            
            # Check key files within each dataset
            if config.path.exists():
                kg_path = config.path / config.kg_file
                validation_results[f"{key}_kg"] = kg_path.exists()
        
        return validation_results
    
    def __str__(self) -> str:
        """String representation of the configuration"""
        return f"""TemporalPathRAGConfig:
  Base Directory: {self.base_dir}
  Datasets Directory: {self.datasets_dir}
  Available Datasets: {[dt.value for dt in self._dataset_configs.keys()]}
  Temporal Window: {self.temporal_config['temporal_window_days']} days
  GPU Enabled: {self.device_config['use_gpu']}"""


# Global configuration instance
_global_config: Optional[TemporalPathRAGConfig] = None


def get_config(base_dir: Optional[str] = None) -> TemporalPathRAGConfig:
    """
    Get the global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = TemporalPathRAGConfig(base_dir)
    return _global_config


def set_config(config: TemporalPathRAGConfig) -> None:
    """Set a custom global configuration"""
    global _global_config
    _global_config = config
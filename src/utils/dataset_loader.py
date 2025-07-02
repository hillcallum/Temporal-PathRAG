"""
Dataset loading utilities for Temporal PathRAG.
"""

import networkx as nx
from typing import Optional, Dict, Any
from pathlib import Path

from .config import get_config, DatasetType
from ..kg.temporal_graph_storage import TemporalGraphDatabase


def load_dataset(dataset_name: str, config_override: Optional[Dict[str, Any]] = None) -> nx.DiGraph:
    """
    Load a temporal knowledge graph dataset by name
    """
    # Get configuration
    config = get_config()
    if config_override:
        # Apply any configuration overrides
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Map string to enum
    dataset_map = {
        "MultiTQ": DatasetType.MULTITQ,
        "TimeQuestions": DatasetType.TIMEQUESTIONS, 
        "toy": DatasetType.TOY
    }
    
    if dataset_name not in dataset_map:
        available = ", ".join(dataset_map.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    dataset_type = dataset_map[dataset_name]
    dataset_config = config.get_dataset_config(dataset_type)
    
    print(f"Loading {dataset_name} from {dataset_config.path}")
    
    # Validate dataset path exists
    if not dataset_config.path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_config.path}")
    
    # Load using TemporalGraphDatabase
    tkg_db = TemporalGraphDatabase()
    tkg_db.load_dataset_from_structure(str(dataset_config.path))
    
    graph = tkg_db.get_main_graph()
    stats = tkg_db.get_statistics()
    
    print(f"Loaded: {stats['total_quadruplets']:,} quadruplets")
    print(f"{stats['unique_entities']:,} entities, {stats['unique_relations']:,} relations")
    print(f"{stats['unique_timestamps']:,} timestamps")
    
    return graph


def load_multitq() -> nx.DiGraph:
    """Convenience function to load MultiTQ dataset"""
    return load_dataset("MultiTQ")


def load_timequestions() -> nx.DiGraph:
    """Convenience function to load TimeQuestions dataset"""
    return load_dataset("TimeQuestions")


def load_toy_graph() -> nx.DiGraph:
    """Convenience function to load toy graph dataset""" 
    return load_dataset("toy")


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset without loading it.
    """
    config = get_config()
    
    dataset_map = {
        "MultiTQ": DatasetType.MULTITQ,
        "TimeQuestions": DatasetType.TIMEQUESTIONS,
        "toy": DatasetType.TOY
    }
    
    if dataset_name not in dataset_map:
        available = ", ".join(dataset_map.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    dataset_type = dataset_map[dataset_name]
    dataset_config = config.get_dataset_config(dataset_type)
    
    # Check what files exist
    info = {
        "name": dataset_config.name,
        "path": str(dataset_config.path),
        "exists": dataset_config.path.exists(),
        "kg_file": dataset_config.kg_file,
        "questions_dir": dataset_config.questions_dir,
        "prompts_dir": dataset_config.prompts_dir
    }
    
    if info["exists"]:
        kg_path = dataset_config.path / dataset_config.kg_file
        questions_path = dataset_config.path / dataset_config.questions_dir
        prompts_path = dataset_config.path / dataset_config.prompts_dir
        
        info.update({
            "kg_file_exists": kg_path.exists(),
            "questions_dir_exists": questions_path.exists(),
            "prompts_dir_exists": prompts_path.exists(),
            "kg_file_path": str(kg_path),
            "questions_path": str(questions_path),
            "prompts_path": str(prompts_path)
        })
        
        # Get file sizes if they exist
        if kg_path.exists():
            info["kg_file_size_mb"] = round(kg_path.stat().st_size / (1024*1024), 2)
    
    return info


def list_available_datasets() -> Dict[str, Dict[str, Any]]:
    """
    List all available datasets with their information
    """
    datasets = ["MultiTQ", "TimeQuestions", "toy"]
    return {name: get_dataset_info(name) for name in datasets}


def validate_dataset_paths() -> Dict[str, bool]:
    """
    Validate that all dataset paths exist
    """
    config = get_config()
    return config.validate_paths()
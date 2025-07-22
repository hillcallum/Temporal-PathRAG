"""
Dataset loading utilities for Temporal PathRAG.
"""

import networkx as nx
import pickle
from typing import Optional, Dict, Any
from pathlib import Path
import hashlib
import json

from .config import get_config, DatasetType
from ..kg.storage.temporal_graph_storage import TemporalGraphDatabase


# Module-level cache for loaded datasets
dataset_cache = {}


def load_dataset(dataset_name: str, config_override: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> nx.DiGraph:
    """
    Load a temporal knowledge graph dataset by name with caching support
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
    
    # Create cache key based on dataset name and config override
    cache_key = f"{dataset_name}"
    if config_override:
        # Add hash of config override to cache key
        config_str = json.dumps(config_override, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_key = f"{dataset_name}_{config_hash}"
    
    # Check memory cache first
    if use_cache and cache_key in dataset_cache:
        print(f"Loading {dataset_name} from memory cache")
        return dataset_cache[cache_key]
    
    # Check disk cache
    import os
    cache_base = os.environ.get('TEMPORAL_PATHRAG_CACHE', str(Path.home() / ".temporal_pathrag_cache"))
    cache_dir = Path(cache_base) / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}_graph.pkl"
    
    # Validate dataset path exists
    if not dataset_config.path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_config.path}")
    
    kg_file_path = dataset_config.path / dataset_config.kg_file
    
    if use_cache and cache_file.exists() and kg_file_path.exists():
        # Check if cache is still valid (not older than source file)
        if cache_file.stat().st_mtime > kg_file_path.stat().st_mtime:
            print(f"Loading {dataset_name} from disk cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    graph = pickle.load(f)
                dataset_cache[cache_key] = graph
                
                # Print stats from cached graph
                num_nodes = graph.number_of_nodes()
                num_edges = graph.number_of_edges()
                print(f"Loaded cached graph: {num_nodes:,} nodes, {num_edges:,} edges")
                
                return graph
            except Exception as e:
                print(f"Warning: Failed to load cache, regenerating: {e}")
                # Continue with normal loading
    
    print(f"Loading {dataset_name} from {dataset_config.path}")
    
    # Load using TemporalGraphDatabase
    tkg_db = TemporalGraphDatabase()
    tkg_db.load_dataset_from_structure(str(dataset_config.path))
    
    graph = tkg_db.get_main_graph()
    stats = tkg_db.get_statistics()
    
    print(f"Loaded: {stats['total_quadruplets']:,} quadruplets")
    print(f"{stats['unique_entities']:,} entities, {stats['unique_relations']:,} relations")
    print(f"{stats['unique_timestamps']:,} timestamps")
    
    # Save to cache
    if use_cache:
        try:
            print(f"Saving to disk cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cache saved successfully")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
        
        # Always save to memory cache
        dataset_cache[cache_key] = graph
    
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


def clear_dataset_cache(dataset_name: Optional[str] = None) -> None:
    """
    Clear the dataset cache
    """
    global dataset_cache
    
    # Clear memory cache
    if dataset_name:
        keys_to_remove = [k for k in dataset_cache.keys() if k.startswith(dataset_name)]
        for key in keys_to_remove:
            del dataset_cache[key]
        print(f"Cleared memory cache for {dataset_name}")
    else:
        dataset_cache.clear()
        print("Cleared all memory cache")
    
    # Clear disk cache
    cache_dir = Path.home() / ".temporal_pathrag_cache" / "datasets"
    if cache_dir.exists():
        if dataset_name:
            # Remove specific dataset cache files
            for cache_file in cache_dir.glob(f"{dataset_name}*.pkl"):
                cache_file.unlink()
                print(f"Removed cache file: {cache_file}")
        else:
            # Remove all cache files
            for cache_file in cache_dir.glob("*.pkl"):
                cache_file.unlink()
                print(f"Removed cache file: {cache_file}")
            
            
def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache state
    """
    cache_dir = Path.home() / ".temporal_pathrag_cache" / "datasets"
    
    info = {
        "memory_cache": {
            "datasets": list(dataset_cache.keys()),
            "count": len(dataset_cache)
        },
        "disk_cache": {
            "path": str(cache_dir),
            "exists": cache_dir.exists(),
            "files": []
        }
    }
    
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pkl"))
        info["disk_cache"]["files"] = [
            {
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "modified": f.stat().st_mtime
            }
            for f in cache_files
        ]
        info["disk_cache"]["total_size_mb"] = sum(f["size_mb"] for f in info["disk_cache"]["files"])
        info["disk_cache"]["count"] = len(cache_files)
    
    return info
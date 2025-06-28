#!/usr/bin/env python3
"""
PathRAG Environment Verification Script
"""

import torch
import sys
import platform

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def setup_device_and_logging():
    """Setup device and logging"""
    device = get_device()
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch CUDA: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"CPU cores: {torch.get_num_threads()}")
    
    return device

def test_pathrag_environment():
    """Test PathRAG-specific functionality"""
    print("\n" + "="*50)
    print("PathRAG Environment Test")
    print("="*50)
    
    # System info
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    
    # Device setup
    device = setup_device_and_logging()
    
    # Test core libraries
    print("\nTesting core libraries...")
    try:
        import transformers
        import networkx
        import faiss
        import sentence_transformers
        import langchain
        import pandas as pd
        import numpy as np
        
        print(f"Transformers: {transformers.__version__}")
        print(f"NetworkX: {networkx.__version__}")
        print(f"FAISS: Available")
        print(f"Sentence-Transformers: Available")
        print(f"LangChain: Available")
        print(f"Pandas: {pd.__version__}")
        print(f"NumPy: {np.__version__}")
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # Test PathRAG modules
    print("\nTesting PathRAG modules")
    try:
        from src.kg.models import PathRAGNode, PathRAGEdge, Path
        from src.kg.path_traversal import BasicPathTraversal
        from data.toy_graph import ToyGraphBuilder
        print("PathRAG modules imported successfully")
        
        # Create toy graph
        builder = ToyGraphBuilder()
        graph = builder.get_graph()
        print(f"Toy graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Test traversal
        traversal = BasicPathTraversal(graph)
        paths = traversal.find_paths("albert_einstein", "ulm", max_hops=2)
        print(f"Path traversal: Found {len(paths)} paths")
        
    except Exception as e:
        print(f"PathRAG error: {e}")
        return False
    
    # Test device operations
    print(f"\nTesting device operations on {device}...")
    try:
        # Test basic tensor operations
        x = torch.randn(100, 768).to(device)
        y = torch.randn(768, 256).to(device)
        z = torch.mm(x, y)
        print(f"Matrix operations: {z.shape}")
        
        # Clean up
        del x, y, z
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Device operation error: {e}")
        return False
    
    print("\nAll tests passed - PathRAG environment is ready.")
    return True

if __name__ == "__main__":
    success = test_pathrag_environment()
    if not success:
        sys.exit(1)
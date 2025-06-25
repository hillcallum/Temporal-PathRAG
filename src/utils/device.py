"""
Device management utilities for PathRAG
"""

import torch
import logging

def get_device():
    """Get the best available device for PathRAG operations"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def setup_device_and_logging():
    """Setup device and configure logging for PathRAG"""
    device = get_device()
    print(f"PathRAG Environment Setup")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"CPU device - consider GPU for large-scale PathRAG operations")
    
    # Configure logging for PathRAG components
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return device

def get_memory_info():
    """Get memory information for the current device"""
    device = get_device()
    
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            'device': 'cuda',
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': total,
            'free_gb': total - allocated
        }
    else:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'device': device.type,
            'total_gb': mem.total / 1e9,
            'available_gb': mem.available / 1e9,
            'used_gb': mem.used / 1e9,
            'percent': mem.percent
        }

def optimise_for_pathrag():
    """Optimise PyTorch settings for PathRAG operations"""
    device = get_device()
    
    # Set optimal number of threads for CPU operations
    if device.type == 'cpu':
        import os
        num_threads = os.cpu_count()
        torch.set_num_threads(min(num_threads, 8))  # Cap at 8 for efficiency
        print(f"CPU threads: {torch.get_num_threads()}")
    
    # Enable cudNN benchmark for consistent input sizes
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"CUDA optimisations enabled")
    
    # Set memory allocation strategy
    if device.type == 'cuda':
        # Use memory pool for better allocation
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
    return device

def test_pathrag_operations():
    """Test basic operations needed for PathRAG"""
    device = get_device()
    print(f"\nTesting PathRAG operations on {device}...")
    
    try:
        # Test tensor operations
        x = torch.randn(100, 768).to(device)  # Typical embedding size
        y = torch.randn(768, 256).to(device)
        z = torch.mm(x, y)  # Matrix multiplication
        print(f"Tensor operations: OK ({z.shape})")
        
        # Test embedding operations
        embedding_dim = 768
        vocab_size = 1000
        embeddings = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
        input_ids = torch.randint(0, vocab_size, (32, 10)).to(device)
        output = embeddings(input_ids)
        print(f"Embedding operations: OK ({output.shape})")
        
        # Test similarity computation
        from torch.nn.functional import cosine_similarity
        sim = cosine_similarity(output.view(-1, embedding_dim), 
                              output.view(-1, embedding_dim))
        print(f"Similarity computation: OK ({sim.shape})")
        
        # Clean up
        del x, y, z, embeddings, output, sim
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"All PathRAG operations working correctly!")
        return True
        
    except Exception as e:
        print(f"Error in PathRAG operations: {e}")
        return False
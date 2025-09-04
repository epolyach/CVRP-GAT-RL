import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")

def test_gpu_operations():
    """Test various operations to see what runs on GPU vs CPU"""
    print("=== DETAILED GPU ANALYSIS ===\n")
    
    # Test 1: Basic tensor operations
    print("1. Basic Tensor Operations:")
    x_cpu = torch.randn(1000, 1000)
    x_gpu = torch.randn(1000, 1000, device=device)
    
    print(f"   CPU tensor device: {x_cpu.device}")
    print(f"   GPU tensor device: {x_gpu.device}")
    
    # Matrix multiplication
    start_time = time.time()
    result_gpu = torch.matmul(x_gpu, x_gpu)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    gpu_time = time.time() - start_time
    
    start_time = time.time()
    result_cpu = torch.matmul(x_cpu, x_cpu)
    cpu_time = time.time() - start_time
    
    print(f"   GPU matmul time: {gpu_time:.4f}s")
    print(f"   CPU matmul time: {cpu_time:.4f}s")
    print(f"   GPU speedup: {cpu_time/gpu_time:.2f}x\n")
    
    # Test 2: Graph attention operations
    print("2. Graph Attention Network Operations:")
    
    # Create sample graph
    num_nodes = 50
    node_features = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4))
    
    # Create GAT layer
    gat_layer = GATConv(16, 32, heads=4, dropout=0.1)
    
    # Test on CPU
    print("   Testing GAT on CPU...")
    gat_cpu = gat_layer
    start_time = time.time()
    output_cpu = gat_cpu(node_features, edge_index)
    cpu_gat_time = time.time() - start_time
    print(f"   CPU GAT time: {cpu_gat_time:.4f}s")
    print(f"   Output shape: {output_cpu.shape}")
    print(f"   Output device: {output_cpu.device}")
    
    # Test on GPU
    print("   Testing GAT on GPU...")
    gat_gpu = gat_layer.to(device)
    node_features_gpu = node_features.to(device)
    edge_index_gpu = edge_index.to(device)
    
    start_time = time.time()
    output_gpu = gat_gpu(node_features_gpu, edge_index_gpu)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    gpu_gat_time = time.time() - start_time
    print(f"   GPU GAT time: {gpu_gat_time:.4f}s")
    print(f"   Output shape: {output_gpu.shape}")
    print(f"   Output device: {output_gpu.device}")
    print(f"   GAT GPU speedup: {cpu_gat_time/gpu_gat_time:.2f}x\n")
    
    # Test 3: Memory usage
    print("3. Memory Usage Analysis:")
    if torch.cuda.is_available():
        print(f"   GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"   GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        print(f"   GPU memory total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB\n")
    
    # Test 4: Data loading and transfer
    print("4. Data Loading and Transfer:")
    
    # Create sample data
    data_list = []
    for i in range(5):
        num_nodes = 20
        node_features = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
        edge_attr = torch.randn(num_nodes * 3, 1)
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            demand=torch.randn(num_nodes),
            capacity=torch.tensor([30.0])
        )
        data_list.append(data)
    
    loader = DataLoader(data_list, batch_size=2, shuffle=False)
    
    for i, batch in enumerate(loader):
        print(f"   Batch {i+1}:")
        print(f"     Before .to(device) - x device: {batch.x.device}")
        
        batch_gpu = batch.to(device)
        print(f"     After .to(device) - x device: {batch_gpu.x.device}")
        print(f"     Edge index device: {batch_gpu.edge_index.device}")
        print(f"     Edge attr device: {batch_gpu.edge_attr.device}")
        print(f"     Demand device: {batch_gpu.demand.device}")
        
        if i >= 1:  # Only show first 2 batches
            break
    
    print("\n=== ANALYSIS COMPLETE ===")
    
    return {
        'gpu_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'gpu_matmul_speedup': cpu_time/gpu_time if torch.cuda.is_available() else 0,
        'gpu_gat_speedup': cpu_gat_time/gpu_gat_time if torch.cuda.is_available() else 0
    }

def check_original_code_gpu_patterns():
    """Check the GPU patterns in the original codebase"""
    print("\n=== ORIGINAL CODE GPU PATTERNS ===")
    
    # Read and analyze key files for GPU usage patterns
    files_to_check = [
        'main.py',
        'decoder/GAT_Decoder.py', 
        'model/Model.py',
        'encoder/GAT_Encoder.py'
    ]
    
    gpu_patterns = {
        'device_assignments': [],
        'to_device_calls': [],
        'cuda_checks': [],
        'device_parameters': []
    }
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                
                if 'torch.device' in line_stripped:
                    gpu_patterns['device_assignments'].append(f"{file_path}:{i} - {line_stripped}")
                
                if '.to(device)' in line_stripped or '.to(' in line_stripped:
                    gpu_patterns['to_device_calls'].append(f"{file_path}:{i} - {line_stripped}")
                
                if 'cuda' in line_stripped.lower():
                    gpu_patterns['cuda_checks'].append(f"{file_path}:{i} - {line_stripped}")
                
                if 'device=' in line_stripped:
                    gpu_patterns['device_parameters'].append(f"{file_path}:{i} - {line_stripped}")
                    
        except FileNotFoundError:
            print(f"   File not found: {file_path}")
    
    print("\n1. Device Assignments:")
    for pattern in gpu_patterns['device_assignments']:
        print(f"   {pattern}")
    
    print("\n2. .to(device) Calls:")
    for pattern in gpu_patterns['to_device_calls'][:10]:  # Limit output
        print(f"   {pattern}")
    
    print("\n3. CUDA References:")
    for pattern in gpu_patterns['cuda_checks']:
        print(f"   {pattern}")
    
    print(f"\nSummary:")
    print(f"   Device assignments: {len(gpu_patterns['device_assignments'])}")
    print(f"   .to(device) calls: {len(gpu_patterns['to_device_calls'])}")
    print(f"   CUDA references: {len(gpu_patterns['cuda_checks'])}")
    
    return gpu_patterns

if __name__ == "__main__":
    results = test_gpu_operations()
    patterns = check_original_code_gpu_patterns()
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"GPU Available: {results['gpu_available']}")
    if results['gpu_available']:
        print(f"GPU Device: {results['device_name']}")
        print(f"Matrix Multiplication Speedup: {results['gpu_matmul_speedup']:.2f}x")
        print(f"GAT Operation Speedup: {results['gpu_gat_speedup']:.2f}x")

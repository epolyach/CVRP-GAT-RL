import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from src_batch.model.Model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def check_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB, Total: {total:.2f}GB")

# Test with batch size 512
print("\nTesting memory requirements for batch_size=512, N=20 nodes")
check_memory()

# Create model
model = Model(
    node_input_dim=3,
    edge_input_dim=1, 
    hidden_dim=128,
    edge_dim=16,
    layers=4,
    negative_slope=0.2,
    dropout=0.1
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
check_memory()

# Generate a test batch
batch_data = []
for _ in range(512):
    n_nodes = 20
    coords = np.random.uniform(0, 1, size=(n_nodes, 2))
    demands = np.zeros(n_nodes)
    demands[1:] = np.random.randint(1, 11, size=n_nodes - 1)
    
    x = torch.FloatTensor(np.column_stack([coords, demands / 50]))
    
    edge_index = []
    edge_attr = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_index.append([i, j])
                dist = np.linalg.norm(coords[i] - coords[j])
                edge_attr.append([dist])
    
    edge_index = torch.LongTensor(edge_index).t()
    edge_attr = torch.FloatTensor(edge_attr)
    demand = torch.FloatTensor(demands)
    capacity = torch.FloatTensor([50])
    
    batch_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                           demand=demand, capacity=capacity))

print("\nCreating DataLoader...")
loader = DataLoader(batch_data, batch_size=512, shuffle=False)

print("\nProcessing batch...")
for batch in loader:
    batch = batch.to(device)
    check_memory()
    
    # Forward pass
    with torch.no_grad():
        actions, log_probs = model(batch, 40, greedy=True, T=1.0)
    
    print(f"Actions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    check_memory()
    break

print("\nMemory test complete!")

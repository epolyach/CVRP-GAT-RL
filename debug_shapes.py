import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from src_batch.model.Model import Model
from src_batch.RL.euclidean_cost import euclidean_cost

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Generate a test batch like production
batch_data = []
for i in range(2):  # Small test with 2 instances
    n_nodes = 20
    coords = np.random.uniform(0, 1, size=(n_nodes, 2))
    demands = np.zeros(n_nodes)
    demands[1:] = np.random.randint(1, 11, size=n_nodes - 1)
    
    node_features = np.column_stack([coords, demands / 50])
    x = torch.FloatTensor(node_features)
    
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
    demand = torch.FloatTensor(demands).unsqueeze(-1)
    capacity = torch.FloatTensor([50])
    
    batch_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                           demand=demand, capacity=capacity))

loader = DataLoader(batch_data, batch_size=2, shuffle=False)

# Initialize model
model = Model(
    node_input_dim=4,
    edge_input_dim=1,
    hidden_dim=128,
    edge_dim=16, 
    layers=4,
    negative_slope=0.2,
    dropout=0.1
).to(device)

model.eval()

for batch in loader:
    batch = batch.to(device)
    
    print(f"batch.x shape: {batch.x.shape}")
    print(f"batch.num_graphs: {batch.num_graphs}")
    print(f"batch.edge_index shape: {batch.edge_index.shape}")
    
    with torch.no_grad():
        actions, log_probs = model(batch, 40, greedy=True, T=1.0)
        
    print(f"actions shape: {actions.shape}")
    
    # Debug euclidean_cost
    static = batch.x[:, :2]  # Only x,y coordinates
    print(f"static shape before reshape: {static.shape}")
    
    num_nodes = batch.x.size(0) // batch.num_graphs
    print(f"num_nodes: {num_nodes}")
    
    static_reshaped = static.reshape(-1, num_nodes, 2)
    print(f"static_reshaped shape: {static_reshaped.shape}")
    
    static_transposed = static_reshaped.transpose(2, 1)
    print(f"static_transposed shape: {static_transposed.shape}")
    
    # Try computing cost
    try:
        cost = euclidean_cost(batch.x[:, :2], actions, batch)
        print(f"Cost computed successfully: {cost}")
    except Exception as e:
        print(f"Error in euclidean_cost: {e}")
    
    break

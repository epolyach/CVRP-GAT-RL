import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, DataLoader
import numpy as np
import logging
from gpu_monitor import GPUMonitor, profile_model_execution

logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")

def create_sample_vrp_instance(n_nodes=10, batch_size=2):
    """Create sample VRP instances for testing"""
    instances = []
    
    for b in range(batch_size):
        # Create random node coordinates (depot + customers)
        coords = torch.rand(n_nodes, 2) * 10  # Scale to reasonable coordinates
        
        # Node features: [x_coord, y_coord, is_depot]
        is_depot = torch.zeros(n_nodes, 1)
        is_depot[0] = 1  # First node is depot
        node_features = torch.cat([coords, is_depot], dim=1)  # Shape: (n_nodes, 3)
        
        # Demands (depot has demand 0)
        demands = torch.cat([torch.zeros(1), torch.randint(1, 10, (n_nodes-1,))]).float()
        
        # Capacity
        capacity = torch.tensor([30.0])  # Vehicle capacity
        
        # Create fully connected graph
        edge_indices = []
        edge_attrs = []
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_indices.append([i, j])
                    # Edge feature: Euclidean distance
                    dist = torch.norm(coords[i] - coords[j]).unsqueeze(0)
                    edge_attrs.append(dist)
        
        edge_index = torch.tensor(edge_indices).t()
        edge_attr = torch.stack(edge_attrs)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            demand=demands,
            capacity=capacity,
            num_nodes=n_nodes
        )
        instances.append(data)
    
    return DataLoader(instances, batch_size=1, shuffle=False)

def create_simple_gat_model():
    """Create a simple GAT model for testing GPU usage"""
    from torch_geometric.nn import GATConv, global_mean_pool
    
    class SimpleGATModel(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=128, output_dim=64):
            super().__init__()
            self.gat1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.1)
            self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.1)
            self.fc = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, data, n_steps=10, greedy=True, T=2.5):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # GAT layers
            x = self.gat1(x, edge_index)
            x = torch.relu(x)
            x = self.gat2(x, edge_index)
            
            # Global pooling
            graph_emb = global_mean_pool(x, batch)
            output = self.fc(graph_emb)
            
            # Mock actions and log_p for compatibility
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
            actions = torch.randint(0, data.x.size(0), (batch_size, n_steps), device=x.device)
            log_p = torch.randn(batch_size, device=x.device)
            
            return actions, log_p
    
    return SimpleGATModel()

def test_gpu_usage():
    """Test GPU usage with monitoring"""
    print("=== GPU USAGE ANALYSIS ===\n")
    
    # Create test data
    print("Creating sample VRP data...")
    data_loader = create_sample_vrp_instance(n_nodes=20, batch_size=3)
    
    # Create model
    print("Creating simple GAT model...")
    model = create_simple_gat_model().to(device)
    
    # Test basic tensor operations
    print("Testing basic GPU tensor operations...")
    monitor = GPUMonitor(monitor_interval=0.2)
    monitor.start_monitoring()
    
    # Test tensor creation and operations on GPU
    with torch.cuda.device(0):
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
    
    monitor.log_tensor_device("test_tensor_x", x)
    monitor.log_tensor_device("test_tensor_result", z)
    
    monitor.stop_monitoring()
    basic_stats = monitor.get_stats_summary()
    
    print("Basic tensor operation results:")
    for key, value in basic_stats.items():
        if key != 'tensor_devices':
            print(f"  {key}: {value}")
    
    # Test model inference with profiling
    print("\nTesting model inference with profiling...")
    prof, monitor = profile_model_execution(
        model, data_loader, device, 
        n_steps=20, greedy=True, T=2.5
    )
    
    return prof, monitor

if __name__ == "__main__":
    test_gpu_usage()

#!/usr/bin/env python3
"""
Paper Replication Training Script for GAT+RL VRP
Matches the exact parameters from paper 131659.pdf:
- 21 nodes (1 depot + 20 customers)  
- 768,000 instances (reusable dataset)
- Vehicle capacity = 30
- Batches per epoch = 1500 
- Dropout rate = 0.6
- Epochs = 101 (0-100)
- Input vertex dimension = 3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeomDataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import time
import os
import datetime
import pandas as pd
from tqdm import tqdm
import copy
import pickle

# Import model components
from src_batch.model.Model import Model
from src_batch.RL.euclidean_cost import euclidean_cost

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_replication_training.log'),
        logging.StreamHandler()
    ]
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# PAPER REPLICATION CONFIG - EXACT PARAMETERS FROM 131659.pdf
CONFIG = {
    'n_nodes': 21,               # 1 depot + 20 customers (paper specification)
    'n_customers': 20,           # Number of customers only
    'n_epochs': 101,             # Total epochs (0-100)
    'batch_size': 512,           # Batch size
    'batches_per_epoch': 1500,   # Batches per epoch (paper: 1500 iterations)
    'total_instances': 768000,   # Total training instances (paper specification)
    'n_instances_valid': 10000,  # Validation instances
    'vehicle_capacity': 30,      # Vehicle capacity (paper specification)
    'max_demand': 10,            # Maximum customer demand
    'learning_rate': 1e-4,       # Learning rate
    'hidden_dim': 128,           # Hidden dimension for GAT
    'edge_dim': 16,             # Edge dimension
    'n_layers': 4,              # Number of GAT layers
    'n_heads': 8,               # Number of attention heads
    'dropout': 0.6,             # Dropout rate (paper specification)
    'temperature': 2.5,         # Temperature for sampling
    'max_grad_norm': 2.0,       # Gradient clipping
}

# IMPORTANT: Input vertex dimension discussion
# Paper uses 3 dimensions: (x, y, demand)
# Our original code needed 4 due to encoder concatenation: (x, y, demand/capacity, demand)
# We'll try with 3 first and see if it works, otherwise use 4
VERTEX_DIM = 4  # Paper: 3, but encoder concatenates demand -> 4

def generate_paper_vrp_instance(n_nodes, capacity, max_demand, seed=None):
    """Generate VRP instance matching paper specifications"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coordinates (depot at index 0)
    # Paper uses [0, 1] x [0, 1] coordinate space
    coords = np.random.uniform(0, 1, size=(n_nodes, 2))
    
    # Generate demands (0 for depot, 1-max_demand for customers)
    demands = np.zeros(n_nodes)
    demands[1:] = np.random.randint(1, max_demand + 1, size=n_nodes - 1)
    
    # Create node features - PAPER FORMAT: [x, y, demand/capacity]
    # This matches the paper's 3-dimensional input
    node_features = np.column_stack([
        coords[:, 0],                    # x coordinate
        coords[:, 1],                    # y coordinate  
        demands / capacity               # normalized demand
    ])
    
    # Create fully connected graph
    edge_index = []
    edge_attr = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_index.append([i, j])
                # Edge feature: Euclidean distance
                dist = np.linalg.norm(coords[i] - coords[j])
                edge_attr.append([dist])
    
    # Convert to tensors
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_index).t()
    edge_attr = torch.FloatTensor(edge_attr)
    demand = torch.FloatTensor(demands).unsqueeze(-1)  # For encoder compatibility
    capacity_tensor = torch.FloatTensor([capacity])
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                demand=demand, capacity=capacity_tensor)

def create_paper_datasets(config):
    """Create datasets matching paper specifications"""
    logging.info("Generating paper-scale datasets...")
    logging.info(f"Total training instances: {config['total_instances']:,}")
    logging.info(f"This will take several minutes...")
    
    # Check if dataset already exists (caching for large dataset)
    train_cache_file = 'paper_train_dataset_768k.pkl'
    valid_cache_file = 'paper_valid_dataset_10k.pkl'
    
    if os.path.exists(train_cache_file) and os.path.exists(valid_cache_file):
        logging.info("Loading cached datasets...")
        with open(train_cache_file, 'rb') as f:
            train_data = pickle.load(f)
        with open(valid_cache_file, 'rb') as f:
            valid_data = pickle.load(f)
        logging.info(f"Loaded {len(train_data)} training and {len(valid_data)} validation instances from cache")
    else:
        logging.info("Generating new datasets (this may take 10-15 minutes)...")
        
        # Generate training data
        train_data = []
        for i in tqdm(range(config['total_instances']), desc="Generating training instances"):
            instance = generate_paper_vrp_instance(
                config['n_nodes'], 
                config['vehicle_capacity'],
                config['max_demand'],
                seed=i
            )
            train_data.append(instance)
        
        # Generate validation data
        valid_data = []
        for i in tqdm(range(config['n_instances_valid']), desc="Generating validation instances"):
            instance = generate_paper_vrp_instance(
                config['n_nodes'],
                config['vehicle_capacity'], 
                config['max_demand'],
                seed=config['total_instances'] + i
            )
            valid_data.append(instance)
        
        # Cache the datasets
        logging.info("Caching datasets for future use...")
        with open(train_cache_file, 'wb') as f:
            pickle.dump(train_data, f)
        with open(valid_cache_file, 'wb') as f:
            pickle.dump(valid_data, f)
    
    # Create DataLoaders
    train_loader = GeomDataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    valid_loader = GeomDataLoader(valid_data, batch_size=config['batch_size'], shuffle=False)
    
    logging.info(f"Created {len(train_data):,} training instances")
    logging.info(f"Created {len(valid_data):,} validation instances")
    logging.info(f"Training batches per epoch: {len(train_loader)} (using first {config['batches_per_epoch']})")
    
    return train_loader, valid_loader

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

class PaperRolloutBaseline:
    """Paper-compliant rollout baseline"""
    def __init__(self, model, config):
        self.config = config
        self.model = copy.deepcopy(model)
        self.model.eval()
        
    def eval(self, batch, n_steps):
        """Evaluate baseline on a batch"""
        self.model.eval()
        with torch.no_grad():
            actions, _ = self.model(batch, n_steps, greedy=True, T=1.0)
            cost = euclidean_cost(batch.x[:, :2], actions, batch)  # Use only x,y coords
        return cost.detach()
    
    def epoch_callback(self, model, epoch):
        """Update baseline every few epochs"""
        if epoch % 10 == 0:  # Update every 10 epochs (less frequent for stability)
            logging.info(f"Updating baseline at epoch {epoch}")
            self.model = copy.deepcopy(model)
            self.model.eval()

def train_epoch(model, train_loader, optimizer, baseline, epoch, config, writer):
    """Train for one epoch with paper parameters"""
    model.train()
    
    epoch_loss = []
    epoch_rewards = []
    
    # Use exactly the number of batches specified in paper
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= config['batches_per_epoch']:
            break
            
        batch = batch.to(device)
        
        # Forward pass with paper's decoding steps
        n_steps = (config['n_nodes'] - 1) * 2  # Paper uses 2x number of customers for decoding
        actions, log_probs = model(batch, n_steps, greedy=False, T=config['temperature'])
        
        # Calculate cost using only x,y coordinates
        cost = euclidean_cost(batch.x[:, :2], actions.detach(), batch)
        
        # Get baseline
        baseline_cost = baseline.eval(batch, n_steps)
        
        # Calculate advantage
        advantage = cost - baseline_cost
        
        # Calculate loss (REINFORCE)
        loss = torch.mean(advantage.detach() * log_probs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        
        # Update weights
        optimizer.step()
        
        # Log metrics
        epoch_loss.append(loss.item())
        epoch_rewards.append(cost.mean().item())
        
        if batch_idx % 100 == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_idx}/{config['batches_per_epoch']}, "
                        f"Loss: {loss.item():.4f}, Avg Cost: {cost.mean().item():.4f}")
    
    # Log epoch metrics
    avg_loss = np.mean(epoch_loss)
    avg_reward = np.mean(epoch_rewards)
    
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Cost/train', avg_reward, epoch)
    
    return avg_loss, avg_reward

def main():
    """Main training function - Paper Replication"""
    logging.info("=" * 60)
    logging.info("PAPER REPLICATION TRAINING - 131659.pdf")
    logging.info("=" * 60)
    logging.info(f"Configuration: {CONFIG}")
    
    # Estimate training time
    estimated_hours = CONFIG['n_epochs'] * CONFIG['batches_per_epoch'] * 0.5 / 3600  # Rough estimate
    logging.info(f"Estimated training time: {estimated_hours:.1f} hours")
    
    # Check initial GPU memory
    check_gpu_memory()
    
    # Create datasets (this will take time!)
    train_loader, valid_loader = create_paper_datasets(CONFIG)
    
    # Initialize model with paper parameters
    logging.info("Initializing model with paper specifications...")
    model = Model(
        node_input_dim=VERTEX_DIM,   # Try paper's 3 dimensions first
        edge_input_dim=1,            # distance
        hidden_dim=CONFIG['hidden_dim'],
        edge_dim=CONFIG['edge_dim'],
        layers=CONFIG['n_layers'],
        negative_slope=0.2,
        dropout=CONFIG['dropout']    # Paper's 0.6 dropout
    ).to(device)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check memory after model initialization
    check_gpu_memory()
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Initialize baseline
    baseline = PaperRolloutBaseline(model, CONFIG)
    
    # Initialize tensorboard
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'runs/paper_replication_{timestamp}')
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/paper_replication_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    import json
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Training history
    history = []
    best_train_cost = float('inf')
    
    # Training loop
    logging.info("Starting paper replication training...")
    logging.info("This will run for many hours - monitor with: tail -f paper_replication_training.log")
    
    try:
        for epoch in range(CONFIG['n_epochs']):
            start_time = time.time()
            
            # Train
            train_loss, train_cost = train_epoch(
                model, train_loader, optimizer, baseline, 
                epoch, CONFIG, writer
            )
            
            # Update baseline
            baseline.epoch_callback(model, epoch)
            
            # Save best model
            if train_cost < best_train_cost:
                best_train_cost = train_cost
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_cost': train_cost,
                    'config': CONFIG
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                logging.info(f"New best model saved! Train cost: {train_cost:.4f}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_cost': train_cost,
                    'config': CONFIG
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
            
            # Log progress
            epoch_time = time.time() - start_time
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_cost': train_cost,
                'time': epoch_time
            })
            
            logging.info(f"Epoch {epoch}/{CONFIG['n_epochs']-1} - "
                        f"Loss: {train_loss:.4f}, Train Cost: {train_cost:.4f}, "
                        f"Time: {epoch_time:.2f}s")
            
            # Check memory every 10 epochs
            if epoch % 10 == 0:
                check_gpu_memory()
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
    
    finally:
        # Save final model and history
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'config': CONFIG
        }, os.path.join(checkpoint_dir, 'final_model.pt'))
        
        # Save history to CSV
        pd.DataFrame(history).to_csv(os.path.join(checkpoint_dir, 'training_history.csv'), index=False)
        
        writer.close()
        logging.info("Paper replication training complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Production training script for GAT-RL VRP model
Configuration: N=20 nodes, 101 epochs, batch_size=512, 15 batches per epoch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
import time
import os
import datetime
import pandas as pd
from tqdm import tqdm

# Import model components
from src_batch.model.Model import Model
from src_batch.RL.euclidean_cost import euclidean_cost
from src_batch.RL.Rollout_Baseline import RolloutBaseline
from src_batch.instance_creator.InstanceGenerator import InstanceGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_training.log'),
        logging.StreamHandler()
    ]
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Training configuration
CONFIG = {
    'n_nodes': 20,           # Number of nodes (customers + depot)
    'n_epochs': 101,         # Total epochs (0-100)
    'batch_size': 512,       # Batch size
    'batches_per_epoch': 15, # Batches per epoch
    'n_instances_train': 512 * 15,  # Total training instances
    'n_instances_valid': 1000,      # Validation instances
    'vehicle_capacity': 50,  # Vehicle capacity
    'max_demand': 10,        # Maximum customer demand
    'learning_rate': 1e-4,   # Learning rate
    'hidden_dim': 128,       # Hidden dimension for GAT
    'edge_dim': 16,         # Edge dimension
    'n_layers': 4,          # Number of GAT layers
    'n_heads': 8,           # Number of attention heads
    'dropout': 0.1,         # Dropout rate
    'temperature': 2.5,     # Temperature for sampling
    'max_grad_norm': 2.0,   # Gradient clipping
}

def generate_vrp_instance(n_nodes, capacity, max_demand, seed=None):
    """Generate a single VRP instance"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coordinates (depot at index 0)
    coords = np.random.uniform(0, 1, size=(n_nodes, 2))
    
    # Generate demands (0 for depot, 1-max_demand for customers)
    demands = np.zeros(n_nodes)
    demands[1:] = np.random.randint(1, max_demand + 1, size=n_nodes - 1)
    
    # Create node features [x, y, demand/capacity]
    node_features = np.column_stack([coords, demands / capacity])
    
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
    demand = torch.FloatTensor(demands).unsqueeze(-1)  # Make it 2D for concatenation
    capacity_tensor = torch.FloatTensor([capacity])
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                demand=demand, capacity=capacity_tensor)

def create_datasets(config):
    """Create training and validation datasets"""
    logging.info("Generating datasets...")
    
    # Generate training data
    train_data = []
    for i in tqdm(range(config['n_instances_train']), desc="Generating training instances"):
        instance = generate_vrp_instance(
            config['n_nodes'], 
            config['vehicle_capacity'],
            config['max_demand'],
            seed=i
        )
        train_data.append(instance)
    
    # Generate validation data
    valid_data = []
    for i in tqdm(range(config['n_instances_valid']), desc="Generating validation instances"):
        instance = generate_vrp_instance(
            config['n_nodes'],
            config['vehicle_capacity'], 
            config['max_demand'],
            seed=config['n_instances_train'] + i
        )
        valid_data.append(instance)
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=False)
    
    logging.info(f"Created {len(train_data)} training instances")
    logging.info(f"Created {len(valid_data)} validation instances")
    
    return train_loader, valid_loader

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

def train_epoch(model, train_loader, optimizer, baseline, epoch, config, writer):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = []
    epoch_rewards = []
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= config['batches_per_epoch']:
            break
            
        batch = batch.to(device)
        
        # Forward pass
        actions, log_probs = model(batch, config['n_nodes'] * 2, greedy=False, T=config['temperature'])
        
        # Calculate reward (negative cost for minimization)
        cost = euclidean_cost(batch.x, actions.detach(), batch)
        
        # Get baseline
        baseline_cost = baseline.eval(batch, config['n_nodes'] * 2)
        
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
        
        if batch_idx % 5 == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_idx}/{config['batches_per_epoch']}, "
                        f"Loss: {loss.item():.4f}, Avg Cost: {cost.mean().item():.4f}")
    
    # Log epoch metrics
    avg_loss = np.mean(epoch_loss)
    avg_reward = np.mean(epoch_rewards)
    
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Cost/train', avg_reward, epoch)
    
    return avg_loss, avg_reward

def validate(model, valid_loader, config):
    """Validate the model"""
    model.eval()
    
    total_cost = []
    
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            
            # Greedy decoding for validation
            actions, _ = model(batch, config['n_nodes'] * 2, greedy=True, T=1.0)
            
            # Calculate cost
            cost = euclidean_cost(batch.x, actions, batch)
            total_cost.extend(cost.cpu().numpy())
    
    avg_cost = np.mean(total_cost)
    std_cost = np.std(total_cost)
    
    return avg_cost, std_cost

def main():
    """Main training function"""
    logging.info("Starting production training")
    logging.info(f"Configuration: {CONFIG}")
    
    # Check initial GPU memory
    check_gpu_memory()
    
    # Create datasets
    train_loader, valid_loader = create_datasets(CONFIG)
    
    # Initialize model
    logging.info("Initializing model...")
    model = Model(
        node_input_dim=4,  # x, y, demand/capacity + demand concatenated
        edge_input_dim=1,  # distance
        hidden_dim=CONFIG['hidden_dim'],
        edge_dim=CONFIG['edge_dim'],
        layers=CONFIG['n_layers'],
        negative_slope=0.2,
        dropout=CONFIG['dropout']
    ).to(device)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check memory after model initialization
    check_gpu_memory()
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Initialize baseline
    baseline = RolloutBaseline(model, valid_loader, n_nodes=CONFIG['n_nodes'], T=CONFIG['temperature'])
    
    # Initialize tensorboard
    writer = SummaryWriter(f'runs/production_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/production_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training history
    history = []
    best_valid_cost = float('inf')
    
    # Training loop
    logging.info("Starting training...")
    
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
            
            # Validate every 5 epochs
            if epoch % 5 == 0:
                valid_cost, valid_std = validate(model, valid_loader, CONFIG)
                writer.add_scalar('Cost/valid', valid_cost, epoch)
                
                # Save best model
                if valid_cost < best_valid_cost:
                    best_valid_cost = valid_cost
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'valid_cost': valid_cost,
                        'config': CONFIG
                    }, os.path.join(checkpoint_dir, 'best_model.pt'))
                    logging.info(f"New best model saved! Valid cost: {valid_cost:.4f}")
            else:
                valid_cost = valid_std = None
            
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
                'valid_cost': valid_cost,
                'valid_std': valid_std,
                'time': epoch_time
            })
            
            logging.info(f"Epoch {epoch}/{CONFIG['n_epochs']-1} - "
                        f"Loss: {train_loss:.4f}, Train Cost: {train_cost:.4f}, "
                        f"Valid Cost: {valid_cost:.4f if valid_cost else 'N/A'}, "
                        f"Time: {epoch_time:.2f}s")
            
            # Check memory periodically
            if epoch % 10 == 0:
                check_gpu_memory()
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
    
    finally:
        # Save final model
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
        logging.info("Training complete!")

if __name__ == "__main__":
    main()

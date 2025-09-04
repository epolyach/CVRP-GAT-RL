#!/usr/bin/env python3
"""
Paper Replication Training Script for GAT+RL VRP with Cached Datasets
Uses same infrastructure as main_train.py but with:
- Vehicle capacity = 30 (scaled up by 10x from 3)
- Demand range = [1,2,...,10] (scaled up by 10x from [0.1,0.2,...,1.0])
- Uses pre-generated cached datasets (paper_train_dataset_768k.pkl, paper_valid_dataset_10k.pkl)
"""

import time
import datetime
import torch
import os
import logging
import pickle
from torch.profiler import profile, record_function, ProfilerActivity
from torch_geometric.loader import DataLoader

from src_batch.model.Model import Model
from src_batch.train.train_model import train

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")

def load_cached_dataset(dataset_path, batch_size):
    """Load cached dataset from pickle file"""
    logging.info(f"Loading cached dataset from {dataset_path}")
    start_time = time.time()
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    
    end_time = time.time()
    logging.info(f"Cached dataset loaded in {end_time - start_time:.2f} seconds")
    logging.info(f"Dataset contains {len(dataset)} instances, {len(data_loader)} batches")
    
    return data_loader

def paper_replication_train_cached():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H")
    logging.info(f"Starting paper replication training with cached datasets: {now}h")
    
    # Define the folder and filename for the model checkpoints
    folder = f'checkpoints\\paper_replication_cached_{now}h'
    filename = 'paper_model_cached.pt'

    # Load cached datasets (already have correct capacity=30 and demand=[1,2,...,10])
    batch_size = 512
    valid_batch_size = 512
    
    train_dataset_path = 'paper_train_dataset_768k.pkl'
    valid_dataset_path = 'paper_valid_dataset_10k.pkl'
    
    # Check if cached datasets exist
    if not os.path.exists(train_dataset_path):
        logging.error(f"Training dataset not found: {train_dataset_path}")
        raise FileNotFoundError(f"Please ensure {train_dataset_path} exists")
        
    if not os.path.exists(valid_dataset_path):
        logging.error(f"Validation dataset not found: {valid_dataset_path}")
        raise FileNotFoundError(f"Please ensure {valid_dataset_path} exists")
    
    # Load cached dataloaders 
    start_to_load = time.time()
    logging.info("Loading cached datasets with paper specifications (capacity=30, demand=[1,2,...,10])")
    
    data_loader = load_cached_dataset(train_dataset_path, batch_size)
    valid_loader = load_cached_dataset(valid_dataset_path, valid_batch_size)
    
    end_of_load = time.time()
    logging.info(f"All cached data loaded in {end_of_load - start_to_load:.2f} seconds")
    
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Model parameters - matching paper specifications  
    node_input_dim = 3      # Paper: (x, y, demand/capacity) - NOTE: demand is now [1-10], capacity is 30
    edge_input_dim = 1      # Distance
    hidden_dim = 128        # Paper specification
    edge_dim = 16          # Paper specification
    layers = 4             # Paper specification
    negative_slope = 0.2   # Standard for GAT
    dropout = 0.6          # Paper specification
    n_steps = 100          # Decoding steps
    lr = 1e-4             # Learning rate
    T = 2.5               # Temperature for sampling
    num_epochs = 101      # Paper: 101 epochs (0-100)
    
    logging.info("Instantiating the model with paper specifications")
    logging.info("Note: Using scaled parameters - capacity=30, demand=[1,2,...,10]")
    
    # Instantiate the Model using same structure as main_train.py
    model = Model(node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout)
    
    logging.info("Calling the train function (same as main_train.py) with cached datasets")
    # Call the train function exactly as main_train.py does
    train(model, data_loader, valid_loader, folder, filename, lr, n_steps, num_epochs, T)
    
    logging.info("Paper replication training with cached datasets finished")

if __name__ == "__main__":
    pipeline_start = time.time()
    paper_replication_train_cached()
    pipeline_end = time.time()
    logging.info(f"Cached pipeline execution time: {pipeline_end - pipeline_start} seconds")

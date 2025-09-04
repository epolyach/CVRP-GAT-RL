#!/usr/bin/env python3
"""
Paper Replication Training Script for GAT+RL VRP
Uses same infrastructure as main_train.py but with paper-specific parameters:
- Vehicle capacity = 30 (paper specification)
- 768,000 training instances
- 10,000 validation instances
- All other parameters from paper 131659.pdf
"""

import time
import datetime
import torch
import os
import logging
from torch.profiler import profile, record_function, ProfilerActivity

from src_batch.model.Model import Model
from src_batch.train.train_model import train
from src_batch.instance_creator.instance_loader import instance_loader

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Running on device: {device}")

def paper_replication_train():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H")
    logging.info(f"Starting paper replication training pipeline: {now}h")
    
    # Define the folder and filename for the model checkpoints
    folder = f'checkpoints\\paper_replication_{now}h'
    filename = 'paper_model.pt'

    # Paper-specific configurations for instances
    # Paper uses 20 customers + 1 depot = 21 nodes, capacity = 30
    config = [
        {'n_customers': 20, 'max_demand': 10, 'max_distance': 100, 'num_instances': 768000}
    ]
    valid_config = [
        {'n_customers': 20, 'max_demand': 10, 'max_distance': 100, 'num_instances': 10000}
    ]
    
    # Create dataloaders with paper-specific vehicle capacity = 30
    start_to_load = time.time()
    logging.info("Creating dataloaders with paper specifications (capacity=30)")
    batch_size = 512
    save_to_csv = False
    vehicle_capacity = 30  # Paper specification (scaled up 10x from 3)
    
    data_loader = instance_loader(config, batch_size, save_to_csv, vehicle_capacity=vehicle_capacity)
    valid_batch_size = 512
    valid_loader = instance_loader(valid_config, valid_batch_size, save_to_csv, vehicle_capacity=vehicle_capacity) 
    end_of_load = time.time()
    logging.info(f"Paper data loaded in {end_of_load - start_to_load} seconds")
    
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Model parameters - matching paper specifications
    node_input_dim = 3      # Paper: (x, y, demand/capacity)
    # NOTE: With scaling fix - demand is now [1,2,...,10], capacity is 30
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
    # Instantiate the Model using same structure as main_train.py
    model = Model(node_input_dim, edge_input_dim, hidden_dim, edge_dim, layers, negative_slope, dropout)
    
    logging.info("Calling the train function (same as main_train.py)")
    # Call the train function exactly as main_train.py does
    train(model, data_loader, valid_loader, folder, filename, lr, n_steps, num_epochs, T)
    
    logging.info("Paper replication training pipeline finished")

if __name__ == "__main__":
    pipeline_start = time.time()
    paper_replication_train()
    pipeline_end = time.time()
    logging.info(f"Paper replication pipeline execution time: {pipeline_end - pipeline_start} seconds")

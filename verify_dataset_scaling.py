#!/usr/bin/env python3
"""
Quick verification script to check if cached datasets have correct scaling:
- Capacity should be 30 (not 3)
- Demand should be [1,2,...,10] (not [0.1,0.2,...,1.0])
"""

import pickle
import torch
import numpy as np

def check_dataset_scaling(dataset_path, sample_size=5):
    print(f"\n=== Checking {dataset_path} ===")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset size: {len(dataset)} instances")
    
    # Check first few samples
    capacities = []
    demands = []
    
    for i in range(min(sample_size, len(dataset))):
        data = dataset[i]
        capacity = data.capacity.item() if hasattr(data.capacity, 'item') else data.capacity[0]
        demand = data.demand.flatten().numpy() if hasattr(data.demand, 'numpy') else data.demand
        
        capacities.append(capacity)
        demands.extend(demand[demand > 0])  # Exclude depot (demand=0)
    
    print(f"Capacity values (sample): {capacities}")
    print(f"Unique capacity: {set(capacities)}")
    print(f"Demand range: min={min(demands):.3f}, max={max(demands):.3f}")
    print(f"Unique demands (first 20): {sorted(set(demands))[:20]}")
    
    # Check if scaling is correct
    expected_capacity = 30
    expected_demand_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    if all(c == expected_capacity for c in capacities):
        print("✅ Capacity scaling correct (30)")
    else:
        print(f"❌ Capacity scaling incorrect. Expected {expected_capacity}, got {set(capacities)}")
    
    actual_demand_range = sorted(set(demands))
    if all(d in expected_demand_range for d in actual_demand_range):
        print("✅ Demand scaling correct ([1,2,...,10])")
    else:
        print(f"❌ Demand scaling incorrect. Expected {expected_demand_range}, got sample: {actual_demand_range[:10]}")

if __name__ == "__main__":
    try:
        check_dataset_scaling('paper_train_dataset_768k.pkl', sample_size=10)
        check_dataset_scaling('paper_valid_dataset_10k.pkl', sample_size=10)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the dataset files exist and are accessible")

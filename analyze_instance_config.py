#!/usr/bin/env python3
"""
Analyze CVRP instance generation configuration
"""
import numpy as np
from typing import Dict, Any, List
import json

def generate_cvrp_instance(num_customers: int, capacity: int, coord_range: int, demand_range: List[int], seed: int = None) -> Dict[str, Any]:
    """Generate a CVRP instance (same as in train.py)"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate coordinates: random integers 0 to coord_range, then divide by coord_range for normalization to [0,1]
    coords = np.zeros((num_customers + 1, 2), dtype=np.float64)
    for i in range(num_customers + 1):
        coords[i] = np.random.randint(0, coord_range + 1, size=2) / coord_range
    
    # Generate integer demands from demand_range
    demands = np.zeros(num_customers + 1, dtype=np.int32)
    for i in range(1, num_customers + 1):  # Skip depot (index 0)
        demands[i] = np.random.randint(demand_range[0], demand_range[1] + 1)
    
    # Compute distance matrix
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    
    return {
        'coords': coords,
        'demands': demands.astype(np.int32),
        'distances': distances,
        'capacity': int(capacity)
    }

def analyze_configuration(num_customers, capacity, coord_range, demand_range, n_instances=10000):
    """Analyze the CVRP configuration to understand problem characteristics"""
    
    print(f"\n=== CVRP Instance Configuration Analysis ===")
    print(f"Number of customers: {num_customers}")
    print(f"Vehicle capacity: {capacity}")
    print(f"Coordinate range: [0, {coord_range}], normalized to [0, 1]")
    print(f"Demand range: [{demand_range[0]}, {demand_range[1]}]")
    print(f"Analyzing {n_instances} instances...\n")
    
    total_demands = []
    num_routes_list = []
    depot_coords = []
    customer_coords_x = []
    customer_coords_y = []
    
    for i in range(n_instances):
        instance = generate_cvrp_instance(num_customers, capacity, coord_range, demand_range, seed=i)
        
        # Total demand (excluding depot)
        total_demand = np.sum(instance['demands'][1:])
        total_demands.append(total_demand)
        
        # Number of routes needed (ceiling of total demand / capacity)
        num_routes = np.ceil(total_demand / capacity)
        num_routes_list.append(num_routes)
        
        # Depot location
        depot_coords.append(instance['coords'][0])
        
        # Customer coordinates
        customer_coords_x.extend(instance['coords'][1:, 0])
        customer_coords_y.extend(instance['coords'][1:, 1])
    
    # Convert to numpy arrays for statistics
    total_demands = np.array(total_demands)
    num_routes_list = np.array(num_routes_list)
    depot_coords = np.array(depot_coords)
    
    # Statistics
    print("=== Demand Statistics ===")
    print(f"Mean demand per customer: {(demand_range[0] + demand_range[1]) / 2:.2f}")
    print(f"Total demand per instance:")
    print(f"  Mean: {np.mean(total_demands):.2f}")
    print(f"  Median: {np.median(total_demands):.2f}")
    print(f"  Min: {np.min(total_demands)}")
    print(f"  Max: {np.max(total_demands)}")
    
    print("\n=== Route Statistics ===")
    print(f"Number of routes required (ceil(total_demand / capacity)):")
    print(f"  Mean: {np.mean(num_routes_list):.2f}")
    print(f"  Median: {np.median(num_routes_list):.0f}")
    print(f"  Min: {np.min(num_routes_list):.0f}")
    print(f"  Max: {np.max(num_routes_list):.0f}")
    
    # Distribution of routes
    unique_routes, counts = np.unique(num_routes_list, return_counts=True)
    print(f"\nRoute distribution:")
    for r, c in zip(unique_routes, counts):
        percentage = (c / n_instances) * 100
        print(f"  {int(r)} route(s): {c:5d} instances ({percentage:6.2f}%)")
    
    # Highlight problematic cases
    single_route_percentage = (np.sum(num_routes_list == 1) / n_instances) * 100
    print(f"\n⚠️  Instances requiring only 1 route (degenerate to TSP): {single_route_percentage:.2f}%")
    
    print("\n=== Coordinate Statistics ===")
    print(f"Depot location:")
    print(f"  Mean: ({np.mean(depot_coords[:, 0]):.3f}, {np.mean(depot_coords[:, 1]):.3f})")
    print(f"  Std: ({np.std(depot_coords[:, 0]):.3f}, {np.std(depot_coords[:, 1]):.3f})")
    
    print(f"\nCustomer coordinates:")
    print(f"  X range: [{np.min(customer_coords_x):.3f}, {np.max(customer_coords_x):.3f}]")
    print(f"  Y range: [{np.min(customer_coords_y):.3f}, {np.max(customer_coords_y):.3f}]")
    print(f"  X mean/std: {np.mean(customer_coords_x):.3f} ± {np.std(customer_coords_x):.3f}")
    print(f"  Y mean/std: {np.mean(customer_coords_y):.3f} ± {np.std(customer_coords_y):.3f}")
    
    # Problem diagnosis
    print("\n=== PROBLEM DIAGNOSIS ===")
    if single_route_percentage > 50:
        print("❌ CRITICAL: Majority of instances collapse to TSP!")
        print(f"   - Capacity ({capacity}) is too large for the demand")
        expected_total_demand = num_customers * (demand_range[0] + demand_range[1]) / 2
        print(f"   - Expected total demand: {expected_total_demand:.1f}")
        print(f"   - Current capacity allows all customers in one route")
        suggested_capacity = int(expected_total_demand / 3)  # Target ~3 routes
        print(f"   - Suggested capacity: {suggested_capacity} (for ~3 routes)")
    elif single_route_percentage > 10:
        print("⚠️  WARNING: Significant fraction of instances are trivial (single route)")
        print(f"   - Consider reducing capacity from {capacity}")
    else:
        print("✅ Configuration seems reasonable for CVRP")
    
    return {
        'num_customers': num_customers,
        'capacity': capacity,
        'coord_range': coord_range,
        'demand_range': demand_range,
        'mean_total_demand': float(np.mean(total_demands)),
        'mean_routes': float(np.mean(num_routes_list)),
        'single_route_percentage': float(single_route_percentage),
        'route_distribution': {int(r): int(c) for r, c in zip(unique_routes, counts)}
    }

# Analyze configurations from the repository
print("=" * 60)
print("ANALYZING CONFIGURATIONS FROM THE REPOSITORY")
print("=" * 60)

# Default configuration (from default.yaml)
print("\n" + "="*60)
print("DEFAULT CONFIGURATION (default.yaml)")
print("="*60)
default_stats = analyze_configuration(
    num_customers=20,
    capacity=30,
    coord_range=100,
    demand_range=[1, 10],
    n_instances=10000
)

# Tiny configuration (from tiny.yaml - inherits most from default)
print("\n" + "="*60)
print("TINY CONFIGURATION (tiny.yaml with inherited defaults)")
print("="*60)
tiny_stats = analyze_configuration(
    num_customers=7,  # Only this is overridden in tiny.yaml
    capacity=30,       # Inherited from default
    coord_range=100,   # Inherited from default
    demand_range=[1, 10],  # Inherited from default
    n_instances=10000
)

# Save analysis results
with open('instance_analysis.json', 'w') as f:
    json.dump({
        'default': default_stats,
        'tiny': tiny_stats
    }, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("Results saved to instance_analysis.json")
print("="*60)

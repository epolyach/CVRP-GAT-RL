#!/usr/bin/env python3
"""
Analyze CVRP instance generation configuration (Pure Python version)
"""
import random
import math
import json

def generate_cvrp_instance(num_customers, capacity, coord_range, demand_range, seed=None):
    """Generate a CVRP instance (same as in train.py, but in pure Python)"""
    if seed is not None:
        random.seed(seed)
    
    # Generate coordinates
    coords = []
    for i in range(num_customers + 1):
        x = random.randint(0, coord_range) / coord_range
        y = random.randint(0, coord_range) / coord_range
        coords.append([x, y])
    
    # Generate demands
    demands = [0]  # Depot has 0 demand
    for i in range(1, num_customers + 1):
        demands.append(random.randint(demand_range[0], demand_range[1]))
    
    return {
        'coords': coords,
        'demands': demands,
        'capacity': capacity
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
    
    for i in range(n_instances):
        instance = generate_cvrp_instance(num_customers, capacity, coord_range, demand_range, seed=i)
        
        # Total demand (excluding depot)
        total_demand = sum(instance['demands'][1:])
        total_demands.append(total_demand)
        
        # Number of routes needed (ceiling of total demand / capacity)
        num_routes = math.ceil(total_demand / capacity)
        num_routes_list.append(num_routes)
    
    # Statistics
    print("=== Demand Statistics ===")
    mean_demand_per_customer = (demand_range[0] + demand_range[1]) / 2
    print(f"Mean demand per customer: {mean_demand_per_customer:.2f}")
    print(f"Total demand per instance:")
    mean_total = sum(total_demands) / len(total_demands)
    print(f"  Mean: {mean_total:.2f}")
    sorted_demands = sorted(total_demands)
    median_total = sorted_demands[len(sorted_demands) // 2]
    print(f"  Median: {median_total}")
    print(f"  Min: {min(total_demands)}")
    print(f"  Max: {max(total_demands)}")
    
    print("\n=== Route Statistics ===")
    print(f"Number of routes required (ceil(total_demand / capacity)):")
    mean_routes = sum(num_routes_list) / len(num_routes_list)
    print(f"  Mean: {mean_routes:.2f}")
    sorted_routes = sorted(num_routes_list)
    median_routes = sorted_routes[len(sorted_routes) // 2]
    print(f"  Median: {median_routes}")
    print(f"  Min: {min(num_routes_list)}")
    print(f"  Max: {max(num_routes_list)}")
    
    # Distribution of routes
    route_counts = {}
    for r in num_routes_list:
        route_counts[r] = route_counts.get(r, 0) + 1
    
    print(f"\nRoute distribution:")
    for r in sorted(route_counts.keys()):
        c = route_counts[r]
        percentage = (c / n_instances) * 100
        print(f"  {r} route(s): {c:5d} instances ({percentage:6.2f}%)")
    
    # Highlight problematic cases
    single_route_count = route_counts.get(1, 0)
    single_route_percentage = (single_route_count / n_instances) * 100
    print(f"\n⚠️  Instances requiring only 1 route (degenerate to TSP): {single_route_percentage:.2f}%")
    
    # Problem diagnosis
    print("\n=== PROBLEM DIAGNOSIS ===")
    if single_route_percentage > 50:
        print("❌ CRITICAL: Majority of instances collapse to TSP!")
        print(f"   - Capacity ({capacity}) is too large for the demand")
        expected_total_demand = num_customers * mean_demand_per_customer
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
        'mean_total_demand': mean_total,
        'mean_routes': mean_routes,
        'single_route_percentage': single_route_percentage,
        'route_distribution': dict(route_counts)
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

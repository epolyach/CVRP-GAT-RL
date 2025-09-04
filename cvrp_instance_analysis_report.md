# CVRP Instance Generator Configuration Analysis

## Executive Summary

The GT and DGT models converge to ~0.449 cost per customer in the tiny configuration because of **incorrect instance generator parameters**. The vehicle capacity (30) is too large relative to the total demand from 7 customers (average ~38.6), resulting in 14.3% of instances collapsing to trivial single-route TSP problems. Additionally, the plot shows a hardcoded naive baseline of 1.053, but the actual models are achieving around 0.449, suggesting they may be learning better than the naive baseline but still not learning proper multi-route CVRP solutions.

## Configuration Analysis

### Current Configuration (tiny.yaml with inherited defaults)

| Parameter | Value | Source |
|-----------|-------|---------|
| **Number of customers** | 7 | tiny.yaml |
| **Vehicle capacity** | 30 | default.yaml (inherited) |
| **Demand range** | [1, 10] | default.yaml (inherited) |
| **Coordinate range** | [0, 100] normalized to [0, 1] | default.yaml (inherited) |
| **Expected total demand** | ~38.6 (7 × 5.5) | Computed |
| **Mean routes needed** | 1.86 | Computed |

### Instance Generation Code (src/pipelines/train.py)

```python
def generate_cvrp_instance(num_customers: int, capacity: int, coord_range: int, demand_range: List[int], seed: int | None = None):
    # Coordinates: random integers 0 to coord_range, then normalized to [0,1]
    coords[i] = np.random.randint(0, coord_range + 1, size=2) / coord_range
    
    # Demands: integer values in [demand_range[0], demand_range[1]]
    demands[i] = np.random.randint(demand_range[0], demand_range[1] + 1)
```

### Statistical Analysis Results

#### Tiny Configuration (7 customers, capacity=30)
- **Mean total demand**: 38.62
- **Route distribution**:
  - 1 route: 1,429 instances (14.29%) ⚠️
  - 2 routes: 8,560 instances (85.60%)
  - 3 routes: 11 instances (0.11%)
- **Problem**: 14.3% of instances degenerate to TSP (single route)

#### Default Configuration (20 customers, capacity=30)  
- **Mean total demand**: 109.91
- **Route distribution**:
  - 2 routes: 1 instance (0.01%)
  - 3 routes: 645 instances (6.45%)
  - 4 routes: 7,327 instances (73.27%)
  - 5 routes: 2,020 instances (20.20%)
  - 6 routes: 7 instances (0.07%)
- **Status**: ✅ Reasonable CVRP configuration

## Root Cause Analysis

### Why Models Converge to ~0.449

1. **Incorrect Baseline in Plot**: The plot uses a hardcoded naive baseline of 1.053 per customer, which represents the "star" solution (depot→customer→depot for each customer individually).

2. **Models Learning Sub-optimal Solutions**: The models achieve ~0.449 per customer, which is better than the naive baseline but suggests they're not learning proper multi-route optimization.

3. **Configuration Issue**: With capacity=30 for only 7 customers:
   - Average total demand is 38.6
   - Only needs ~1.86 routes on average
   - 14.3% of instances fit in a single vehicle (TSP)
   - This makes the problem too easy and doesn't force the models to learn proper routing

## Recommended Fixes

### Configuration Changes

#### For tiny.yaml:
```yaml
problem:
  num_customers: 7
  vehicle_capacity: 15  # Reduced from 30 (inherited)
  # This will give ~2.6 routes on average
```

#### For small/default configurations:
```yaml
problem:
  num_customers: 20
  vehicle_capacity: 30  # Keep current - already good
  demand_range: [1, 10]  # Keep current
```

### Expected Impact
With capacity=15 for tiny:
- Mean routes needed: ~2.6 
- Single-route instances: <5%
- Forces actual routing decisions
- Models should diverge from naive baseline

## Verification

The naive baseline calculation shows:
- **Naive cost (depot→customer→depot for each)**: ~1.05 per customer
- **Current model performance**: ~0.449 per customer
- **Gap**: Models are 57% better than naive, but this might be due to learning simple TSP-like solutions rather than proper CVRP routing

## Conclusion

The instance generator configuration is **incorrectly set** for the tiny configuration. The capacity (30) inherited from default.yaml is appropriate for 20 customers but too large for 7 customers. This causes:

1. 14.3% of instances to collapse to trivial single-route problems
2. Models to learn simplified solutions rather than proper multi-route optimization
3. Convergence to a sub-optimal but better-than-naive value (~0.449)

**Recommendation**: Reduce vehicle_capacity to 15 in tiny.yaml to create properly challenging CVRP instances that require 2-3 routes on average.

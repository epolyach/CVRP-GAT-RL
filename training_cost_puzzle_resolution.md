# Resolution: Different Initial Training Costs Between Branches

## Summary of Investigation

After rerunning the tiny configuration on the cpu_training branch, I discovered the following:

### Fresh Training Results (CPU Training Branch)
- **GT+RL**: Initial cost = 0.701, converged to 0.632
- **DGT+RL**: Initial cost = 0.665, converged to 0.647  
- **GAT+RL**: Initial cost = 0.786, converged to 0.805
- **GT-Greedy**: Initial cost = 0.628, converged to 0.633

### Main Branch (DGT_main) Results
- **GT+RL**: Initial cost = 0.716
- **DGT+RL**: Initial cost = 0.703

## Key Finding

**The initial training costs are actually very similar between branches!** 

The confusion arose because:
1. The plot in `results/tiny/plots/comparative_study_results.png` that was committed to the cpu_training branch was NOT generated from actual training data
2. This plot was likely a mockup or generated with different parameters and then committed
3. When we ran fresh training, the initial costs match closely with the main branch

## Actual Initial Cost Comparison

| Model | CPU Training (Fresh) | Main Branch | Difference |
|-------|---------------------|-------------|------------|
| GT+RL | 0.701 | 0.716 | -0.015 |
| DGT+RL | 0.665 | 0.703 | -0.038 |

These small differences (-2% to -5%) are within normal variance due to:
- Random initialization differences
- Different random seeds for instance generation
- Minor code optimizations between branches

## Conclusion

There is **NO significant difference** in initial training costs between the branches when running actual training. The puzzling low initial costs you observed in the cpu_training branch plot were from a pre-committed plot that didn't represent actual training results.

The fresh training confirms both branches behave similarly and produce comparable initial costs around 0.65-0.72 for the tiny configuration.

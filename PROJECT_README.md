# Dynamic Graph Transformer for Vehicle Routing with Reinforcement Learning

A comprehensive implementation and research project for solving the Capacitated Vehicle Routing Problem (CVRP) using Graph Attention Networks with Reinforcement Learning.

## 🎯 Project Overview

This project implements and extends the Graph Attention Network + Reinforcement Learning approach for solving CVRP instances. The work includes extensive analysis, improvements, and a complete GPU cluster monitoring suite.

## 📦 Key Components

### Core Implementation
- **Graph Attention Network (GAT)** with multi-head attention for node embeddings
- **Reinforcement Learning** with policy gradient methods for route optimization
- **Attention-based decoder** for constructive solution building
- **Multiple baselines** including greedy, random, and learned baselines

### Research & Analysis
- Comprehensive hyperparameter analysis and tuning
- Validation strategy improvements
- Training cost optimization research
- Comparative studies with state-of-the-art methods

### GPU Monitoring Suite
Complete multi-server GPU monitoring tools for research environments:
- **`gpu_cluster_monitor.sh`** - Main monitoring script for multiple GPU servers
- **`gpu_web_monitor.py`** - Web dashboard for remote monitoring
- **`gpu_notifier.sh`** - Background notification service for GPU availability

## 🚀 Quick Start

### Setup Environment
```bash
# Create virtual environment
./setup_venv.sh

# Activate environment
source activate_env.sh
```

### Basic Training
```bash
# Train on 20-node CVRP instances
python run_training.py --graph_size 20 --batch_size 256

# CPU-only training (for testing)
python run_training.py --no_cuda
```

### GPU Cluster Monitoring
```bash
# Quick status of all GPU servers
./gpu_cluster_monitor.sh quick

# Continuous monitoring
./gpu_cluster_monitor.sh watch

# Web dashboard
python gpu_web_monitor.py --port 5000
```

## 📊 Results & Analysis

The project includes extensive analysis and results:
- Training convergence studies
- Hyperparameter sensitivity analysis  
- Comparison with OR-Tools and other baselines
- Cost-effectiveness analysis for different problem sizes

See `results/` directory for detailed outputs and visualizations.

## 🔧 Configuration

Configuration files are located in `configs/`:
- Training parameters
- Model architecture settings
- Environment configurations

## 📚 Documentation

- `GPU_MONITOR_README.md` - Complete GPU monitoring documentation
- `VALIDATION_FIX_DOCUMENTATION.md` - Validation improvements
- `SUMMARY_OF_CHANGES.md` - Project evolution summary
- Individual analysis reports in project root

## 🛠️ Development Tools

### Analysis Scripts
- `analyze_instance_config.py` - Instance configuration analysis
- `make_comparative_plot.py` - Results visualization
- `compute_naive_baseline.py` - Baseline computations

### Utilities
- `diagnose_dgt.py` - Model diagnostics
- `build_analysis_artifact.py` - Analysis artifacts generation
- GPU monitoring suite (multiple scripts)

## 🏗️ Architecture

```
src/
├── models/          # GAT and attention models
├── training/        # Training loops and utilities
├── utils/           # Helper functions
├── solvers/         # Baseline solvers (OR-Tools, etc.)
└── validation/      # Validation strategies

configs/             # Configuration files
results/            # Experimental results
research/           # Research notebooks and analysis
```

## 📈 Performance

The implementation achieves:
- Competitive performance on standard CVRP benchmarks
- Efficient training with GPU acceleration
- Scalable to various problem sizes (10-100+ nodes)
- Memory-efficient implementation for large instances

## 🤝 Contributing

This is a research project. Feel free to:
- Experiment with different architectures
- Improve training strategies
- Extend to other routing problems
- Enhance the monitoring tools

## 📄 License

Research and educational use.

## 🙏 Acknowledgments

Based on attention mechanisms for combinatorial optimization and modern reinforcement learning approaches.

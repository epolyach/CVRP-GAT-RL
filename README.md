# GAT_RL

GAT_RL implements a Graph Attention Network (GAT)–based reinforcement‑learning solver for the Capacitated Vehicle Routing Problem (CVRP). This code originates from a dissertation project and had an article presented at ICORES 2025 and unifies recent advances in learned routing heuristics into a single end‑to‑end framework. At its core, GAT_RL combines the attention‑based encoder–decoder architecture introduced by Kool et al. in "Attention, Learn to Solve Routing Problems!" with the residual edge‑graph attention enhancements proposed by Lei et al. (2021) [ARXIV](https://arxiv.org/abs/2105.02730). Training relies on the DiCE estimator (Differentiable Monte Carlo Estimator) to eliminate dual‑actor overhead while preserving solution quality.

## Features
- Attention‑based GAT encoder with residual edge features
- DiCE estimator for low‑variance, single‑actor policy gradients
- Flexible decoding strategies (greedy, sampling, beam search)
- Synthetic and CVRPLIB benchmark support
- Near‑optimal solutions with inference in seconds

## Installation
```bash
git clone https://github.com/DanielSacy/GAT_RL.git
cd GAT_RL
pip install -r requirements.txt
```

## Training Scripts

### Paper Replication Training

For reproducing research results, use the specialized paper replication scripts with correct scaling parameters (capacity=30, demand=[1,2,...,10]):

#### 🚀 **paper_replication_train.py** (Recommended for new experiments)
Generates fresh datasets with paper-specific parameters and trains the model.

**Usage:**
```bash
python3 paper_replication_train.py
```

**Features:**
- Generates 768,000 training instances + 10,000 validation instances  
- Uses correct scaling: capacity=30, demand=[1,2,...,10]
- Paper-specific hyperparameters (dropout=0.6, epochs=101)
- Saves model to `checkpoints/paper_replication_*/paper_model.pt`

**Duration:** ~2-3 hours (includes dataset generation)

#### ⚡ **paper_replication_train_cached.py** (Recommended for fast iteration)
Uses pre-generated cached datasets for rapid training iterations.

**Prerequisites:**
```bash
# Ensure cached datasets exist:
ls -la paper_train_dataset_768k.pkl paper_valid_dataset_10k.pkl
```

**Usage:**
```bash
python3 paper_replication_train_cached.py
```

**Features:**
- Loads existing cached datasets (no generation time)
- Same training parameters as fresh version
- Saves model to `checkpoints/paper_replication_cached_*/paper_model_cached.pt`

**Duration:** ~30-60 minutes (training only)

#### 🔍 **verify_dataset_scaling.py** (Dataset verification tool)
Verifies that cached datasets have correct scaling parameters.

**Usage:**
```bash
python3 verify_dataset_scaling.py
```

**Output Example:**
```
=== Checking paper_train_dataset_768k.pkl ===
Dataset size: 768000 instances
Capacity values (sample): [30.0, 30.0, 30.0, 30.0, 30.0]
Unique capacity: {30.0}
✅ Capacity scaling correct (30)
✅ Demand scaling correct ([1,2,...,10])
```

### Legacy Training Options

#### **run_training.py** (Advanced configuration)
Original sophisticated training system with flexible model architectures.

```bash
python3 run_training.py --help
```

Features multiple model types (GAT+RL, GT+RL, DGT+RL) and advanced configuration options.

## Quickstart

### Quick Paper Replication
```bash
# Option 1: Fresh training (first time)
python3 paper_replication_train.py

# Option 2: Using cached datasets (faster)
python3 verify_dataset_scaling.py  # Verify datasets first
python3 paper_replication_train_cached.py
```

### Legacy Workflow
```bash
# Generate synthetic CVRP data
python generate_data.py --problem CVRP --num_nodes 50 --seed 42 --output data/cvrp50.pkl

# Train
python train.py \
  --dataset data/cvrp50.pkl \
  --model gat \
  --estimator dice \
  --epochs 100 \
  --batch_size 128

# Evaluate  
python evaluate.py \
  --checkpoint checkpoints/cvrp50_dice.pt \
  --dataset data/cvrp50.pkl \
  --decode greedy
```

## File Structure

```
.
├── paper_replication_train.py          # Paper replication (fresh datasets)
├── paper_replication_train_cached.py   # Paper replication (cached datasets) 
├── verify_dataset_scaling.py           # Dataset verification tool
├── run_training.py                      # Advanced training system
├── TRAINING_SCRIPTS_README.md           # Detailed training documentation
│
├── src/                                 # Original sophisticated framework
│   ├── models/                          # Model implementations  
│   ├── training/                        # Advanced training infrastructure
│   └── data/                            # Data generation utilities
│
├── src_batch/                           # Legacy training infrastructure
│   ├── train/                           # Basic training functions
│   ├── model/                           # Model definitions
│   └── instance_creator/                # Dataset generation
│
└── checkpoints/                         # Trained model storage
```

## Troubleshooting

### Dataset Issues
- **No cached datasets**: Use `paper_replication_train.py` to generate fresh ones
- **Wrong scaling in cached datasets**: Verify with `verify_dataset_scaling.py`
- **Memory issues**: Reduce batch size or use cached datasets

### Training Issues  
- **CUDA out of memory**: Reduce batch size from 512 to 256 or 128
- **Slow training**: Use `paper_replication_train_cached.py` with pre-generated datasets
- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

## Citation
If you use GAT_RL in your research, please cite:

```bibtex
@inproceedings{ICORES25,
  title={Integrating Machine Learning and Optimisation to Solve the Capacitated Vehicle Routing Problem},
  author={Pedrozo, Daniel Antunes and Gupta, Prateek and Meira, Jorge Augusto and Silva, Fabiano},
  booktitle={ICORES},
  year={2025}
}

@inproceedings{wouter_kool,
  title={Attention, Learn to Solve Routing Problems!},
  author={Kool, Wouter and van Hoof, Herke and Welling, Max},
  booktitle={ICLR},
  year={2019}
}

@article{kun_lei_21,
  title={Solve routing problems with a residual edge-graph attention neural network},
  author={Lei, Kun and Guo, Peng and Wang, Yi and Wu, Xiao and Zhao, Wenchao},
  journal={arXiv preprint arXiv:2105.02730},
  year={2021}
}

@article{DiCE,
  title={DiCE: The Infinitely Differentiable Monte Carlo Estimator},
  author={Foerster, Jakob and Farquhar, Gregory and Al-Shedivat, Maruan and Rockt{\"a}schel, Tim and Xing, Eric P and Whiteson, Shimon},
  journal={arXiv preprint arXiv:1802.05098},
  year={2018}
}
```

## License
This project is released under the MIT License.

# Training Scripts Documentation

## Available Training Scripts

### 1. **paper_replication_train.py** ⭐ (Recommended for new experiments)
- **Purpose**: Paper replication with fresh dataset generation
- **Features**: 
  - Uses same infrastructure as `src_batch/train/main_train.py`
  - Generates fresh datasets with correct scaling (capacity=30, demand=[1,2,...,10])
  - 768K training instances, 10K validation instances
  - Paper-specific parameters (dropout=0.6, epochs=101, etc.)
- **Usage**: `python3 paper_replication_train.py`
- **Duration**: ~2-3 hours (includes dataset generation time)

### 2. **paper_replication_train_cached.py** ⚡ (Recommended for fast iteration)
- **Purpose**: Paper replication using pre-generated cached datasets
- **Features**:
  - Uses existing `paper_train_dataset_768k.pkl` & `paper_valid_dataset_10k.pkl`
  - Same training infrastructure as above
  - Fast startup (no dataset generation)
- **Prerequisites**: Cached dataset files must exist and have correct scaling
- **Usage**: `python3 paper_replication_train_cached.py`
- **Duration**: ~30-60 minutes (just training, no data generation)

### 3. **run_training.py** 📋 (Original from git repository)
- **Purpose**: Original configurable training script from git repository
- **Features**:
  - Uses advanced configuration system
  - Supports multiple model types (GAT+RL, GT+RL, DGT+RL, etc.)
  - More flexible but complex
- **Usage**: See original repository documentation
- **Note**: This is the sophisticated training system from the original git repo

### 4. **src_batch/train/main_train.py** 🔧 (Infrastructure)
- **Purpose**: Simple training infrastructure used by legacy codebase
- **Features**:
  - Basic training loop implementation
  - Used as template for paper replication scripts
  - Originally had hardcoded capacity=3
- **Usage**: Can be called directly, but paper replication scripts are better

## Supporting Infrastructure

### **src_batch/train/train_model.py**
- Contains the actual `train()` function used by all simplified scripts
- Handles training loops, baseline updates, model saving

### **src_batch/instance_creator/**
- `InstanceGenerator.py` - Modified to support configurable vehicle capacity
- `instance_loader.py` - Modified to pass vehicle capacity parameter

## Verification

### **verify_dataset_scaling.py**
- **Purpose**: Check if cached datasets have correct scaling
- **Usage**: `python3 verify_dataset_scaling.py`
- **Output**: Verifies capacity=30 and demand=[1,2,...,10] in cached files

## Removed Scripts

The following obsolete scripts have been cleaned up:
- `paper_replication_train_backup.py` - Old complex implementation
- `production_train*.py` - Experimental versions with various hardcoded parameters
- `*_backup.py` files - Backups from modifications

## Recommendations

**For Paper Replication:**
1. First time: Use `paper_replication_train.py` to generate fresh datasets
2. Subsequent runs: Use `paper_replication_train_cached.py` for faster iterations
3. Verify scaling: Run `verify_dataset_scaling.py` to check cached datasets

**For Research:**
- Use `run_training.py` for advanced experiments with different model architectures
- Modify parameters in paper replication scripts for ablation studies

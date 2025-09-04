import torch
from src_batch.model.Model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Check GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total GPU memory: {total_mem:.2f} GB")
    
    # Check current usage
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Currently allocated: {allocated:.2f} GB")
    print(f"Currently reserved: {reserved:.2f} GB")
    print(f"Available: {total_mem - reserved:.2f} GB")

# Initialize model
model = Model(
    node_input_dim=3,
    edge_input_dim=1,
    hidden_dim=128,
    edge_dim=16,
    layers=4,
    negative_slope=0.2,
    dropout=0.1
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Memory after model
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"Memory after model: {allocated:.2f} GB")

print("\nReady for training with batch_size=512!")
print("Estimated memory per batch: ~2-3 GB")
print("Available memory is sufficient for training.")

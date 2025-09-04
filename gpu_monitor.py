import torch
import threading
import time
import psutil
import subprocess
import logging
from torch.profiler import profile, record_function, ProfilerActivity
import json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class GPUMonitor:
    def __init__(self, monitor_interval=1.0):
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.gpu_stats = []
        self.tensor_locations = {}
        
    def start_monitoring(self):
        """Start monitoring GPU usage"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logging.info("GPU monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring GPU usage"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        logging.info("GPU monitoring stopped")
        
    def _monitor_loop(self):
        """Internal monitoring loop"""
        while self.monitoring:
            try:
                # Get GPU memory info
                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    
                    # Get GPU utilization using nvidia-smi
                    try:
                        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                               '--format=csv,noheader,nounits'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            gpu_util, gpu_mem_used, gpu_mem_total_smi = result.stdout.strip().split(',')
                            gpu_utilization = float(gpu_util.strip())
                            gpu_memory_used = float(gpu_mem_used.strip()) / 1024  # Convert MB to GB
                        else:
                            gpu_utilization = 0
                            gpu_memory_used = gpu_mem_allocated
                    except:
                        gpu_utilization = 0
                        gpu_memory_used = gpu_mem_allocated
                    
                    stats = {
                        'timestamp': time.time(),
                        'gpu_utilization': gpu_utilization,
                        'gpu_memory_allocated': gpu_mem_allocated,
                        'gpu_memory_reserved': gpu_mem_reserved,
                        'gpu_memory_used_smi': gpu_memory_used,
                        'gpu_memory_total': gpu_mem_total
                    }
                    self.gpu_stats.append(stats)
                    
                time.sleep(self.monitor_interval)
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                
    def log_tensor_device(self, name, tensor):
        """Log where a tensor is located"""
        if torch.is_tensor(tensor):
            device = str(tensor.device)
            self.tensor_locations[name] = device
            logging.info(f"Tensor '{name}' is on device: {device}")
            
    def get_stats_summary(self):
        """Get summary of monitoring statistics"""
        if not self.gpu_stats:
            return "No monitoring data collected"
            
        avg_gpu_util = sum(s['gpu_utilization'] for s in self.gpu_stats) / len(self.gpu_stats)
        max_gpu_util = max(s['gpu_utilization'] for s in self.gpu_stats)
        max_gpu_mem = max(s['gpu_memory_allocated'] for s in self.gpu_stats)
        
        summary = {
            'avg_gpu_utilization': f"{avg_gpu_util:.2f}%",
            'max_gpu_utilization': f"{max_gpu_util:.2f}%", 
            'max_gpu_memory_allocated': f"{max_gpu_mem:.2f} GB",
            'total_monitoring_points': len(self.gpu_stats),
            'tensor_devices': self.tensor_locations
        }
        
        return summary

def profile_model_execution(model, data_loader, device, n_steps=10, greedy=True, T=2.5):
    """Profile model execution with GPU monitoring"""
    monitor = GPUMonitor(monitor_interval=0.5)
    
    logging.info("Starting model execution profiling")
    monitor.start_monitoring()
    
    # Profile PyTorch operations
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        with record_function("model_inference"):
            model.eval()
            with torch.inference_mode():
                for i, batch in enumerate(data_loader):
                    if i >= 3:  # Limit to first 3 batches for testing
                        break
                        
                    logging.info(f"Processing batch {i+1}")
                    
                    # Log input tensor locations
                    monitor.log_tensor_device(f"batch_{i}_x", batch.x)
                    monitor.log_tensor_device(f"batch_{i}_edge_attr", batch.edge_attr)
                    
                    # Move batch to device
                    batch = batch.to(device)
                    monitor.log_tensor_device(f"batch_{i}_x_after_to_device", batch.x)
                    
                    # Run model inference
                    with record_function(f"model_forward_batch_{i}"):
                        actions, log_p = model(batch, n_steps, greedy, T)
                        
                    # Log output tensor locations
                    monitor.log_tensor_device(f"actions_batch_{i}", actions)
                    monitor.log_tensor_device(f"log_p_batch_{i}", log_p)
                    
                    # Force GPU synchronization
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
    
    monitor.stop_monitoring()
    
    # Generate profiling report
    logging.info("Generating profiling report...")
    
    # Print PyTorch profiler results
    print("\n=== PYTORCH PROFILER RESULTS ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Print GPU monitoring results  
    print("\n=== GPU MONITORING RESULTS ===")
    stats = monitor.get_stats_summary()
    for key, value in stats.items():
        if key != 'tensor_devices':
            print(f"{key}: {value}")
    
    print("\n=== TENSOR DEVICE LOCATIONS ===")
    for tensor_name, device in stats['tensor_devices'].items():
        print(f"{tensor_name}: {device}")
        
    return prof, monitor

if __name__ == "__main__":
    print("GPU Monitor utility loaded")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available")

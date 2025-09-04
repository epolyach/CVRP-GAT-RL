#!/bin/bash

# GPU Monitor Script
# Shows GPU usage, processes, and users

echo "=== GPU Usage Monitor ==="
echo "Timestamp: $(date)"
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. No NVIDIA GPU or drivers not installed."
    exit 1
fi

# Main GPU status
echo "🖥️  GPU Status:"
echo "=================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits | while IFS=',' read -r index name mem_used mem_total gpu_util temp power_draw power_limit; do
    # Remove leading/trailing spaces
    index=$(echo "$index" | xargs)
    name=$(echo "$name" | xargs)
    mem_used=$(echo "$mem_used" | xargs)
    mem_total=$(echo "$mem_total" | xargs)
    gpu_util=$(echo "$gpu_util" | xargs)
    temp=$(echo "$temp" | xargs)
    power_draw=$(echo "$power_draw" | xargs)
    power_limit=$(echo "$power_limit" | xargs)
    
    # Calculate memory percentage
    mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "N/A")
    
    echo "GPU $index: $name"
    echo "  Memory: ${mem_used}MB / ${mem_total}MB (${mem_percent}%)"
    echo "  GPU Utilization: ${gpu_util}%"
    echo "  Temperature: ${temp}°C"
    echo "  Power: ${power_draw}W / ${power_limit}W"
    echo ""
done

echo ""
echo "👥 Processes Using GPU:"
echo "======================="

# Get detailed process information
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | while IFS=',' read -r pid process_name gpu_memory; do
    # Remove leading/trailing spaces
    pid=$(echo "$pid" | xargs)
    process_name=$(echo "$process_name" | xargs)
    gpu_memory=$(echo "$gpu_memory" | xargs)
    
    if [ -n "$pid" ] && [ "$pid" != "No running processes found" ]; then
        # Get process details from ps
        if ps -p "$pid" > /dev/null 2>&1; then
            user=$(ps -o user= -p "$pid" | xargs)
            cpu_percent=$(ps -o %cpu= -p "$pid" | xargs)
            mem_percent=$(ps -o %mem= -p "$pid" | xargs)
            start_time=$(ps -o lstart= -p "$pid" | xargs)
            full_command=$(ps -o command= -p "$pid")
            
            echo "🔹 PID: $pid | User: $user | Process: $process_name"
            echo "   GPU Memory: ${gpu_memory}MB"
            echo "   CPU: ${cpu_percent}% | RAM: ${mem_percent}%"
            echo "   Started: $start_time"
            echo "   Command: $full_command"
            echo ""
        else
            echo "🔹 PID: $pid | Process: $process_name (process no longer exists)"
            echo "   GPU Memory: ${gpu_memory}MB"
            echo ""
        fi
    fi
done

# Check if no processes found
if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "✅ No processes currently using GPU"
fi

echo ""
echo "📊 Quick Summary:"
echo "=================="

# Total GPU memory usage
total_gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
total_gpu_capacity=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | paste -sd+ | bc)
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)

echo "GPUs: $gpu_count"
echo "Total GPU Memory Used: ${total_gpu_mem}MB / ${total_gpu_capacity}MB"

# Count users
unique_users=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | while read pid; do
    if [ -n "$pid" ] && [ "$pid" != "No running processes found" ] && ps -p "$pid" > /dev/null 2>&1; then
        ps -o user= -p "$pid"
    fi
done | sort | uniq | wc -l)

echo "Active GPU Users: $unique_users"


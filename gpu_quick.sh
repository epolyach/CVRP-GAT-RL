#!/bin/bash

# Quick GPU check script
echo "🖥️ GPU Quick Status - $(date '+%H:%M:%S')"
echo "================================================"

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not available"
    exit 1
fi

# GPU usage summary
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r index name mem_used mem_total gpu_util temp; do
    mem_percent=$(echo "scale=0; $mem_used * 100 / $mem_total" | bc 2>/dev/null)
    printf "GPU%s: %s%% mem, %s%% util, %s°C\n" "$index" "$mem_percent" "$gpu_util" "$temp"
done

echo ""
echo "👥 Active Users:"
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | while read pid; do
    if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
        user=$(ps -o user= -p "$pid")
        cmd=$(ps -o comm= -p "$pid")
        printf "  • %s (PID:%s) running %s\n" "$user" "$pid" "$cmd"
    fi
done 2>/dev/null | sort -u

if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "  • No active users"
fi

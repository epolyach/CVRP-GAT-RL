#!/bin/bash

# Setup GPU monitoring aliases
echo "Setting up GPU monitoring aliases..."

# Add to bashrc if not already there
if ! grep -q "# GPU monitoring aliases" ~/.bashrc; then
    cat >> ~/.bashrc << 'ALIASES'

# GPU monitoring aliases
alias gpu='~/CVRP/GAT_RL/gpu_quick.sh'
alias gpufull='~/CVRP/GAT_RL/gpu_monitor.sh'
alias gpuwatch='watch -n 2 ~/CVRP/GAT_RL/gpu_quick.sh'
alias gpukill='nvidia-smi | grep python | awk "{print \$5}" | xargs -r kill'
ALIASES
    
    echo "✅ Aliases added to ~/.bashrc"
    echo "🔄 Run 'source ~/.bashrc' or start a new terminal to use them"
else
    echo "⚠️  Aliases already exist in ~/.bashrc"
fi

echo ""
echo "Available commands after sourcing ~/.bashrc:"
echo "  gpu      - Quick GPU status"
echo "  gpufull  - Detailed GPU status with processes"
echo "  gpuwatch - Watch GPU status (updates every 2 seconds)"
echo "  gpukill  - Kill all python processes using GPU (be careful!)"

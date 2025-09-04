#!/usr/bin/env python3
"""
Create publication-quality figure for GAT+RL VRP paper
Single curve only, with all parameters
Dimensions: 160mm x 80mm 
Output: PNG only
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 2.0

def mm_to_inches(mm):
    """Convert millimeters to inches for matplotlib"""
    return mm / 25.4

def create_paper_figure(csv_path, save_path='gat_rl_paper_figure.png'):
    """Create publication-quality figure with exact dimensions - single curve only"""
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure with exact dimensions: 160mm x 80mm
    fig_width_inches = mm_to_inches(160)
    fig_height_inches = mm_to_inches(80)
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width_inches, fig_height_inches))
    
    # Extract data
    epochs = df['epoch'].values
    train_costs = df['train_cost'].values
    
    # Calculate total training time
    total_time_seconds = df['time'].sum()
    total_time_minutes = total_time_seconds / 60
    
    # Plot SINGLE training cost curve only
    ax.plot(epochs, train_costs, 
            color='#1f77b4',  # Professional blue
            linewidth=2.2,
            label='GAT+RL rollout',
            alpha=0.95,
            marker='o',
            markersize=2,
            markevery=10)  # Marker every 10 epochs
    
    # Labels
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Cost', fontsize=12)
    
    # Grid - subtle
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Set axis limits with padding
    ax.set_xlim(-2, 102)
    y_margin = (train_costs.max() - train_costs.min()) * 0.08
    ax.set_ylim(train_costs.min() - y_margin, train_costs.max() + y_margin)
    
    # Add complete parameter text box
    params_text = (
        r'$\bf{GAT+RL\ rollout}$' + '\n' +
        '──────────────────\n' +
        f'N = 20\n' 
        f'Batch size = 512\n'
        f'Batches/epoch = 15\n'
        f'Learning rate = 1e-4\n'
        f'Temperature = 2.5\n'
        f'Training time = {total_time_minutes:.1f} min'
    )
    
    # Place text box in upper right
    ax.text(0.97, 0.96, params_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.6',
                     facecolor='white',
                     edgecolor='#1f77b4',
                     linewidth=1.2,
                     alpha=0.95))
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Enhance remaining spines
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Adjust ticks
    ax.tick_params(axis='both', which='major', length=4, width=0.8)
    
    # Set x-axis ticks at regular intervals
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(12, 16, 0.5))
    
    # Add minor ticks for professional look
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', length=2, width=0.5)
    
    # Tight layout with specific padding
    plt.tight_layout(pad=0.5)
    
    # Save ONLY as PNG
    plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.02, dpi=300)
    
    plt.close()  # Close to free memory
    
    print(f"✅ Figure saved (PNG only):")
    print(f"   - {save_path}")
    
    # Print statistics
    best_cost = train_costs.min()
    best_epoch = df.loc[df['train_cost'].idxmin(), 'epoch']
    
    print(f"\n📊 Training Statistics:")
    print(f"   - Initial cost: {train_costs[0]:.3f}")
    print(f"   - Final cost: {train_costs[-1]:.3f}") 
    print(f"   - Best cost: {best_cost:.3f} (epoch {best_epoch})")
    print(f"   - Improvement: {((train_costs[0] - train_costs[-1])/train_costs[0]*100):.1f}%")
    print(f"   - Total training time: {total_time_minutes:.1f} minutes")
    
    print(f"\n📐 Figure Dimensions:")
    print(f"   - Width: 160 mm")
    print(f"   - Height: 80 mm")
    
    return fig, ax

def main():
    """Main function"""
    
    # Find the most recent training history
    checkpoint_dirs = [d for d in os.listdir('checkpoints') if d.startswith('production_fixed_')]
    if not checkpoint_dirs:
        print("❌ No training history found")
        return
        
    latest_dir = sorted(checkpoint_dirs)[-1]
    csv_path = f'checkpoints/{latest_dir}/training_history.csv'
    
    print(f"📂 Loading data from: {csv_path}\n")
    
    # Create figure - PNG only
    output_file = f'gat_rl_paper_figure.png'
    fig, ax = create_paper_figure(csv_path, save_path=output_file)
    
    print("\n✨ Done! Single curve figure with all parameters created.")

if __name__ == "__main__":
    main()

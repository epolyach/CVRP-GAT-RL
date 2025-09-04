#!/usr/bin/env python3
"""
Create simplified publication-quality figure for GAT+RL VRP paper
Shows only training cost vs epoch with key parameters
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

def create_simple_training_figure(csv_path, save_path='gat_rl_training_simple.pdf'):
    """Create simplified publication-quality figure showing only training costs"""
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure with appropriate size for single column
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Extract data
    epochs = df['epoch'].values
    train_costs = df['train_cost'].values
    
    # Plot training cost as single curve
    ax.plot(epochs, train_costs, 
            color='#2E4057',  # Dark professional blue
            linewidth=2.0,
            label='GAT+RL rollout',
            alpha=0.95)
    
    # Smooth the curve with rolling average for visual clarity
    window = 5
    if len(train_costs) > window:
        smoothed = pd.Series(train_costs).rolling(window=window, center=True).mean()
        ax.plot(epochs, smoothed, 
                color='#048A81',  # Teal accent
                linewidth=2.5,
                alpha=0.7,
                linestyle='-')
    
    # Labels
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Cost', fontsize=12)
    
    # Grid - subtle
    ax.grid(True, linestyle=':', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits with padding
    ax.set_xlim(0, 100)
    y_margin = (train_costs.max() - train_costs.min()) * 0.05
    ax.set_ylim(train_costs.min() - y_margin, train_costs.max() + y_margin)
    
    # Add parameter text box (simplified)
    params_text = (
        'GAT+RL rollout\n'
        '─────────────\n'
        f'N = 20\n' 
        f'Batch size = 512\n'
        f'Batches/epoch = 15'
    )
    
    # Place text box
    ax.text(0.97, 0.95, params_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5',
                     facecolor='white',
                     edgecolor='#2E4057',
                     linewidth=1.0,
                     alpha=0.9))
    
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
    
    # Tight layout
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    
    print(f"✅ Simple figure saved as:")
    print(f"   - {save_path}")
    print(f"   - {save_path.replace('.pdf', '.png')}")
    
    # Print statistics
    print(f"\n📊 Training Results:")
    print(f"   - Initial cost: {train_costs[0]:.2f}")
    print(f"   - Final cost: {train_costs[-1]:.2f}") 
    print(f"   - Best cost: {train_costs.min():.2f}")
    print(f"   - Improvement: {((train_costs[0] - train_costs[-1])/train_costs[0]*100):.1f}%")
    
    return fig, ax

def main():
    """Main function"""
    
    # Find the most recent training history
    checkpoint_dirs = [d for d in os.listdir('checkpoints') if d.startswith('production_fixed_')]
    latest_dir = sorted(checkpoint_dirs)[-1]
    csv_path = f'checkpoints/{latest_dir}/training_history.csv'
    
    print(f"Creating simplified figure from: {csv_path}\n")
    
    # Create figure
    output_file = f'gat_rl_simple_{datetime.now().strftime("%Y%m%d")}.pdf'
    fig, ax = create_simple_training_figure(csv_path, save_path=output_file)
    
    plt.show()

if __name__ == "__main__":
    main()

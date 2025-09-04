#!/usr/bin/env python3
"""
Create publication-quality figure for GAT+RL VRP paper
Shows training cost vs epoch with key parameters
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5

def load_training_data(csv_path):
    """Load training history from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def create_training_figure(df, save_path='gat_rl_training_figure.pdf'):
    """Create publication-quality figure of training progress"""
    
    # Create figure with specific dimensions for single column paper
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    # Extract data
    epochs = df['epoch'].values
    train_costs = df['train_cost'].values
    
    # Plot training cost
    ax.plot(epochs, train_costs, 
            color='#1f77b4',  # Professional blue color
            linewidth=2.5,
            label='GAT+RL rollout',
            marker='o',
            markersize=3,
            markevery=5,  # Show markers every 5 epochs for clarity
            alpha=0.9)
    
    # Add validation costs where available (every 5 epochs)
    valid_epochs = df[df['valid_cost'].notna()]['epoch'].values
    valid_costs = df[df['valid_cost'].notna()]['valid_cost'].values
    
    if len(valid_costs) > 0:
        ax.plot(valid_epochs, valid_costs,
                color='#ff7f0e',  # Professional orange color
                linewidth=2,
                linestyle='--',
                label='Validation',
                marker='s',
                markersize=5,
                alpha=0.8)
    
    # Add shaded area for standard deviation if available
    if 'valid_std' in df.columns and len(valid_costs) > 0:
        valid_std = df[df['valid_std'].notna()]['valid_std'].values
        ax.fill_between(valid_epochs, 
                        valid_costs - valid_std, 
                        valid_costs + valid_std,
                        color='#ff7f0e',
                        alpha=0.15)
    
    # Labels and title
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Route Cost', fontsize=14, fontweight='bold')
    ax.set_title('GAT+RL Training Progress on VRP', fontsize=16, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)  # Put grid behind the data
    
    # Set axis limits with some padding
    ax.set_xlim(-2, max(epochs) + 2)
    y_min = min(train_costs.min(), valid_costs.min() if len(valid_costs) > 0 else train_costs.min())
    y_max = max(train_costs.max(), valid_costs.max() if len(valid_costs) > 0 else train_costs.max())
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add parameter box
    params_text = (
        r'$\bf{Parameters}$' + '\n' +
        f'Nodes (N): 20\n' +
        f'Batch size: 512\n' +
        f'Batches/epoch: 15\n' +
        f'Learning rate: 1e-4\n' +
        f'Temperature: 2.5'
    )
    
    # Place text box in upper right corner
    ax.text(0.98, 0.97, params_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.7',
                     facecolor='white',
                     edgecolor='gray',
                     linewidth=1.5,
                     alpha=0.95))
    
    # Add final performance metrics
    final_train_cost = train_costs[-1]
    best_epoch = df.loc[df['train_cost'].idxmin(), 'epoch']
    best_train_cost = df['train_cost'].min()
    
    # Add horizontal line for best performance
    ax.axhline(y=best_train_cost, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(max(epochs) * 0.7, best_train_cost + y_padding * 0.05,
            f'Best: {best_train_cost:.3f} (epoch {best_epoch})',
            fontsize=10, color='green', fontweight='bold')
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    
    # Customize spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', length=3, width=0.5)
    ax.tick_params(axis='both', which='major', length=5, width=1.2)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(save_path.replace('.pdf', '.eps'), format='eps', bbox_inches='tight', pad_inches=0.1)
    
    print(f"✅ Figure saved as:")
    print(f"   - {save_path} (PDF for LaTeX)")
    print(f"   - {save_path.replace('.pdf', '.png')} (PNG for preview)")
    print(f"   - {save_path.replace('.pdf', '.eps')} (EPS for some journals)")
    
    return fig, ax

def main():
    """Main function to create the figure"""
    
    # Find the most recent training history
    checkpoint_dirs = [d for d in os.listdir('checkpoints') if d.startswith('production_fixed_')]
    if not checkpoint_dirs:
        print("❌ No training history found in checkpoints/")
        return
    
    # Use the most recent one
    latest_dir = sorted(checkpoint_dirs)[-1]
    csv_path = f'checkpoints/{latest_dir}/training_history.csv'
    
    print(f"📊 Loading training data from: {csv_path}")
    
    # Load data
    df = load_training_data(csv_path)
    
    print(f"📈 Training statistics:")
    print(f"   - Total epochs: {len(df)}")
    print(f"   - Initial cost: {df.iloc[0]['train_cost']:.3f}")
    print(f"   - Final cost: {df.iloc[-1]['train_cost']:.3f}")
    print(f"   - Best cost: {df['train_cost'].min():.3f}")
    print(f"   - Improvement: {(1 - df['train_cost'].min()/df.iloc[0]['train_cost'])*100:.1f}%")
    
    # Create figure
    output_file = f'gat_rl_training_figure_{datetime.now().strftime("%Y%m%d")}.pdf'
    fig, ax = create_training_figure(df, save_path=output_file)
    
    # Show figure
    plt.show()
    
    print("\n📝 LaTeX usage:")
    print("\\begin{figure}[h]")
    print("    \\centering")
    print(f"    \\includegraphics[width=\\columnwidth]{{{output_file.replace('.pdf', '')}}}")
    print("    \\caption{Training progress of GAT+RL with rollout baseline on VRP-20. "
          "The model shows consistent improvement in route cost optimization over 101 epochs.}")
    print("    \\label{fig:gat_rl_training}")
    print("\\end{figure}")

if __name__ == "__main__":
    main()

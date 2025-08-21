#!/usr/bin/env python3
"""
Visualize latent embeddings at different epochs for selected experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import json
import argparse
from typing import List, Dict, Tuple

def load_embedding(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load embedding and labels from npz file."""
    data = np.load(file_path)
    # Handle different key names
    if 'embeddings' in data:
        embeddings = data['embeddings']
    elif 'coordinates' in data:
        embeddings = data['coordinates']
    else:
        raise KeyError(f"No embeddings or coordinates found in {file_path}")
    
    labels = data['labels'] if 'labels' in data else None
    return embeddings, labels

def get_available_epochs(exp_dir: Path) -> List[int]:
    """Get list of available epochs for an experiment."""
    latent_dir = exp_dir / "latent_viz"
    if not latent_dir.exists():
        return []
    
    epochs = []
    for file in latent_dir.glob("epoch_*.npz"):
        epoch_num = int(file.stem.split('_')[1])
        epochs.append(epoch_num)
    return sorted(epochs)

def parse_exp_params(exp_name: str) -> Dict[str, str]:
    """Parse experiment parameters from directory name."""
    params = {}
    parts = exp_name.split('_')
    for part in parts:
        if part.startswith('q'):
            params['queue'] = part[1:]
        elif part.startswith('k') and part[1:].isdigit():
            params['k'] = part[1:]
        elif part.startswith('n') and part[1:].isdigit():
            params['n_neg'] = part[1:]
        elif part.startswith('ui'):
            params['update_int'] = part[2:]
        elif part.startswith('lr'):
            params['lr'] = part[2:]
        elif part.startswith('w') and part[1:].isdigit():
            params['warmup'] = part[1:]
    return params

def plot_embeddings_evolution(exp_dir: Path, epochs_to_plot: List[int] = None, 
                             save_path: str = None):
    """Plot embeddings evolution across epochs."""
    
    # Get available epochs
    available_epochs = get_available_epochs(exp_dir)
    if not available_epochs:
        print(f"No embeddings found in {exp_dir}")
        return
    
    # If epochs_to_plot not specified, plot every 100 epochs
    if epochs_to_plot is None:
        epochs_to_plot = [e for e in range(0, 1001, 100) if e in available_epochs]
        # Add final epoch if not already included
        if available_epochs[-1] not in epochs_to_plot:
            epochs_to_plot.append(available_epochs[-1])
    else:
        # Filter to only available epochs
        epochs_to_plot = [e for e in epochs_to_plot if e in available_epochs]
    
    if not epochs_to_plot:
        print(f"No epochs to plot for {exp_dir}")
        return
    
    # Create figure
    n_epochs = len(epochs_to_plot)
    n_cols = min(4, n_epochs)
    n_rows = (n_epochs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_epochs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    # Parse parameters and create title
    exp_name = exp_dir.name
    params = parse_exp_params(exp_name)
    title_parts = []
    if 'queue' in params:
        title_parts.append(f"Queue={params['queue']}")
    if 'k' in params:
        title_parts.append(f"k={params['k']}")
    if 'n_neg' in params:
        title_parts.append(f"n_neg={params['n_neg']}")
    if 'update_int' in params:
        title_parts.append(f"update_interval={params['update_int']}")
    if 'warmup' in params:
        title_parts.append(f"warmup={params['warmup']}")
    if 'lr' in params:
        title_parts.append(f"lr={params['lr']}")
    
    title = "Latent Space Evolution - " + ", ".join(title_parts)
    fig.suptitle(title, fontsize=14, y=1.02)
    
    for idx, epoch in enumerate(epochs_to_plot):
        ax = axes[idx] if n_epochs > 1 else axes[0]
        
        # Load embeddings
        file_path = exp_dir / "latent_viz" / f"epoch_{epoch:04d}.npz"
        embeddings, labels = load_embedding(file_path)
        
        # Plot
        if labels is not None:
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                               c=labels, cmap='tab10', s=1, alpha=0.6)
            # Add colorbar for first plot only
            if idx == 0:
                plt.colorbar(scatter, ax=ax, label='Class')
        else:
            ax.scatter(embeddings[:, 0], embeddings[:, 1], s=1, alpha=0.6)
        
        ax.set_title(f'Epoch {epoch}')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    if n_epochs < len(axes):
        for idx in range(n_epochs, len(axes)):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    return fig

def plot_multiple_experiments(exp_dirs: List[Path], epochs_to_plot: List[int] = None,
                             save_dir: str = None):
    """Plot embeddings for multiple experiments."""
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            print(f"Experiment directory {exp_dir} does not exist")
            continue
        
        print(f"\nProcessing {exp_dir.name}...")
        
        save_path = None
        if save_dir:
            save_path = save_dir / f"{exp_dir.name}_evolution.png"
        
        plot_embeddings_evolution(exp_dir, epochs_to_plot, save_path)

def compare_experiments_at_epoch(exp_dirs: List[Path], epoch: int, save_path: str = None):
    """Compare multiple experiments at a specific epoch."""
    
    valid_exps = []
    for exp_dir in exp_dirs:
        file_path = exp_dir / "latent_viz" / f"epoch_{epoch:04d}.npz"
        if file_path.exists():
            valid_exps.append(exp_dir)
    
    if not valid_exps:
        print(f"No experiments have data for epoch {epoch}")
        return
    
    n_exps = len(valid_exps)
    n_cols = min(3, n_exps)
    n_rows = (n_exps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_exps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    fig.suptitle(f'Comparison at Epoch {epoch}', fontsize=16, y=1.02)
    
    for idx, exp_dir in enumerate(valid_exps):
        ax = axes[idx] if n_exps > 1 else axes[0]
        
        # Load embeddings
        file_path = exp_dir / "latent_viz" / f"epoch_{epoch:04d}.npz"
        embeddings, labels = load_embedding(file_path)
        
        # Plot
        if labels is not None:
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                               c=labels, cmap='tab10', s=1, alpha=0.6)
        else:
            ax.scatter(embeddings[:, 0], embeddings[:, 1], s=1, alpha=0.6)
        
        # Parse experiment parameters from directory name
        exp_name = exp_dir.name
        params = parse_exp_params(exp_name)
        
        # Create multi-line title with parameters
        title_lines = []
        title_lines.append(f"Queue={params.get('queue', '?')}, k={params.get('k', '?')}, n_neg={params.get('n_neg', '?')}")
        if 'update_int' in params or 'warmup' in params or 'lr' in params:
            line2 = []
            if 'update_int' in params:
                line2.append(f"ui={params['update_int']}")
            if 'warmup' in params:
                line2.append(f"w={params['warmup']}")
            if 'lr' in params:
                line2.append(f"lr={params['lr']}")
            title_lines.append(", ".join(line2))
        
        ax.set_title("\n".join(title_lines), fontsize=10)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    if n_exps < len(axes):
        for idx in range(n_exps, len(axes)):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize latent embeddings')
    parser.add_argument('--exp-dir', type=str, default='experiments/cifar10_latent_viz_sweep',
                       help='Base directory containing experiments')
    parser.add_argument('--experiments', nargs='+', type=str,
                       help='Specific experiment directories to plot')
    parser.add_argument('--epochs', nargs='+', type=int,
                       help='Specific epochs to plot (default: every 100)')
    parser.add_argument('--compare-epoch', type=int,
                       help='Compare all experiments at this specific epoch')
    parser.add_argument('--save-dir', type=str, default='latent_viz_plots',
                       help='Directory to save plots')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Plot top N experiments (by final epoch count)')
    
    args = parser.parse_args()
    
    base_dir = Path(args.exp_dir)
    
    # Get experiment directories
    if args.experiments:
        exp_dirs = [base_dir / exp for exp in args.experiments]
    else:
        # Get all experiment directories
        exp_dirs = [d for d in base_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('q')]
        
        # Sort by number of available epochs (descending) and take top N
        exp_dirs_with_epochs = [(d, len(get_available_epochs(d))) for d in exp_dirs]
        exp_dirs_with_epochs.sort(key=lambda x: x[1], reverse=True)
        exp_dirs = [d for d, _ in exp_dirs_with_epochs[:args.top_n]]
        
        print(f"Selected top {args.top_n} experiments with most epochs:")
        for d, n_epochs in exp_dirs_with_epochs[:args.top_n]:
            print(f"  {d.name}: {n_epochs} epochs")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare_epoch is not None:
        # Compare all experiments at a specific epoch
        print(f"\nComparing experiments at epoch {args.compare_epoch}")
        compare_experiments_at_epoch(
            exp_dirs, 
            args.compare_epoch,
            save_dir / f"comparison_epoch_{args.compare_epoch}.png"
        )
    else:
        # Plot evolution for each experiment
        plot_multiple_experiments(exp_dirs, args.epochs, save_dir)
    
    print(f"\nAll plots saved to {save_dir}/")

if __name__ == "__main__":
    main()
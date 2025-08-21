"""Analysis pipeline for LatentViz parameter sweep experiments."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


class ExperimentAnalyzer:
    """Analyze results from LatentViz parameter sweep experiments."""
    
    def __init__(self, base_dir: Path = Path("experiments/cifar10_latent_viz_sweep/cifar10")):
        self.base_dir = base_dir
        self.results_df = None
        
    def load_experiment_results(self) -> pd.DataFrame:
        """Load all experiment results into a DataFrame."""
        results = []
        
        # Find all experiment directories
        for exp_dir in self.base_dir.glob("q*_k*_n*"):
            if not exp_dir.is_dir():
                continue
                
            # Load config
            config_path = exp_dir / "config.json"
            if not config_path.exists():
                continue
                
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load metrics CSV if available
            metrics_path = exp_dir / "metrics" / "version_0" / "metrics.csv"
            if metrics_path.exists():
                metrics_df = pd.read_csv(metrics_path)
                
                # Extract final metrics
                final_metrics = {
                    'final_loss': metrics_df['train/embedding_viz_loss'].iloc[-1] if 'train/embedding_viz_loss' in metrics_df else np.nan,
                    'final_knn_acc': metrics_df['knn_probe/accuracy'].iloc[-1] if 'knn_probe/accuracy' in metrics_df else np.nan,
                    'final_linear_top1': metrics_df['linear_probe/top1'].iloc[-1] if 'linear_probe/top1' in metrics_df else np.nan,
                    'final_rankme': metrics_df['rankme'].iloc[-1] if 'rankme' in metrics_df else np.nan,
                    'convergence_epoch': self._find_convergence_epoch(metrics_df),
                }
            else:
                final_metrics = {}
            
            # Compute embedding metrics
            embedding_metrics = self._compute_embedding_metrics(exp_dir)
            
            # Combine all data
            result = {**config, **final_metrics, **embedding_metrics}
            results.append(result)
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def _find_convergence_epoch(self, metrics_df: pd.DataFrame, 
                               loss_col: str = 'train/embedding_viz_loss',
                               threshold: float = 0.01) -> int:
        """Find epoch where loss converges (changes less than threshold)."""
        if loss_col not in metrics_df:
            return -1
            
        loss = metrics_df[loss_col].dropna().values
        if len(loss) < 10:
            return -1
            
        # Look for point where loss stabilizes
        for i in range(10, len(loss)):
            recent_std = np.std(loss[i-10:i])
            if recent_std < threshold:
                return i
        
        return len(loss)
    
    def _compute_embedding_metrics(self, exp_dir: Path) -> Dict[str, float]:
        """Compute metrics from saved 2D embeddings."""
        metrics = {}
        
        # Find latest embedding file
        latent_viz_dir = exp_dir / "latent_viz"
        if not latent_viz_dir.exists():
            return metrics
            
        embedding_files = sorted(latent_viz_dir.glob("epoch_*.npz"))
        if not embedding_files:
            return metrics
            
        # Load last epoch embeddings
        last_file = embedding_files[-1]
        data = np.load(last_file)
        coords = data['coordinates']
        labels = data.get('labels', None)
        
        if labels is not None and len(np.unique(labels)) > 1:
            # Compute clustering metrics
            metrics['silhouette_score'] = silhouette_score(coords, labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(coords, labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(coords, labels)
        
        # Compute spread metrics
        metrics['embedding_std'] = np.std(coords)
        metrics['embedding_range'] = np.ptp(coords)
        
        # Compute temporal stability if multiple epochs available
        if len(embedding_files) > 1:
            metrics['temporal_stability'] = self._compute_temporal_stability(embedding_files)
        
        return metrics
    
    def _compute_temporal_stability(self, embedding_files: List[Path]) -> float:
        """Compute stability of embeddings across epochs."""
        if len(embedding_files) < 2:
            return 1.0
            
        # Compare last two epochs
        data1 = np.load(embedding_files[-2])
        data2 = np.load(embedding_files[-1])
        
        coords1 = data1['coordinates']
        coords2 = data2['coordinates']
        
        # Align embeddings (handle potential permutations)
        min_size = min(len(coords1), len(coords2))
        coords1 = coords1[:min_size]
        coords2 = coords2[:min_size]
        
        # Compute correlation between pairwise distances
        from scipy.spatial.distance import pdist
        dist1 = pdist(coords1)
        dist2 = pdist(coords2)
        
        correlation, _ = spearmanr(dist1, dist2)
        return correlation if not np.isnan(correlation) else 0.0
    
    def plot_parameter_effects(self, save_dir: Optional[Path] = None):
        """Create plots showing effect of each parameter on metrics."""
        if self.results_df is None:
            self.load_experiment_results()
        
        if save_dir is None:
            save_dir = self.base_dir / "analysis_plots"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters to analyze
        params = ['queue_length', 'k_neighbors', 'n_negatives', 
                 'update_interval', 'lr', 'warmup_iterations']
        
        # Metrics to plot
        metrics = ['silhouette_score', 'final_knn_acc', 'final_rankme', 
                  'convergence_epoch', 'temporal_stability']
        
        # Create parameter effect plots
        fig, axes = plt.subplots(len(params), len(metrics), 
                                figsize=(20, 24))
        
        for i, param in enumerate(params):
            for j, metric in enumerate(metrics):
                if metric not in self.results_df.columns:
                    continue
                    
                ax = axes[i, j]
                
                # Group by parameter value and compute statistics
                grouped = self.results_df.groupby(param)[metric].agg(['mean', 'std'])
                
                # Plot with error bars
                x = grouped.index
                y = grouped['mean']
                yerr = grouped['std']
                
                ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5)
                ax.set_xlabel(param)
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
                
                if param == 'lr':
                    ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(save_dir / "parameter_effects.png", dpi=150)
        plt.close()
        
        print(f"Saved parameter effect plots to {save_dir}/parameter_effects.png")
    
    def create_heatmaps(self, save_dir: Optional[Path] = None):
        """Create heatmaps showing parameter interactions."""
        if self.results_df is None:
            self.load_experiment_results()
        
        if save_dir is None:
            save_dir = self.base_dir / "analysis_plots"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Key parameter pairs to analyze
        param_pairs = [
            ('k_neighbors', 'n_negatives'),
            ('queue_length', 'k_neighbors'),
            ('update_interval', 'lr'),
        ]
        
        metrics = ['silhouette_score', 'final_knn_acc', 'final_rankme']
        
        for metric in metrics:
            if metric not in self.results_df.columns:
                continue
                
            fig, axes = plt.subplots(1, len(param_pairs), figsize=(15, 5))
            
            for idx, (param1, param2) in enumerate(param_pairs):
                ax = axes[idx]
                
                # Create pivot table
                pivot = self.results_df.pivot_table(
                    values=metric,
                    index=param1,
                    columns=param2,
                    aggfunc='mean'
                )
                
                # Plot heatmap
                sns.heatmap(pivot, annot=True, fmt='.3f', 
                           cmap='RdYlBu_r', ax=ax, cbar_kws={'label': metric})
                ax.set_title(f'{metric}: {param1} vs {param2}')
            
            plt.tight_layout()
            plt.savefig(save_dir / f"heatmap_{metric}.png", dpi=150)
            plt.close()
            
        print(f"Saved heatmaps to {save_dir}/heatmap_*.png")
    
    def compare_with_rankme(self, save_dir: Optional[Path] = None):
        """Compare LatentViz metrics with RankMe scores."""
        if self.results_df is None:
            self.load_experiment_results()
        
        if save_dir is None:
            save_dir = self.base_dir / "analysis_plots"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we have both metrics
        if 'final_rankme' not in self.results_df or 'silhouette_score' not in self.results_df:
            print("Missing required metrics for comparison")
            return
        
        # Create scatter plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RankMe vs Silhouette Score
        ax = axes[0]
        valid_data = self.results_df.dropna(subset=['final_rankme', 'silhouette_score'])
        ax.scatter(valid_data['final_rankme'], valid_data['silhouette_score'], alpha=0.6)
        ax.set_xlabel('RankMe Score')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('RankMe vs Silhouette Score')
        
        # Add correlation
        if len(valid_data) > 2:
            corr, p_val = spearmanr(valid_data['final_rankme'], valid_data['silhouette_score'])
            ax.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {p_val:.3f}',
                   transform=ax.transAxes, verticalalignment='top')
        
        # RankMe vs KNN Accuracy
        ax = axes[1]
        valid_data = self.results_df.dropna(subset=['final_rankme', 'final_knn_acc'])
        ax.scatter(valid_data['final_rankme'], valid_data['final_knn_acc'], alpha=0.6)
        ax.set_xlabel('RankMe Score')
        ax.set_ylabel('KNN Accuracy')
        ax.set_title('RankMe vs KNN Accuracy')
        
        # RankMe vs Convergence
        ax = axes[2]
        valid_data = self.results_df.dropna(subset=['final_rankme', 'convergence_epoch'])
        ax.scatter(valid_data['final_rankme'], valid_data['convergence_epoch'], alpha=0.6)
        ax.set_xlabel('RankMe Score')
        ax.set_ylabel('Convergence Epoch')
        ax.set_title('RankMe vs Convergence Speed')
        
        plt.tight_layout()
        plt.savefig(save_dir / "rankme_comparison.png", dpi=150)
        plt.close()
        
        print(f"Saved RankMe comparison to {save_dir}/rankme_comparison.png")
    
    def generate_paper_table(self, save_path: Optional[Path] = None):
        """Generate LaTeX table for paper with best configurations."""
        if self.results_df is None:
            self.load_experiment_results()
        
        if save_path is None:
            save_path = self.base_dir / "paper_results.tex"
        
        # Find best configurations for each metric
        best_configs = {}
        
        metrics_to_optimize = {
            'silhouette_score': 'max',
            'final_knn_acc': 'max',
            'final_rankme': 'max',
            'convergence_epoch': 'min',
            'davies_bouldin_score': 'min',
        }
        
        for metric, direction in metrics_to_optimize.items():
            if metric not in self.results_df.columns:
                continue
                
            valid_df = self.results_df.dropna(subset=[metric])
            if len(valid_df) == 0:
                continue
                
            if direction == 'max':
                best_idx = valid_df[metric].idxmax()
            else:
                best_idx = valid_df[metric].idxmin()
                
            best_configs[metric] = valid_df.loc[best_idx]
        
        # Create LaTeX table
        latex_lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Best LatentViz configurations for different metrics}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Metric & Queue & k-NN & Neg. & Update & LR & Score \\\\",
            "\\midrule"
        ]
        
        for metric, config in best_configs.items():
            metric_name = metric.replace('_', ' ').title()
            line = (f"{metric_name} & {int(config['queue_length'])} & "
                   f"{int(config['k_neighbors'])} & {int(config['n_negatives'])} & "
                   f"{int(config['update_interval'])} & {config['lr']:.0e} & "
                   f"{config[metric]:.3f} \\\\")
            latex_lines.append(line)
        
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        # Save table
        with open(save_path, 'w') as f:
            f.write('\n'.join(latex_lines))
        
        print(f"Saved LaTeX table to {save_path}")
        
        # Also save as CSV for easier access
        best_df = pd.DataFrame(best_configs).T
        best_df.to_csv(self.base_dir / "best_configurations.csv")
        
        return best_df
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        if self.results_df is None:
            self.load_experiment_results()
        
        report_lines = [
            "=" * 60,
            "LatentViz Parameter Sweep Analysis Report",
            "=" * 60,
            f"Total experiments analyzed: {len(self.results_df)}",
            "",
            "Parameter Ranges Tested:",
            "-" * 30,
        ]
        
        params = ['queue_length', 'k_neighbors', 'n_negatives', 
                 'update_interval', 'lr', 'warmup_iterations']
        
        for param in params:
            if param in self.results_df:
                unique_vals = sorted(self.results_df[param].unique())
                report_lines.append(f"{param}: {unique_vals}")
        
        report_lines.extend([
            "",
            "Best Performing Configurations:",
            "-" * 30,
        ])
        
        # Find top 5 by different metrics
        for metric in ['silhouette_score', 'final_knn_acc', 'final_rankme']:
            if metric not in self.results_df.columns:
                continue
                
            valid_df = self.results_df.dropna(subset=[metric])
            if len(valid_df) == 0:
                continue
                
            top5 = valid_df.nlargest(5, metric)
            report_lines.append(f"\nTop 5 by {metric}:")
            for idx, row in top5.iterrows():
                report_lines.append(
                    f"  {row['exp_id']}: {row[metric]:.4f}"
                )
        
        report_lines.extend([
            "",
            "Key Insights:",
            "-" * 30,
        ])
        
        # Compute correlations
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns
        correlations = []
        
        for param in params:
            if param not in numeric_cols:
                continue
            for metric in ['silhouette_score', 'final_knn_acc', 'final_rankme']:
                if metric not in numeric_cols:
                    continue
                    
                valid_data = self.results_df[[param, metric]].dropna()
                if len(valid_data) > 10:
                    corr, p_val = spearmanr(valid_data[param], valid_data[metric])
                    if abs(corr) > 0.3 and p_val < 0.05:
                        correlations.append((param, metric, corr, p_val))
        
        if correlations:
            report_lines.append("\nSignificant correlations (|ρ| > 0.3, p < 0.05):")
            for param, metric, corr, p_val in correlations:
                report_lines.append(
                    f"  {param} vs {metric}: ρ = {corr:.3f} (p = {p_val:.3e})"
                )
        
        # Save report
        report_path = self.base_dir / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print('\n'.join(report_lines))
        print(f"\nFull report saved to {report_path}")


def main():
    """Run complete analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LatentViz experiment results")
    parser.add_argument("--base-dir", type=str, default="experiments/cifar10_latent_viz_sweep/cifar10",
                       help="Base directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for plots and tables")
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer(Path(args.base_dir))
    
    # Load results
    print("Loading experiment results...")
    df = analyzer.load_experiment_results()
    print(f"Loaded {len(df)} experiments")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Generate all analyses
    print("\nGenerating parameter effect plots...")
    analyzer.plot_parameter_effects(output_dir)
    
    print("\nGenerating interaction heatmaps...")
    analyzer.create_heatmaps(output_dir)
    
    print("\nComparing with RankMe...")
    analyzer.compare_with_rankme(output_dir)
    
    print("\nGenerating paper table...")
    analyzer.generate_paper_table()
    
    print("\nGenerating summary report...")
    analyzer.generate_summary_report()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
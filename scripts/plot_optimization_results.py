"""
Plot Optimization Results

Plotting functions for optimization history from CSV files.

Usage:
    from plot_optimization_results import plot_history, plot_summary

    # Plot single file
    plot_history(["results/history_sigma_5.csv"])

    # Plot multiple files overlaid
    plot_history(["results/history_sigma_4.csv", "results/history_sigma_5.csv"])

    # Plot all files in results folder
    from glob import glob
    plot_history(glob("results/history_sigma_*.csv"))

    # Plot summary
    plot_summary("results/sigma_sweep_summary.csv")
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')


def plot_history(csv_files, output_file=None):
    """
    Plot optimization history from CSV file(s).

    If single file: plots stages separately
    If multiple files: overlays all sigmas
    """

    if isinstance(csv_files, str):
        csv_files = [csv_files]
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    sigmas = sorted(df['sigma'].unique())
    target_entropy = df['target_entropy'].iloc[0]

    if len(sigmas) == 1:
        return _plot_single(df, sigmas[0], target_entropy, output_file)
    else:
        return _plot_multi(df, sigmas, target_entropy, output_file)


def _plot_single(df, sigma, target_entropy, output_file=None):
    """Plot optimization history for a single sigma value."""
    rounds = df['round'].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax0 = axes[0]
    ax0.semilogy(rounds, df['determinant'], 'b.-', markersize=8)
    ax0.set_xlabel('Round')
    ax0.set_ylabel('Determinant')
    ax0.set_title(f'Determinant (sigma={sigma})')
    ax0.set_xticks(rounds)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(rounds, df['gaussian_entropy'], 'b.-', markersize=8)
    ax1.axhline(target_entropy, color='green', linestyle='--', linewidth=2, label=f'Target={target_entropy:.4f}')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('H(Gaussian) [nats]')
    ax1.set_title(f'Gaussian Entropy (sigma={sigma})')
    ax1.set_xticks(rounds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[2]
    ax2.plot(rounds, df['gap'], 'b.-', markersize=8)
    ax2.axhline(0, color='green', linestyle='--', linewidth=2, label='Target (gap=0)')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Gap [nats]')
    ax2.set_title(f'Gap to Target (sigma={sigma})')
    ax2.set_xticks(rounds)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")

    return fig


def _plot_multi(df, sigmas, target_entropy, output_file=None):
    """Plot optimization history for multiple sigma values overlaid."""
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigmas)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, sigma in enumerate(sigmas):
        sigma_df = df[df['sigma'] == sigma]
        rounds = sigma_df['round'].values

        axes[0].semilogy(rounds, sigma_df['determinant'], '.-', color=colors[i], label=f'σ={sigma}', markersize=6, alpha=0.8)
        axes[1].plot(rounds, sigma_df['gaussian_entropy'], '.-', color=colors[i], label=f'σ={sigma}', markersize=6, alpha=0.8)
        axes[2].plot(rounds, sigma_df['gap'], '.-', color=colors[i], label=f'σ={sigma}', markersize=6, alpha=0.8)

    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Determinant')
    axes[0].set_title('Determinant vs Round')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(target_entropy, color='black', linestyle='--', linewidth=2, label=f'Target={target_entropy:.4f}')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('H(Gaussian) [nats]')
    axes[1].set_title('Gaussian Entropy vs Round')
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(0, color='black', linestyle='--', linewidth=2, label='Target (gap=0)')
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('Gap [nats]')
    axes[2].set_title('Gap to Target vs Round')
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Optimization Progress for Different Sigma Values', fontsize=14)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")

    return fig


def plot_summary(summary_csv, output_file=None):
    """Plot summary bar charts comparing different sigma values."""
    df = pd.read_csv(summary_csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sigmas = df['sigma'].values
    x = np.arange(len(sigmas))

    ax0 = axes[0]
    colors = ['green' if g < 0.1 else 'orange' if g < 0.5 else 'red' for g in np.abs(df['gap'])]
    bars = ax0.bar(x, df['gap'], color=colors, alpha=0.7, edgecolor='black')
    ax0.axhline(0, color='black', linestyle='--', linewidth=1)
    ax0.set_xticks(x)
    ax0.set_xticklabels(sigmas)
    ax0.set_xlabel('Sigma')
    ax0.set_ylabel('Gap [nats]')
    ax0.set_title('Final Gap to Target Entropy')
    ax0.grid(True, alpha=0.3, axis='y')

    best_idx = np.argmin(np.abs(df['gap']))
    bars[best_idx].set_edgecolor('blue')
    bars[best_idx].set_linewidth(3)

    ax1 = axes[1]
    ax1.bar(x, df['det_reduction'], color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sigmas)
    ax1.set_xlabel('Sigma')
    ax1.set_ylabel('Reduction Factor')
    ax1.set_title('Determinant Reduction (Initial / Final)')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[2]
    target = df['target_entropy'].iloc[0]
    ax2.bar(x, df['final_entropy'], color='coral', alpha=0.7, edgecolor='black')
    ax2.axhline(target, color='green', linestyle='--', linewidth=2, label=f'Target={target:.4f}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sigmas)
    ax2.set_xlabel('Sigma')
    ax2.set_ylabel('H(Gaussian) [nats]')
    ax2.set_title('Final Gaussian Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Sigma Sweep Summary', fontsize=14)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")

    return fig


if __name__ == "__main__":
    results_dir = Path("../results")

    # Get all history files
    csv_files = sorted(results_dir.glob("history_sigma_*.csv"))

    # Or specify manually:
    # csv_files = [results_dir / "history_sigma_4.0.csv"]

    if csv_files:
        plot_history(
            [str(f) for f in csv_files],
            output_file=results_dir / "optimization_history.png"
        )

        summary_files = list(results_dir.glob("sigma_sweep_summary_*.csv"))
        if summary_files:
            plot_summary(
                str(summary_files[-1]),
                output_file=results_dir / "sigma_sweep_summary.png"
            )
    else:
        print(f"No CSV files found in {results_dir}.")

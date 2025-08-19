#!/usr/bin/env python3
"""
Generate plots from saved inference_loop.py results.
Run this after you have your data and want to create visualizations.

Usage:
    python3 generate_plots_from_results.py results_file.json
    python3 generate_plots_from_results.py --help
"""

import json
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_results(results_path: str) -> Dict:
    """Load results from JSON file."""
    print(f"📁 Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        print(f"✅ Loaded {len(data)} problem results")
        return {"problems": data}
    elif isinstance(data, dict):
        if "problems" in data:
            print(f"✅ Loaded {len(data['problems'])} problem results")
        else:
            print(f"✅ Loaded structured results")
        return data
    else:
        raise ValueError("Unknown results format")

def extract_metrics(results: Dict) -> Dict:
    """Extract key metrics from results for plotting."""
    
    metrics = {
        'adaptation_latencies': [],
        'prob_ratios': [],
        'change_projections': [],
        'embedding_sims_orig': [],
        'embedding_sims_inj': [],
        'injection_points': [],
        'problem_types': [],
        'adaptation_success': [],
        'accuracies': []
    }
    
    problems = results.get('problems', results if isinstance(results, list) else [])
    
    for problem in problems:
        # Extract basic info
        injection_point = problem.get('injection_point', 5)
        problem_type = problem.get('problem_type', 'unknown')
        accuracy = 1 if problem.get('generated_answer') == problem.get('expected_answer') else 0
        
        metrics['injection_points'].append(injection_point)
        metrics['problem_types'].append(problem_type)
        metrics['accuracies'].append(accuracy)
        
        # Extract token-level metrics
        token_states = problem.get('token_states', [])
        if token_states:
            # Get sequences
            prob_ratios = [t.get('prob_ratio', 1.0) for t in token_states]
            change_projs = [t.get('change_projection', 0.0) for t in token_states]
            embed_orig = [t.get('embedding_similarity_original', 0.5) for t in token_states]
            embed_inj = [t.get('embedding_similarity_injected', 0.5) for t in token_states]
            
            metrics['prob_ratios'].append(prob_ratios)
            metrics['change_projections'].append(change_projs)
            metrics['embedding_sims_orig'].append(embed_orig)
            metrics['embedding_sims_inj'].append(embed_inj)
            
            # Find adaptation point (when prob_ratio > 1.0 consistently)
            adaptation_point = -1
            for i in range(len(prob_ratios) - 2):
                if all(r > 1.0 for r in prob_ratios[i:i+3]):  # 3 consecutive
                    adaptation_point = i
                    break
            
            if adaptation_point > injection_point:
                latency = adaptation_point - injection_point
            else:
                latency = -1  # Never adapted
            
            metrics['adaptation_latencies'].append(latency)
            metrics['adaptation_success'].append(1 if latency > 0 else 0)
        else:
            # No token data
            metrics['prob_ratios'].append([])
            metrics['change_projections'].append([])
            metrics['embedding_sims_orig'].append([])
            metrics['embedding_sims_inj'].append([])
            metrics['adaptation_latencies'].append(-1)
            metrics['adaptation_success'].append(0)
    
    return metrics

def plot_adaptation_latency_analysis(metrics: Dict, save_dir: Path):
    """Plot adaptation latency analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Adaptation latency by injection point
    latencies = np.array(metrics['adaptation_latencies'])
    injection_points = np.array(metrics['injection_points'])
    
    # Remove failed adaptations for latency analysis
    valid_mask = latencies >= 0
    valid_latencies = latencies[valid_mask]
    valid_injections = injection_points[valid_mask]
    
    if len(valid_latencies) > 0:
        injection_vals = sorted(set(valid_injections))
        mean_latencies = [np.mean(valid_latencies[valid_injections == inj]) for inj in injection_vals]
        std_latencies = [np.std(valid_latencies[valid_injections == inj]) for inj in injection_vals]
        
        axes[0, 0].bar(injection_vals, mean_latencies, yerr=std_latencies, 
                      capsize=5, alpha=0.7, color='steelblue')
        axes[0, 0].set_xlabel('Injection Point')
        axes[0, 0].set_ylabel('Adaptation Latency (tokens)')
        axes[0, 0].set_title('Mean Adaptation Latency by Injection Point')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No successful adaptations found', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Adaptation Latency (No Data)')
    
    # 2. Success rate by injection point
    injection_vals = sorted(set(injection_points))
    success_rates = []
    for inj in injection_vals:
        mask = injection_points == inj
        success_rate = np.mean(np.array(metrics['adaptation_success'])[mask])
        success_rates.append(success_rate)
    
    axes[0, 1].bar(injection_vals, success_rates, alpha=0.7, color='forestgreen')
    axes[0, 1].set_xlabel('Injection Point')
    axes[0, 1].set_ylabel('Adaptation Success Rate')
    axes[0, 1].set_title('Adaptation Success Rate by Injection Point')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Adaptation latency distribution
    if len(valid_latencies) > 0:
        axes[1, 0].hist(valid_latencies, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(np.mean(valid_latencies), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(valid_latencies):.1f}')
        axes[1, 0].axvline(np.median(valid_latencies), color='blue', linestyle='--', 
                          label=f'Median: {np.median(valid_latencies):.1f}')
        axes[1, 0].set_xlabel('Adaptation Latency (tokens)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Adaptation Latency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No successful adaptations', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Latency Distribution (No Data)')
    
    # 4. Accuracy by problem type
    problem_types = metrics['problem_types']
    accuracies = metrics['accuracies']
    
    type_vals = sorted(set(problem_types))
    type_accuracies = []
    for ptype in type_vals:
        mask = np.array(problem_types) == ptype
        acc = np.mean(np.array(accuracies)[mask])
        type_accuracies.append(acc)
    
    axes[1, 1].bar(type_vals, type_accuracies, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Problem Type')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy by Problem Type')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Adaptation Latency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / "adaptation_latency_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")

def plot_token_evolution_heatmap(metrics: Dict, save_dir: Path, max_problems: int = 50):
    """Plot token evolution as heatmaps."""
    
    prob_ratios = metrics['prob_ratios'][:max_problems]
    change_projs = metrics['change_projections'][:max_problems]
    injection_points = metrics['injection_points'][:max_problems]
    
    # Find max sequence length
    max_len = max(len(seq) for seq in prob_ratios if seq) if prob_ratios else 20
    max_len = min(max_len, 50)  # Cap at 50 tokens for visibility
    
    if not prob_ratios or all(len(seq) == 0 for seq in prob_ratios):
        print("   ⚠️ No token sequences found, skipping heatmap")
        return
    
    # Create matrices
    prob_matrix = np.ones((len(prob_ratios), max_len))
    change_matrix = np.zeros((len(change_projs), max_len))
    
    for i, (prob_seq, change_seq, inj_point) in enumerate(zip(prob_ratios, change_projs, injection_points)):
        if prob_seq:
            prob_len = min(len(prob_seq), max_len)
            prob_matrix[i, :prob_len] = prob_seq[:prob_len]
            
        if change_seq:
            change_len = min(len(change_seq), max_len)
            change_matrix[i, :change_len] = change_seq[:change_len]
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Probability ratios heatmap
    im1 = axes[0].imshow(prob_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=3)
    axes[0].set_title('Probability Ratios Across Problems and Tokens')
    axes[0].set_xlabel('Token Position')
    axes[0].set_ylabel('Problem Index')
    
    # Add injection point markers
    for i, inj_point in enumerate(injection_points):
        if inj_point < max_len:
            axes[0].axvline(inj_point, color='blue', alpha=0.3, linewidth=0.5)
    
    plt.colorbar(im1, ax=axes[0], label='Probability Ratio')
    
    # Change projections heatmap
    im2 = axes[1].imshow(change_matrix, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('Change Vector Projections Across Problems and Tokens')
    axes[1].set_xlabel('Token Position')
    axes[1].set_ylabel('Problem Index')
    
    # Add injection point markers
    for i, inj_point in enumerate(injection_points):
        if inj_point < max_len:
            axes[1].axvline(inj_point, color='blue', alpha=0.3, linewidth=0.5)
    
    plt.colorbar(im2, ax=axes[1], label='Change Projection')
    
    plt.tight_layout()
    
    save_path = save_dir / "token_evolution_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")

def plot_example_trajectories(metrics: Dict, save_dir: Path, num_examples: int = 6):
    """Plot example adaptation trajectories."""
    
    prob_ratios = metrics['prob_ratios']
    change_projs = metrics['change_projections']
    embed_orig = metrics['embedding_sims_orig']
    embed_inj = metrics['embedding_sims_inj']
    injection_points = metrics['injection_points']
    
    # Select diverse examples
    valid_indices = [i for i, seq in enumerate(prob_ratios) if len(seq) > 10]
    
    if len(valid_indices) == 0:
        print("   ⚠️ No valid trajectories found")
        return
    
    # Sample examples
    selected_indices = valid_indices[:num_examples]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, prob_idx in enumerate(selected_indices):
        if idx >= 6:
            break
            
        tokens = list(range(len(prob_ratios[prob_idx])))
        inj_point = injection_points[prob_idx]
        
        ax = axes[idx]
        
        # Plot probability ratios
        ax.plot(tokens, prob_ratios[prob_idx], 'b-', label='Prob Ratio', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=inj_point, color='red', linestyle='--', alpha=0.7, label='Injection')
        
        # Plot change projection
        ax2 = ax.twinx()
        ax2.plot(tokens, change_projs[prob_idx], 'g-', alpha=0.7, label='Change Proj')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_title(f'Problem {prob_idx} (Inj @ {inj_point})')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Probability Ratio', color='blue')
        ax2.set_ylabel('Change Projection', color='green')
        
        # Color adaptation regions
        for i, ratio in enumerate(prob_ratios[prob_idx]):
            if ratio > 1.0:
                ax.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.suptitle('Example Adaptation Trajectories', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / "example_trajectories.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate plots from inference_loop.py results')
    parser.add_argument('results_file', help='Path to results JSON file')
    parser.add_argument('--output-dir', default='analysis_plots_generated', 
                       help='Directory to save plots (default: analysis_plots_generated)')
    parser.add_argument('--max-problems', type=int, default=100,
                       help='Maximum problems to include in plots (default: 100)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERATING PLOTS FROM SAVED RESULTS")
    print("=" * 60)
    
    # Load results
    results = load_results(args.results_file)
    
    # Extract metrics
    print("\n📊 Extracting metrics for plotting...")
    metrics = extract_metrics(results)
    
    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True)
    print(f"\n📁 Saving plots to: {save_dir}")
    
    # Generate plots
    print("\n🎨 Generating plots...")
    
    print("  1. Adaptation latency analysis...")
    plot_adaptation_latency_analysis(metrics, save_dir)
    
    print("  2. Token evolution heatmaps...")
    plot_token_evolution_heatmap(metrics, save_dir, max_problems=args.max_problems)
    
    print("  3. Example trajectories...")
    plot_example_trajectories(metrics, save_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("PLOT GENERATION COMPLETE")
    print("=" * 60)
    
    plots = list(save_dir.glob("*.png"))
    print(f"\n✅ Generated {len(plots)} plots:")
    for plot in sorted(plots):
        size_kb = plot.stat().st_size / 1024
        print(f"   - {plot.name} ({size_kb:.1f} KB)")
    
    print(f"\n📁 All plots saved in: {save_dir}/")
    print("\nYou can now:")
    print("1. View the plots to understand adaptation patterns")
    print("2. Customize this script to generate additional plots")
    print("3. Use the metrics dict to create your own visualizations")

if __name__ == "__main__":
    main()
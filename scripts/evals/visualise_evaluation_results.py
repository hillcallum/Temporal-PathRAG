#!/usr/bin/env python3
"""
Visualise evaluation results from comprehensive baseline comparison
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))


def load_results(results_dir: Path, dataset: str) -> dict:
    """Load all baseline results for a dataset"""
    results = {}
    
    # Load individual baseline results
    for result_file in results_dir.glob(f"*_{dataset}_results.json"):
        baseline_name = result_file.stem.replace(f"_{dataset}_results", "")
        with open(result_file, 'r') as f:
            results[baseline_name] = json.load(f)
            
    # Load comparison summary
    summary_file = results_dir / f"baseline_comparison_{dataset}_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            return results, summary
    
    return results, None


def create_performance_comparison_plot(results: dict, dataset: str, output_dir: Path):
    """Create bar plot comparing performance across baselines"""
    # Extract metrics
    data = []
    for baseline, result in results.items():
        if 'overall_metrics' in result:
            metrics = result['overall_metrics']
            data.append({
                'Baseline': baseline,
                'Exact Match': metrics.get('exact_match', 0),
                'F1 Score': metrics.get('f1_score', 0),
                'Temporal Accuracy': metrics.get('temporal_accuracy', 0)
            })
    
    if not data:
        print(f"No data to plot for {dataset}")
        return
        
    df = pd.DataFrame(data)
    
    # Sort by Exact Match score
    df = df.sort_values('Exact Match', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set positions
    y_pos = np.arange(len(df))
    width = 0.25
    
    # Create bars
    bars1 = ax.barh(y_pos - width, df['Exact Match'], width, label='Exact Match')
    bars2 = ax.barh(y_pos, df['F1 Score'], width, label='F1 Score')
    bars3 = ax.barh(y_pos + width, df['Temporal Accuracy'], width, label='Temporal Accuracy')
    
    # Highlight Temporal PathRAG
    for i, baseline in enumerate(df['Baseline']):
        if 'temporal_pathrag' in baseline.lower():
            for bars in [bars1, bars2, bars3]:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(2)
    
    # Customise plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Baseline'])
    ax.set_xlabel('Score')
    ax.set_title(f'Performance Comparison on {dataset}')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1.0)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"{dataset}_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance comparison to {output_path}")


def create_efficiency_plot(results: dict, dataset: str, output_dir: Path):
    """Create scatter plot of accuracy vs efficiency"""
    data = []
    for baseline, result in results.items():
        if 'overall_metrics' in result and 'efficiency_metrics' in result:
            metrics = result['overall_metrics']
            efficiency = result['efficiency_metrics']
            
            total_time = efficiency.get('avg_retrieval_time', 0) + efficiency.get('avg_reasoning_time', 0)
            if total_time > 0:  # Only include if we have timing data
                data.append({
                    'Baseline': baseline,
                    'Exact Match': metrics.get('exact_match', 0),
                    'Total Time (s)': total_time
                })
    
    if not data:
        print(f"No efficiency data to plot for {dataset}")
        return
        
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    for i, row in df.iterrows():
        color = 'red' if 'temporal_pathrag' in row['Baseline'].lower() else 'blue'
        size = 150 if 'temporal_pathrag' in row['Baseline'].lower() else 100
        ax.scatter(row['Total Time (s)'], row['Exact Match'], 
                  s=size, c=color, alpha=0.7, edgecolors='black')
        ax.annotate(row['Baseline'], 
                   (row['Total Time (s)'], row['Exact Match']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Customise plot
    ax.set_xlabel('Average Time per Question (seconds)')
    ax.set_ylabel('Exact Match Score')
    ax.set_title(f'Accuracy vs Efficiency on {dataset}')
    ax.grid(True, alpha=0.3)
    
    # Add ideal region (high accuracy, low time)
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Good accuracy threshold')
    ax.axvline(x=1.0, color='orange', linestyle='--', alpha=0.3, label='Fast response threshold')
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / f"{dataset}_efficiency_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved efficiency plot to {output_path}")


def create_breakdown_heatmap(results: dict, dataset: str, output_dir: Path):
    """Create heatmap showing performance breakdown by question type"""
    # Collect breakdown data
    breakdown_data = {}
    baselines = []
    
    for baseline, result in results.items():
        if 'breakdown' in result and 'by_type' in result['breakdown']:
            baselines.append(baseline)
            for qtype, metrics in result['breakdown']['by_type'].items():
                if qtype not in breakdown_data:
                    breakdown_data[qtype] = {}
                breakdown_data[qtype][baseline] = metrics.get('exact_match', 0)
    
    if not breakdown_data:
        print(f"No breakdown data to plot for {dataset}")
        return
        
    # Create DataFrame
    df = pd.DataFrame(breakdown_data)
    
    # Sort baselines by average performance
    df['Average'] = df.mean(axis=1)
    df = df.sort_values('Average', ascending=False)
    df = df.drop('Average', axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Exact Match Score'},
                ax=ax)
    
    # Highlight Temporal PathRAG row
    for i, baseline in enumerate(df.index):
        if 'temporal_pathrag' in baseline.lower():
            ax.add_patch(plt.Rectangle((0, i), df.shape[1], 1, 
                                     fill=False, edgecolor='red', lw=3))
    
    # Customise plot
    ax.set_title(f'Performance Breakdown by Question Type on {dataset}')
    ax.set_xlabel('Question Type')
    ax.set_ylabel('Baseline')
    
    plt.tight_layout()
    output_path = output_dir / f"{dataset}_breakdown_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved breakdown heatmap to {output_path}")


def create_combined_summary_table(all_results: dict, output_dir: Path):
    """Create a summary table across all datasets"""
    summary_data = []
    
    for dataset, (results, summary) in all_results.items():
        for baseline, result in results.items():
            if 'overall_metrics' in result:
                metrics = result['overall_metrics']
                row = {
                    'Dataset': dataset,
                    'Baseline': baseline,
                    'Exact Match': metrics.get('exact_match', 0),
                    'F1 Score': metrics.get('f1_score', 0),
                    'Temporal Accuracy': metrics.get('temporal_accuracy', 0),
                    'MRR': metrics.get('mrr', 0),
                    'Hits@1': metrics.get('hits_at_1', 0)
                }
                summary_data.append(row)
    
    if not summary_data:
        print("No data for combined summary")
        return
        
    df = pd.DataFrame(summary_data)
    
    # Create pivot table
    pivot = df.pivot_table(
        values=['Exact Match', 'F1 Score', 'Temporal Accuracy'],
        index='Baseline',
        columns='Dataset',
        aggfunc='mean'
    )
    
    # Calculate average across datasets
    for metric in ['Exact Match', 'F1 Score', 'Temporal Accuracy']:
        if metric in pivot.columns.levels[0]:
            pivot[(metric, 'Average')] = pivot[metric].mean(axis=1)
    
    # Sort by average Exact Match
    pivot = pivot.sort_values(('Exact Match', 'Average'), ascending=False)
    
    # Save as CSV
    csv_path = output_dir / "combined_results_summary.csv"
    pivot.to_csv(csv_path)
    print(f"Saved combined summary to {csv_path}")
    
    # Create visualisation
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot_plot = pivot[['Exact Match']].droplevel(0, axis=1)
    pivot_plot.plot(kind='bar', ax=ax)
    
    ax.set_title('Exact Match Performance Across Datasets')
    ax.set_xlabel('Baseline')
    ax.set_ylabel('Exact Match Score')
    ax.legend(title='Dataset')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "combined_performance_bars.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined performance plot to {plot_path}")


def main():
    """Main function to create all visualisations"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualise Temporal PathRAG evaluation results"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("analysis_results/baseline_comparison"),
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for visualisations"
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Results directory not found: {args.results_dir}")
        print("Run the evaluation first")
        return
        
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results for all datasets
    all_results = {}
    for dataset in ["MultiTQ", "TimeQuestions"]:
        results, summary = load_results(args.results_dir, dataset)
        if results:
            all_results[dataset] = (results, summary)
            
            # Create individual dataset visualisations
            print(f"\nCreating visualisations for {dataset}")
            create_performance_comparison_plot(results, dataset, output_dir)
            create_efficiency_plot(results, dataset, output_dir)
            create_breakdown_heatmap(results, dataset, output_dir)
    
    # Create combined visualisations
    if all_results:
        print("\nCreating combined visualisations")
        create_combined_summary_table(all_results, output_dir)
    
    print(f"\nAll visualisations saved to: {output_dir}")


if __name__ == "__main__":
    main()
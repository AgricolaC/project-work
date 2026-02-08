import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import networkx as nx
from problem import Problem
from src.core.solver import GA_Solver

def set_style():
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

# extract_trip_data removed as per user request (Cost Breakdown/Load Distribution figures removed)
# plot_cost_breakdown removed
# plot_trip_load_distribution removed

def plot_performance_comparison(df, out_path, show=False):
    """Figure 2: Baseline vs GA vs GA+Refine"""
    # Check available columns
    required = ['improvement', 'beta']
    if not all(col in df.columns for col in required):
        print(f"Skipping Fig 2: Missing columns {set(required) - set(df.columns)}")
        return

    plt.figure()
    # Group by beta to show regime impact
    sns.barplot(data=df, x='beta', y='improvement', hue='density', errorbar=None)
    plt.title("Fig 2: Solver Improvement Ratio over Baseline")
    plt.ylabel("Improvement Ratio (Baseline / Solver)")
    plt.xlabel("Beta (Convexity)")
    plt.axhline(1.0, color='red', linestyle='--', label='Baseline')
    plt.legend(title='Density')
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close()

def plot_runtime_scaling(df, out_path, show=False):
    """Figure 4: Runtime Scaling"""
    required = ['n_cities', 'time']
    if not all(col in df.columns for col in required):
        print(f"Skipping Fig 4: Missing columns {set(required) - set(df.columns)}")
        return

    plt.figure()
    sns.lineplot(data=df, x='n_cities', y='time', hue='density', style='beta', markers=True, dashes=False)
    plt.title("Fig 4: Solver Runtime vs Problem Size")
    plt.xlabel("N Cities")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close()

def plot_ablation_heatmap(df_abl, out_path, show=False):
    """Figure 5: Ablation Heatmap"""
    if df_abl is None or df_abl.empty:
        print("Skipping Fig 5: No ablation data.")
        return
        
    required = ['config', 'beta', 'ratio']
    if not all(col in df_abl.columns for col in required):
        print(f"Skipping Fig 5: Missing columns {set(required) - set(df_abl.columns)}")
        return

    # Pivot: Index=Config, Col=Beta, Val=Ratio
    pivot = df_abl.pivot_table(index='config', columns='beta', values='ratio', aggfunc='mean')
    
    # Sort index to put Baseline first, then specific ablations, then minimal
    # Desired order: Baseline, No_*, Fixed_*, Single_*, Constructive, Random
    def sort_key(idx):
        if idx == 'Baseline': return 0
        if idx.startswith('No_'): return 1
        if idx.startswith('Fixed') or idx.startswith('Single'): return 2
        if idx.startswith('Constructive'): return 3
        if idx.startswith('Random'): return 4
        return 5
        
    sorted_index = sorted(pivot.index, key=sort_key)
    pivot = pivot.reindex(sorted_index)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Improvement Ratio'})
    plt.title("Fig 5: Ablation Study - Performance Impact by Component")
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path)
        print(f"Saved {out_path}")
    if show:
        plt.show()
    plt.close()

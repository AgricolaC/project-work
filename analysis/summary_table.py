import pandas as pd
import os

def generate_summary_table(df, out_path=None):
    """
    Generates a summary table grouped by Alpha, Beta, Density.
    Columns: Mean Ratio, Best Ratio, Mean Runtime, N Tested.
    """
    required = ['alpha', 'beta', 'density', 'improvement', 'time', 'n_cities']
    if not all(col in df.columns for col in required):
        print(f"Cannot generate table. Missing: {set(required) - set(df.columns)}")
        return

    # Groupby
    grouped = df.groupby(['alpha', 'beta', 'density', 'n_cities']).agg({
        'improvement': ['mean', 'max'],
        'time': 'mean',
    })
    
    grouped.columns = ['Mean Ratio', 'Best Ratio', 'Mean Time (s)']
    grouped = grouped.round(2)
    
    print("\nSummary Table:")
    print(grouped)
    
    if out_path:
        latex_str = grouped.to_latex(float_format="%.2f")
        with open(out_path, 'w') as f:
            f.write(latex_str)
        print(f"\nSaved LaTeX table to {out_path}")

import argparse
import pandas as pd
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plots, summary_table

def main():
    parser = argparse.ArgumentParser(description="Generate analysis figures and tables.")
    parser.add_argument("--csv", default="results/tables/benchmark_results.csv", help="Path to benchmark results CSV")
    parser.add_argument("--ablation-csv", default="results/tables/ablation_results.csv", help="Path to ablation results CSV")
    parser.add_argument("--out", default="results/figures", help="Output directory for figures")
    parser.add_argument("--format", default="png", choices=['png', 'pdf', 'svg'], help="Output format")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--only", help="Comma-separated list of figures to generate (e.g., fig1,fig3)")
    parser.add_argument("--latex-table", action="store_true", help="Generate summary LaTeX table")
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    plots.set_style()
    
    # Load Data
    try:
        df_bench = pd.read_csv(args.csv)
        print(f"Loaded {len(df_bench)} rows from {args.csv}")
    except FileNotFoundError:
        print(f"Warning: Benchmark CSV not found at {args.csv}")
        df_bench = pd.DataFrame()

    try:
        df_abl = pd.read_csv(args.ablation_csv)
        print(f"Loaded {len(df_abl)} rows from {args.ablation_csv}")
    except FileNotFoundError:
        df_abl = pd.DataFrame()
        
    # Determine what to plot
    # Figures 1 and 3 removed as per request
    to_plot = ['fig2', 'fig4', 'fig5']
    if args.only:
        to_plot = args.only.split(',')
        
    # Generate Figures
    # fig1 removed
        
    if 'fig2' in to_plot and not df_bench.empty:
        plots.plot_performance_comparison(df_bench, os.path.join(args.out, f"fig2_performance.{args.format}"), args.show)
        
    # fig3 removed
        
    if 'fig4' in to_plot and not df_bench.empty:
        plots.plot_runtime_scaling(df_bench, os.path.join(args.out, f"fig4_runtime.{args.format}"), args.show)
        
    if 'fig5' in to_plot and not df_abl.empty:
        plots.plot_ablation_heatmap(df_abl, os.path.join(args.out, f"fig5_ablation.{args.format}"), args.show)
        
    # Summary Table
    if args.latex_table and not df_bench.empty:
        table_path = os.path.join(args.out, "table1_summary.tex")
        summary_table.generate_summary_table(df_bench, table_path)

if __name__ == "__main__":
    main()

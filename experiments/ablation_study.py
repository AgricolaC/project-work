import collections
import csv
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import Problem
from src.core.solver import GA_Solver
from experiments.common import run_solver_pipeline, get_experiment_parser


def run_ablation_study(alphas, betas, densities, seeds, ns, output_dir, debug=False, compare_minimal=False, pop_size=None, max_generations=None):
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Formal Ablation Study - Structured Comparisons")
    print(f"Alphas: {alphas}")
    print(f"Betas: {betas}")
    print(f"Densities: {densities}")
    print(f"Seeds: {seeds}")
    print(f"Ns: {ns}")
    print("="*80)
    
    all_configs = {
        'Baseline': {}, 
        'No_LNS': {'enable_post_lns': False},
        'No_2opt': {'local_search': False},
        'Seeding_Minimal': {'seeding_mode': 'minimal'},
        'Single_Island': {'island_mode': 'single'},
        'Fixed_Mutation': {'adaptive_mutation': False},
        'Constructive_Only': {'max_generations': 0, 'enable_post_lns': False, 'local_search': False},
        'Random_Search': {'seeding': False, 'max_generations': 0, 'enable_post_lns': False, 'local_search': False}
    }

    if compare_minimal:
        configs = {k: all_configs[k] for k in ['Baseline', 'Constructive_Only', 'Random_Search']}
    else:
        configs = all_configs

    # Pre-calculate scenarios
    scenarios = []
    for n_cities in ns:
        for a in alphas:
            for b in betas:
                for d in densities:
                    for s in seeds:
                        scenarios.append((n_cities, a, b, d, s))
                    
    total_runs = len(scenarios) * len(configs)
    print(f"Total Runs: {total_runs} ({len(scenarios)} scenarios x {len(configs)} configs)")
    
    results = []
    global_count = 0
    import time
    start_time = time.time()
    
    for (n_cities, alpha, beta, density, seed) in scenarios:
        p = Problem(n_cities, alpha=alpha, beta=beta, density=density, seed=seed)
        baseline_cost = p.baseline()
        
        for cfg_name, cfg_overrides in configs.items():
            global_count += 1
            print(f"[{global_count}/{total_runs}] {cfg_name:<16} (N={n_cities}, a={alpha}, b={beta}, d={density}, s={seed}) ... ", end="", flush=True)
            
            # Prepare Solver Config
            # Start with default baseline settings
            solver_ablation_config = {
                'seeding': True,
                'seeding_mode': 'full',
                'local_search': True,
                'island_mode': '3-island',
                'adaptive_mutation': True
            }
            
            run_post_lns = True # Default for Baseline
            target_gens = max_generations

            # Apply overrides to dict or specific variables
            # We separate 'runner control' args from 'solver init' args
            # run_solver_pipeline takes 'solver_config' and 'lns_config'
            
            # Mutable copy for this run
            current_solver_abl = solver_ablation_config.copy()

            for k, v in cfg_overrides.items():
                if k == 'enable_post_lns':
                    run_post_lns = v
                elif k == 'max_generations':
                    # Configuration forces specific generation count (e.g. 0)
                    target_gens = v
                else:
                    current_solver_abl[k] = v
                    
            solver_config = {
                'pop_size_per_island': pop_size,
                'max_generations': target_gens,
                'ablation_config': current_solver_abl,
                'seed': seed
            }
            
            lns_config = {'enable': run_post_lns, 'iters': None}

            # Execute
            try:
                res = run_solver_pipeline(p, GA_Solver, solver_config, lns_config)
                final_cost = res['cost']
                runtime = res['runtime']
            except Exception as e:
                print(f"ERROR: {e}")
                final_cost = float('inf')
                runtime = 0.0
                import traceback
                traceback.print_exc()
            
            # Metrics
            if final_cost > 0 and baseline_cost > 0:
                ratio = baseline_cost / final_cost
            else:
                ratio = 0.0
            
            print(f"Ratio: {ratio:.4f}x ({runtime:.2f}s)")
            
            results.append({
                'n_cities': n_cities,
                'alpha': alpha,
                'beta': beta,
                'density': density,
                'seed': seed,
                'config': cfg_name,
                'baseline_cost': baseline_cost,
                'solver_cost': final_cost,
                'ratio': ratio,
                'runtime': runtime
            })

    # Saving Results
    keys = results[0].keys() if results else []
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {csv_path}")
        
    print("\n" + "="*80)
    print("Comparison Report")
    print("="*80)
    
    if not results:
        print("No results to report.")
        return

    # Manual Aggregation
    stats = collections.defaultdict(lambda: {'ratios': [], 'times': []})
    for r in results:
        cfg = r['config']
        stats[cfg]['ratios'].append(r['ratio'])
        stats[cfg]['times'].append(r['runtime'])
        
    print(f"{'Config':<20} | {'Mean Ratio':<10} | {'Median':<10} | {'Best':<10} | {'Std Dev':<10} | {'Mean Time':<10} | {'Med Time':<10}")
    print("-" * 100)
    
    # Store means for baseline comparison
    config_means = {}
    
    sorted_configs = sorted(stats.keys(), key=lambda k: np.mean(stats[k]['ratios']), reverse=True)
    
    import numpy as np
    
    for cfg in sorted_configs:
        ratios = stats[cfg]['ratios']
        times = stats[cfg]['times']
        
        mean_r = np.mean(ratios)
        median_r = np.median(ratios)
        best_r = np.max(ratios)
        std_r = np.std(ratios)
        mean_t = np.mean(times)
        median_t = np.median(times)
        
        config_means[cfg] = mean_r
        
        print(f"{cfg:<20} | {mean_r:<10.3f} | {median_r:<10.3f} | {best_r:<10.3f} | {std_r:<10.3f} | {mean_t:<10.2f} | {median_t:<10.2f}")
        
    # Delta from Baseline
    if 'Baseline' in config_means:
        base_mean = config_means['Baseline']
        print("\nImpact vs Baseline (Mean Ratio Delta):")
        for cfg in sorted_configs:
            if cfg == 'Baseline': continue
            val = config_means[cfg]
            delta = val - base_mean
            if base_mean > 0:
                pct = (delta / base_mean) * 100
                print(f"{cfg:<20}: {delta:+.4f} ({pct:+.2f}%)")
            else:
                print(f"{cfg:<20}: {delta:+.4f} (N/A%)")
            
    total_time = time.time() - start_time
    print(f"\nTotal Study Time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    parser = get_experiment_parser(description="Run Formal Ablation Study")
    parser.add_argument("--debug", action="store_true", help="Run fast debug mode (override grid vars)")
    parser.add_argument("--compare-minimal", action="store_true", help="Run only Baseline vs Minimal baselines")

    args = parser.parse_args()
    
    if args.compare_minimal:
        print("[INFO] Comparison Mode: Baseline vs Minimal Only")

    if args.debug:
        print("[DEBUG] Running reduced grid for verification...")
        # Override args
        # Argparse attributes are mutable
        args.alphas = [0.0001, 1.0, 1000]
        args.betas = [0.1, 1.0, 2, 20]
        args.densities = [0.5]
        args.seeds = [42]
        args.ns = [10]
        args.output_dir = "results_debug"

    run_ablation_study(
        alphas=args.alphas,
        betas=args.betas,
        densities=args.densities,
        seeds=args.seeds,
        ns=args.ns,
        output_dir=args.output_dir,
        debug=args.debug,
        compare_minimal=args.compare_minimal,
        pop_size=args.pop_size,
        max_generations=args.generations
    )

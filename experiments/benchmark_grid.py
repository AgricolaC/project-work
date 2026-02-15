import time
import collections
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from problem import Problem
from src.core.solver import GA_Solver
from experiments.common import run_solver_pipeline, get_experiment_parser


def run_benchmark(alphas, betas, densities, seeds, ns, output_dir, pop_size=None, generations=None):
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Solver vs Baseline Benchmark - Grid Search")
    print(f"Alphas: {alphas}")
    print(f"Betas: {betas}")
    print(f"Densities: {densities}")
    print(f"Seeds: {seeds}")
    print(f"Ns: {ns}")
    print("="*60)
    
    results = []
    
    total_runs = len(alphas) * len(betas) * len(densities) * len(seeds) * len(ns)
    count = 0
    
    for beta in betas:
        for alpha in alphas:
            for density in densities:
                for n_cities in ns:
                    for seed in seeds:
                        count += 1
                        print(f"[{count}/{total_runs}] Running: N={n_cities}, a={alpha}, b={beta}, d={density}, s={seed} ...", end="", flush=True)
                        
                        p = Problem(n_cities, alpha=alpha, beta=beta, density=density, seed=seed)
                        baseline = p.baseline()
                        
                        # Run Solver Pipeline
                        # LNS enabled by default for benchmark
                        lns_config = {'enable': True, 'iters': None} # Default dynamic
                        
                        # Solver kwargs
                        solver_config = {
                            'pop_size_per_island': pop_size,
                            'max_generations': generations,
                            'ablation_config': {'seeding': True, 'local_search': False},
                            'seed': seed
                        }

                        res = run_solver_pipeline(p, GA_Solver, solver_config, lns_config)
                        
                        refined_cost = res['cost']
                        ga_best = res['ga_best']
                        trips = res['trips']
                        runtime = res['runtime']
                        stats = res['diagnostics']
                        solver_ref = res['solver_ref']
                        
                        imp_ratio = baseline / refined_cost if refined_cost > 0 else 0
                        refine_gain = (ga_best - refined_cost) / ga_best * 100 if ga_best > 0 else 0
                        
                        abs_delta = baseline - refined_cost
                        rel_delta = abs_delta / baseline if baseline > 0 else 0.0
                        
                        regime = getattr(solver_ref, 'regime', {'curvature': '?', 'rho_global_bucket': '?'})
                        regime_str = f" | {regime['curvature']} rho={regime.get('rho_global_bucket', '?')}"
                        rho_sol = stats.get('rho_bucket', '?')
                        
                        diag_info = (f"\n    [DIAG] Trips: {stats['n_trips']} AvgLen: {stats['avg_len']:.2f} "
                                     f"RhoSol: {rho_sol}")
                                         
                        print(f" Done. Imp: {imp_ratio:.2f}x (Refine: +{refine_gain:.1f}%) Time: {runtime:.2f}s{regime_str}{diag_info}")
    
                        results.append({
                            'alpha': alpha,
                            'beta': beta,
                            'density': density,
                            'seed': seed,
                            'n_cities': n_cities,
                            'baseline': baseline,
                            'ga_best': ga_best,
                            'abs_delta': abs_delta,
                            'rel_delta': rel_delta,
                            'refined': refined_cost,
                            'improvement': imp_ratio,
                            'refine_gain_pct': refine_gain,
                            'trips': len(trips),
                            'time': runtime
                        })
                    
    # Analysis
    print("\n" + "="*60)
    print("Benchmark Results Summary")
    print("="*60)
    print(f"{'Alpha':<6} {'Beta':<6} {'Dens':<6} | {'Mean Imp':<10} | {'Min Imp':<10} | {'Refine+%':<10}")
    print("-" * 65)
    
    # Group by (alpha, beta, density)
    groups = collections.defaultdict(list)
    for r in results:
        key = (r['alpha'], r['beta'], r['density'])
        groups[key].append(r)
        
    for key, items in sorted(groups.items()):
        alpha, beta, dens = key
        imps = [r['improvement'] for r in items]
        gains = [r['refine_gain_pct'] for r in items]
        
        mean_imp = sum(imps) / len(imps)
        min_imp = min(imps)
        mean_gain = sum(gains) / len(gains)
        
        print(f"{alpha:<6} {beta:<6} {dens:<6} | {mean_imp:<10.2f} | {min_imp:<10.2f} | {mean_gain:<10.2f}")
        
    print("\nWeak Regimes (Ratio < 1.05):")
    weak_found = False
    for key, items in sorted(groups.items()):
        imps = [r['improvement'] for r in items]
        mean_imp = sum(imps) / len(imps)
        if mean_imp < 1.05:
            print(f"Regime {key}: Mean Improvement {mean_imp:.2f}x")
            weak_found = True
            
    if not weak_found:
        print("None! Solver consistently beats baseline.")
        
    # Convex Gain Summary
    print("\n" + "="*60)
    print("Convex/Structure Analysis (Beta >= 2.0)")
    print("="*60)
    
    convex_gains = []
    for r in results:
        if r['beta'] >= 1.0:
            convex_gains.append(r['refine_gain_pct'])
            
    if convex_gains:
        avg_convex_gain = sum(convex_gains) / len(convex_gains)
        fraction_positive = sum(1 for g in convex_gains if g > 0.1) / len(convex_gains)
        print(f"Mean Refine Gain: {avg_convex_gain:.2f}%")
        print(f"Fraction > 0.1%:  {fraction_positive*100:.1f}%")
    else:
        print("No convex regimes tested.")
        
    print("\n" + "="*60)
    print("Top 10 Refine Improvements")
    print("="*60)
    sorted_res = sorted(results, key=lambda x: x['refine_gain_pct'], reverse=True)
    for r in sorted_res[:10]:
        print(f"Alpha={r['alpha']}, Beta={r['beta']}, Dens={r['density']} -> +{r['refine_gain_pct']:.2f}% (Imp: {r['improvement']:.2f}x)")

    # CSV Export
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    with open(csv_path, "w") as f:
        # Added n_cities and time to header
        header = "alpha,beta,density,seed,n_cities,baseline,ga_best,refined,improvement,abs_delta,rel_delta,trips,time,refine_gain_pct\n"
        f.write(header)
        for r in results:
            line = f"{r['alpha']},{r['beta']},{r['density']},{r['seed']},{r['n_cities']},{r['baseline']},{r['ga_best']},{r['refined']},{r['improvement']},{r['abs_delta']},{r['rel_delta']},{r['trips']},{r['time']},{r['refine_gain_pct']}\n"
            f.write(line)
    print(f"\nFull results saved to {csv_path}")

if __name__ == "__main__":
    parser = get_experiment_parser(description="Run Solver Benchmark with Grid Search using argparse for reproducibility.")
    args = parser.parse_args()
    
    run_benchmark(
        alphas=args.alphas,
        betas=args.betas,
        densities=args.densities,
        seeds=args.seeds,
        ns=args.ns,
        output_dir=args.output_dir,
        pop_size=args.pop_size,
        generations=args.generations
    )


import sys
import os

# Add parent directory to path to import problem.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import csv
import logging
import random
import numpy as np
import networkx as nx
from problem import Problem
from src.solver import GA_Solver

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def run_ablation_study(num_cities, density, beta, alpha=1.0, seed=42):
    # Set Seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate Problem (Fixed seed for reproducibility)
    problem = Problem(num_cities, alpha=alpha, beta=beta, density=density, seed=seed)
    
    # Calculate Baseline (Simple separate trips)
    # problem.baseline() usually provides a naive solution
    baseline_cost = problem.baseline()
    
    # Define Configurations
    # Now that we removed chunking/granular/seeding toggles in the solver logic (except seeding),
    # we only compare different SEEDING strategies or just run the ONE correct solver.
    # The user wanted to "Run full pipeline and comparing to baselines".
    # Since "No Chunking" is now the ONLY mode, the old variants don't make sense physically.
    # But we can ablate SEEDING and LOCAL SEARCH.
    
    variants = {
        "Full System":    {'seeding': True, 'local_search': True},
        "No Seeding":     {'seeding': False, 'local_search': True},
        "No LocalSearch": {'seeding': True, 'local_search': False},
        "Minimal":        {'seeding': False, 'local_search': False}
    }
    
    results = []
    
    for name, config in variants.items():
        print(f"Running {name} (N={num_cities}, D={density}, B={beta})...", flush=True)
        
        t0 = time.time()
        
        # Setup Seeds (if enabled)
        initial_seeds = []
        if config['seeding']:
            # 1. Identity
            initial_seeds.append(list(range(1, num_cities)))
            # 2. Cheapest First (Gold)
            golds = nx.get_node_attributes(problem.graph, 'gold')
            sorted_by_gold = sorted(range(1, num_cities), key=lambda x: golds.get(x, 0))
            initial_seeds.append(sorted_by_gold)
            
        try:
            # Init Solver
            # We must use permutations of 1..N-1
            solver = GA_Solver(problem, 
                               pop_size_per_island=24, 
                               max_generations=50,     
                               initial_individuals=initial_seeds, 
                               ablation_config=config)
            
            # Run
            for _ in range(solver.max_generations):
                solver.step_generation()
            
            # Extract Solution
            best_ind = solver.global_best
            
            # Recalculate cost using split_route to be sure (it returns (cost, trips))
            # The inputs to split_route are (permutation, win_scale).
            # We use a large window for final eval to get precision.
            raw_cost, tour_trips = solver.split_route(best_ind.genome, 2.0)
            
            # Final Cost is exactly raw_cost because physics are unified now.
            final_cost = raw_cost
            
            runtime = time.time() - t0
            improvement = baseline_cost / final_cost if final_cost > 0 else 0
            
            results.append({
                'num_cities': num_cities,
                'density': density,
                'beta': beta,
                'variant': name,
                'baseline_cost': baseline_cost,
                'final_cost': final_cost,
                'improvement': improvement,
                'runtime': runtime
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.error(f"Failed {name}: {e}")
            results.append({
                'num_cities': num_cities,
                'density': density,
                'beta': beta,
                'variant': name,
                'error': str(e)
            })
            
    return results

def main():
    experiments = [
        # (N, D, Beta)
        (20, 0.5, 1.0),   
        (20, 0.5, 1.1),   
        (20, 0.1, 2.0),   
        (20, 1.0, 2.0),  
        (20, 0.5, 0.1),    
        (20, 0.9, 0.1),   
        (20, 0.1, 0.1)   
    ]
    all_results = []
    
    for (n, d, b) in experiments:
        print(f"\n--- Experiment: N={n}, D={d}, B={b} ---")
        res = run_ablation_study(n, d, b)
        all_results.extend(res)
        
    # Save
    csv_file = 'src/early_results/ablation_results.csv'
    if all_results:
        all_keys = set()
        for r in all_results:
            all_keys.update(r.keys())
        keys = sorted(list(all_keys))
        
        with open(csv_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
            
    print(f"\n--- Ablation Study Complete. Results saved to {csv_file} ---")
    
    print(f"{'Variant':<15} | {'D':<3} | {'Beta':<5} | {'Imp':<6} | {'Time':<5}")
    print("-" * 55)
    for r in all_results:
        if 'error' not in r:
            print(f"{r['variant']:<15} | {r['density']:<3} | {r['beta']:<5} | {r['improvement']:.4f} | {r['runtime']:.2f}s")
        else:
            print(f"{r['variant']:<15} | {r['density']:<3} | {r['beta']:<5} | ERROR")

if __name__ == "__main__":
    main()

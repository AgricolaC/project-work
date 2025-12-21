
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
from src.utils import granular_path_expansion

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def run_ablation_study(num_cities, density, beta, alpha=1.0, seed=42):
    # Set Seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate Problem (Fixed seed for reproducibility)
    problem = Problem(num_cities, alpha=alpha, beta=beta, density=density, seed=seed)
    
    # Calculate Baseline
    baseline_cost = problem.baseline()
    
    # Define Configurations
    variants = {
        "Full System":    {'seeding': True, 'granular': True, 'chunking': True},
        "No Seeding":     {'seeding': False, 'granular': True, 'chunking': True},
        #"No Granular":    {'seeding': True, 'granular': False, 'chunking': True},
       # "No Chunking":    {'seeding': True, 'granular': True, 'chunking': False}
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
            solver = GA_Solver(problem, 
                               pop_size_per_island=50, # Boost pop slightly
                               max_generations=50,     # Fixed gens for fair comparison
                               initial_individuals=initial_seeds, 
                               ablation_config=config)
            
            # Run
            for _ in range(solver.max_generations):
                solver.step_generation()
            
            # Extract Solution
            best_ind = solver.global_best
            raw_cost, tour_trips = solver.split_route(best_ind.genome, 1.0)
            
            # Flatten tour
            flat_tour = [0]
            for t in tour_trips:
                flat_tour.extend(t)
                flat_tour.append(0)
                
            # Final Cost Calculation
            # 1. Standard (solver reported) - This is the "Fantasy" cost if Granular Physics is enabled
            ga_prediction = raw_cost
            
            # 2. Actual Realized Cost
            # If Beta > 1.0, we must expand the path using the *real* graph.
            # If the graph is Dense, this will find direct edges and pay the Concvex Penalty.
            # If the graph is Sparse, this will find detours and get Granular Gains.
            if beta > 1.0:
                expanded_cost, _ = granular_path_expansion(problem, flat_tour, solver.virtual_map, solver.node_golds)
                final_cost = expanded_cost if expanded_cost is not None else ga_prediction
            else:
                final_cost = ga_prediction # No difference for Beta <= 1.0
            
            runtime = time.time() - t0
            improvement = baseline_cost / final_cost if final_cost > 0 else 0
            prediction_error = final_cost - ga_prediction
            
            results.append({
                'num_cities': num_cities,
                'density': density,
                'beta': beta,
                'variant': name,
                'baseline_cost': baseline_cost,
                'ga_prediction': ga_prediction, 
                'final_cost': final_cost,
                'prediction_error': prediction_error,
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
        (50, 0.5, 1.0),   # Linear Baseline
        #(50, 0.9, 1.0),   # Density Benchmark (Linear)
        (50, 0.5, 1.1),   # Low Convex Stress
        #(50, 0.5, 2.0),   # Medium Convex Stress
        (50, 0.1, 2.0),   # Sparse Stress 
        (50, 1.0, 2.0),  # Dense Stress 
        #(50, 0.5, 4.0),   # High Convex Stress
        (50, 0.5, 0.1),    # Concave Stress
        (50, 0.9, 0.1),   
        (50, 0.1, 0.1)   
        
    ]
    all_results = []
    
    for (n, d, b) in experiments:
        print(f"\n--- Experiment: N={n}, D={d}, B={b} ---")
        res = run_ablation_study(n, d, b)
        all_results.extend(res)
        
    # Save
    csv_file = 'src/results/ablation_results.csv'
    if all_results:
        # Collect all keys
        all_keys = set()
        for r in all_results:
            all_keys.update(r.keys())
        keys = sorted(list(all_keys))
        
        with open(csv_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
            
    print(f"\n--- Ablation Study Complete. Results saved to {csv_file} ---")
    
    # Simple Table Print
    print(f"{'Variant':<15} | {'D':<3} | {'Beta':<5} | {'PredErr':<10} | {'Imp':<6} | {'Time':<5}")
    print("-" * 65)
    for r in all_results:
        if 'error' not in r:
            pred_err = r.get('prediction_error', 0.0)
            print(f"{r['variant']:<15} | {r['density']:<3} | {r['beta']:<5} | {pred_err:<10.2f} | {r['improvement']:.4f} | {r['runtime']:.2f}s")
        else:
            print(f"{r['variant']:<15} | {r['density']:<3} | {r['beta']:<5} | ERROR")

if __name__ == "__main__":
    main()

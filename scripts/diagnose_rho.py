
import sys
import os
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problem import Problem
from src.core.solver import GA_Solver

def diagnose_problem(n_cities=20, alpha=0.1, beta=1.0, seed=42):
    print(f"\n--- Diagnosing Problem (N={n_cities}, a={alpha}, b={beta}, s={seed}) ---")
    
    print("Creating Problem...")
    problem = Problem(n_cities, seed=seed, alpha=alpha, beta=beta)
    print("Creating Solver...")
    solver = GA_Solver(problem, seed=seed)
    print("Solver Created.")
    
    # Global Rho
    print("Computing Rho Global...")
    rho_g = solver.compute_rho_global()
    print("Computed Rho Global.")
    regime = solver.describe_regime()
    print("Described Regime.")
    print(f"Global Rho (Edges + Load Quantiles): {rho_g:.4e}")
    print(f"Regime Bucket: {regime['rho_global_bucket']}")
    print(f"Curvature: {regime['curvature']}")
    
    # Run a quick initialization to get a solution
    print("Initializing Population (if needed)...")
    # Initialize one island
    if not solver.islands:
        # Force initialization if not already done (GA_Solver init does it mostly, but let's check)
        pass 
        
    pop = solver.islands[0].population
    print(f"Population Size: {len(pop)}")
    best_ind = pop[0] # Best seed
    
    # Compute Solution Rho
    # Convert best_ind genome to actions? 
    # Solver uses trips usually.
    # describe_regime(trips) expands them.
    
    print("Computing Solution Rho...")
    # Get trips from best individual
    _, trips = solver.split_route(best_ind.genome, win_scale=None)
    
    sol_regime = solver.describe_regime(trips=trips)
    rho_s = sol_regime.get('rho_solution')
    rho_s_bucket = sol_regime.get('rho_solution_bucket')
    
    print(f"Solution Rho (Pickup-Aware): {rho_s:.4e} ({rho_s_bucket})")
    print(f"Final Bucket Preference: {sol_regime['final_bucket']}")
    
    return {
        'n': n_cities, 'alpha': alpha, 'beta': beta,
        'rho_g': rho_g, 'rho_s': rho_s,
        'bucket_g': regime['rho_global_bucket'],
        'bucket_s': rho_s_bucket
    }

def main():
    scenarios = [
        {'n': 10, 'a': 0.001, 'b': 1.0}, # Distance (Linear)
        {'n': 10, 'a': 1.0, 'b': 1.0},   # Mixed/Penalty (Linear)
        {'n': 10, 'a': 0.01, 'b': 20.0}, # Convex High Penalty
        {'n': 10, 'a': 0.0001, 'b': 0.5}, # Concave
    ]
    
    print("Running Diagnostics...")
    for s in scenarios:
        diagnose_problem(n_cities=s['n'], alpha=s['a'], beta=s['b'])

if __name__ == "__main__":
    main()

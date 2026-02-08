
import sys
import os
import networkx as nx
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problem import Problem
from src.core.solver import GA_Solver
from src.core.utils import evaluate_solution, validate_encoded_solution

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_alignment(seed=42):
    print(f"\n--- Verifying Alignment (Seed={seed}) ---")
    
    # 1. Create Problem (Small, Linear)
    # Linear regime is easiest to verify exact match without float drift from huge powers?
    # Actually, we want to verify robust leg_cost vs evaluate_solution match.
    p = Problem(20, alpha=0.1, beta=2.0, density=0.5, seed=seed)
    
    # 2. Initialize Solver
    solver = GA_Solver(p, pop_size_per_island=10, max_generations=1, seed=seed)
    
    # 3. Create a Synthetic Route (Permutation)
    # Just grab first 5 cities
    perm = [1, 2, 3, 4, 5]
    
    # 4. Run Split
    print("Running Split...")
    cost_split, trips = solver.split_route(perm, win_scale=None)
    print(f"Split Cost: {cost_split:.6f}")
    print(f"Trips: {trips}")
    
    # 5. Expand
    print("Expanding to Action List...")
    actions = solver.expand_solution_to_action_list(trips)
    # Print first few
    print(f"Actions (first 10): {actions[:10]}...")
    
    # 6. Evaluate
    print("Evaluating Action List...")
    cost_eval = evaluate_solution(p, actions)
    print(f"Eval Cost:  {cost_eval:.6f}")
    
    # 7. Compare
    diff = abs(cost_split - cost_eval)
    print(f"Difference: {diff:.9f}")
    
    if diff < 1e-5:
        print("SUCCESS: Costs align.")
    else:
        print("FAILURE: Costs diverge!")
        # Debug: Check if fallback happened?
        # Only strict if expansion follows parent pointers used by L/Sbeta.
        pass

    # 8. Verify Rho Solution
    print("\nVerifying Rho Solution...")
    rho_s = solver.compute_rho_solution(actions)
    print(f"Rho Solution: {rho_s:.6f}")
    
    # Manual check for single edge?
    # Let's take first edge u->v
    u, gu = actions[0]
    v, gv = actions[1]
    # u is 0. v is first step.
    # Load = 0.
    if p.graph.has_edge(u, v):
        d = p.graph[u][v]['dist']
        # rho = ((alpha*w)^beta * Sbeta) / L
        # For single edge: w=0 -> rho=0?
        # Formula: beta*(...)
        # w=0 -> log(w) = -inf -> rho=0.
        # Check logic:
        # if w <= 1e-9: return -float('inf') -> exp -> 0.
        pass
        
    # Check a later edge with load > 0
    found_loaded = False
    for i in range(len(actions)-1):
        u, gu = actions[i]
        v, gv = actions[i+1]
        
        # Calculate load at u
        # This is hard to track without running whole loop. 
        # But compute_rho_solution does it.
        pass

    # 9. Verify Validation
    print("\nVerifying Validation...")
    golds_map = {n: p.graph.nodes[n]['gold'] for n in range(1, 20)}
    total_gold = sum(golds_map.values())
    
    # Should fail because permutation only covers 1..5?
    valid, msg = validate_encoded_solution(actions, 20, total_gold, golds_map)
    print(f"Validation (Partial Permutation): {valid}, {msg}")
    
    if not valid and "Missing Pickups" in msg:
        print("SUCCESS: Detected missing cities.")
    else:
        print("FAILURE: Validation should have failed for missing cities.")

if __name__ == "__main__":
    verify_alignment(seed=42)

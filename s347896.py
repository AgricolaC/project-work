from problem import Problem
from src.solver import GA_Solver
import numpy as np

def solution(p: Problem):
    """
    Solves VRP using a Budget-Constrained Adaptive GA.
    Optimized for safety: Min Budget 3000, Max Budget 12000.
    Includes a 'Panic Mode' for extremely large instances.
    """    
    # 1. Analyze Instance
    N = p.graph.number_of_nodes()
    beta = p.beta
    alpha = p.alpha
    
    # Calculate Density (Safe)
    max_edges = (N * (N - 1) / 2)
    density = p.graph.number_of_edges() / max_edges if max_edges > 0 else 1.0
    density = max(density, 0.05) # Prevent division by zero
    
    # --- SAFETY OVERRIDE FOR EXTREME N ---
    # If N is massive, the O(N^2) distance matrix initialization 
    # will eat most of the runtime. We default to bare minimums to return a valid result.
    if N > 800:
        total_pop = 15
        generations = 5
        pop_size_per_island = 5
        complexity = 999.99
        total_evals = 75
        print(f"_"*60)
        print(f"!!! MASSIVE INSTANCE DETECTED (N={N}) !!!")
        print(f"Defaulting to Safety Mode: Pop={total_pop} Gens={generations}")
        print(f"_"*60)
        
    else:
        # 2. Estimate Complexity (Risk Assessment)
        # Base complexity from problem size (Log scaling prevents explosion)
        size_factor = np.log10(1 + N) * 1.5  
        
        # Beta Penalty: convex costs need more time
        beta_factor = 1.0 + (min(beta, 5.0) * 0.2)
        
        # Alpha Penalty: high alpha forces shorter trips
        alpha_factor = 1.0 + (min(alpha, 5.0) * 0.1)
        
        # Density Penalty: sparse graphs are harder
        density_penalty = 1.0 + (max(0, 0.4 - density) * 1.5)
        
        # Total Complexity Score
        complexity = size_factor * beta_factor * alpha_factor * density_penalty
        
        # 3. Define Computational Budget (Hard Caps)
        # We scale budget with complexity but clamp it strictly between 3k and 12k
        raw_budget = 1500 * complexity
        total_evals = int(np.clip(raw_budget, 3000, 12000))

        # 4. Calculate Aspect Ratio (Width vs. Depth)
        # High Beta -> Needs Depth (Gens)
        # Low Density -> Needs Width (Pop)
        gamma = 0.8  # Density sensitivity
        delta = 1.0  # Beta sensitivity
        base_ratio = 0.6 
        
        ratio = base_ratio * (1.0 / (density ** gamma)) * (1.0 / (beta ** delta))
        
        # 5. Solve for Pop and Gens
        pop_target = np.sqrt(total_evals * ratio)
        
        # Constraints
        num_islands = 3 
        min_pop = 15  # 5 per island minimum
        max_pop = 120 # Cap width to ensure we get enough generations
        
        total_pop = int(np.clip(pop_target, min_pop, max_pop))
        
        # Generations derived from remaining budget
        generations = int(total_evals / total_pop)
        
        # Ensure min generations for convergence
        generations = max(generations, 20) 

        # Distribute to islands
        pop_size_per_island = int(total_pop / num_islands)

        # --- Debug Output ---
        print(f"_"*60)
        print(f"Instance Analysis   | N={N:<3} A={alpha:<4.1f} B={beta:<4.1f} D={density:<4.2f} => C={complexity:.2f}")
        print(f"Adaptive Config     | Budget={total_evals:<5} Pop={total_pop:<3} ({pop_size_per_island}/isl) Gens={generations:<3}")
        print(f"_"*60)

    # Run Solver
    sim = GA_Solver(p, pop_size_per_island=pop_size_per_island, max_generations=generations) 

    # Run Loop
    for _ in range(generations):
        sim.step_generation()
            
    best_ind = sim.global_best
    formatted_solution = []
        
    for trip in best_ind.trips:
        for v_node in trip:
            real_node = sim.virtual_map[v_node]
            gold_amount = sim.node_golds[v_node]
            formatted_solution.append((real_node, gold_amount))
        formatted_solution.append((0, 0))
            
    return formatted_solution
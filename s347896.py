from problem import Problem
from src.solver import GA_Solver
import numpy as np

def solution(p: Problem):
    """
    Solves VRP using a Budget-Constrained Adaptive GA.
    Optimized for safety: Min Budget 3000, Max Budget 12000.
    """    
    # 1. Analyze Instance
    N = p.graph.number_of_nodes()
    beta = p.beta
    alpha = p.alpha
    
    # Calculate Density (Safe)
    max_edges = (N * (N - 1) / 2)
    density = p.graph.number_of_edges() / max_edges if max_edges > 0 else 1.0
    density = max(density, 0.05) # Prevent division by zero
    
    # 2. Estimate Complexity (Risk Assessment)
    # We dampen the multipliers so they don't explode.
    # Logarithmic scaling for N helps prevent runaway budgets on large instances.
    
    # Base complexity from problem size
    size_factor = np.log10(1 + N) * 1.5  # Scales slowly: N=10->1.5, N=500->4.0
    
    # Beta Penalty: convex costs need more time, but we cap the multiplier
    beta_factor = 1.0 + (min(beta, 5.0) * 0.2) # Max 2.0x multiplier
    
    # Alpha Penalty: high alpha forces shorter trips -> harder to optimize
    alpha_factor = 1.0 + (min(alpha, 5.0) * 0.1) # Max 1.5x multiplier
    
    # Density Penalty: sparse graphs are harder to navigate
    density_penalty = 1.0 + (max(0, 0.4 - density) * 1.5) # Max 1.6x multiplier
    
    # Total Complexity Score
    complexity = size_factor * beta_factor * alpha_factor * density_penalty
    
    # 3. Define Computational Budget (Hard Caps)
    # We want a budget between 3000 and 12000 evaluations.
    
    # Base budget scaled by complexity
    raw_budget = 2000 * complexity
    
    # Strict Clamping
    total_evals = int(np.clip(raw_budget, 3000, 12000))

    # 4. Calculate Aspect Ratio (Width vs. Depth)
    # R = Pop / Gens. 
    # High Beta/Alpha -> Needs Depth (Gens) -> Lower R
    # Low Density     -> Needs Width (Pop)  -> Higher R
    
    gamma = 0.8  # Density sensitivity
    delta = 1.0  # Beta sensitivity
    
    base_ratio = 0.6 
    
    # The "Smart" Trade-off Formula
    ratio = base_ratio * (1.0 / (density ** gamma)) * (1.0 / (beta ** delta))
    
    # 5. Solve for Pop and Gens
    # Pop = sqrt(Budget * Ratio)
    pop_target = np.sqrt(total_evals * ratio)
    
    # Constraints
    num_islands = 3 # Optimized to 3 islands
    min_pop = 15 # 5 per island
    max_pop = 99 # Cap width to ensure we get enough generations
    
    total_pop = int(np.clip(pop_target, min_pop, max_pop))
    
    # Generations derived from budget
    generations = int(total_evals / total_pop)
    
    # Ensure min generations for convergence
    generations = max(generations, 30) 

    # Distribute to islands
    pop_size_per_island = int(total_pop / num_islands)

    # --- Debug Output ---
    print(f"\nInstance: N={N}, A={alpha:.1f}, B={beta:.1f}, D={density:.2f}")
    print(f"Complexity Score: {complexity:.2f}")
    print(f"Config: Budget={total_evals} | Pop={total_pop} ({pop_size_per_island}/isl) | Gens={generations}")

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
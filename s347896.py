from problem import Problem
from src.solver import GA_Solver

def solution(p: Problem):
    """
    Solves the problem using a Genetic Algorithm with Optimal Splitting (Route First, Cluster Second).
    """    
    # We pass 'p' (the problem instance) instead of 'self' (which doesn't exist in a function)
    sim = GA_Solver(p, pop_size_per_island=25) 

    # The Simulation class is designed to step one generation at a time.
    generations = 120
    for _ in range(generations):
        sim.step_generation()
            
    # The simulation works with "Virtual Nodes" (splits of gold).
    # We must convert these back to "Real Nodes" for the final answer.
    best_ind = sim.global_best
    formatted_solution = []
        
    # best_ind.trips is a list of lists, e.g., [[v1, v2], [v3, v4]]
    for trip in best_ind.trips:
        for v_node in trip:
            # Map virtual node ID back to real city ID
            real_node = sim.virtual_map[v_node]
            # distinct gold amount for this specific virtual chunk
            gold_amount = sim.node_golds[v_node]
                
            formatted_solution.append((real_node, gold_amount))
            
        # Append "Return to Base" (City 0, 0 Gold) after every trip
        formatted_solution.append((0, 0))
            
    return formatted_solution
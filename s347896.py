import networkx as nx
import numpy as np
from src.core.solver import GA_Solver
from src.core.utils import encode_solution_visits

def solution(problem):
    """
    Solver entry point.
    
    Args:
        problem (Problem): Instance containing graph, alpha, beta, etc. 
        
    Returns:
        list of tuples: [(city_id, gold_collected), ...] 
        Starts and ends with (0,0).
    """
    # 0. Determine Budget
    n_nodes = problem.graph.number_of_nodes()
    budget = GA_Solver.get_budget(n_nodes)
    
    # 1. Initialize Solver
    # Use defaults from GA_Solver which now uses get_budget internally for pop/gens
    solver = GA_Solver(problem)
    
    # 2. Run GA
    for _ in range(solver.max_generations):
        solver.step_generation()
        
    # 3. Refine Best Solution
    #    Use Robust Large Neighborhood Search (LNS)
    best_ind = solver.global_best
    
    # Dynamic Destroy Fraction for Large N
    destroy_frac = (0.15, 0.35)
    if n_nodes > 500:
        destroy_frac = (0.05, 0.15)
        
    refined_ind = solver.improve_with_lns(best_ind.clone(), 
                                          iters=budget['lns_iters'], 
                                          destroy_frac=destroy_frac)
    
    # 4. Final Exact Split (Evaluation)
    #    Ensure output matches exactly optimal split for this permutation
    final_cost, final_trips = solver.split_route(refined_ind.genome, win_scale=None)
    
    # 5. Format Output
    #    Use canonical expansion to ensure edge validity and correct gold semantics
    #    flattened [(node, gold), ...] including pass-throughs
    formatted_solution = solver.expand_solution_to_action_list(final_trips)
    return formatted_solution

def is_valid(problem, path):
    """
    Validates whether the solution path consists of valid edges in the problem graph.
    """
    for (c1, gold1), (c2, gold2) in zip(path, path[1:]):
        yield problem.graph.has_edge(c1,c2)
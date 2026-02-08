
import argparse
import sys
import os
import time
import numpy as np

# Add project root to sys.path
def setup_path():
    # experiments/common.py -> experiments/ -> root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.append(root)

setup_path()

try:
    from src.core.utils import evaluate_solution
    from src.core.solver import GA_Solver
except ImportError:
    pass

def get_experiment_parser(description="Run Solver Experiment"):
    """
    Returns an ArgumentParser with common experiment arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Grid Parameters
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.001,1.0,100], help="List of alpha values")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 1.0, 4], help="List of beta values")
    parser.add_argument("--densities", type=float, nargs="+", default=[0.2,0.8], help="List of density values")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds")
    parser.add_argument("--ns", "--num_cities", dest="ns", type=int, nargs="+", default=[10,20,40], help="List of problem sizes (N)")
    
    # IO
    parser.add_argument("--output_dir", type=str, default="src/results", help="Directory to save CSV results")
    
    # Solver Config
    parser.add_argument("--pop-size", type=int, default=None, help="Fixed population size (default: dynamic)")
    parser.add_argument("--generations", type=int, default=None, help="Fixed generations (default: dynamic)")
    
    return parser

def run_solver_pipeline(problem, solver_class, solver_config=None, lns_config=None):
    """
    Executes the standard solver pipeline:
    1. Initialize Solver
    2. Run Generations
    3. Run LNS (optional)
    4. Exact Split
    5. Expand to Action List
    6. Evaluate

    Args:
        problem: Problem instance
        solver_class: Class to instantiate (usually GA_Solver)
        solver_config: Dict of kwargs for solver __init__ (e.g. pop_size, ablation_config)
        lns_config: Dict with 'enable' (bool) and 'iters' (int)
    
    Returns:
        dict: {
            'cost': float,
            'trips': list,
            'runtime': float,
            'ga_best': float,
            'diagnostics': dict
        }
    """
    solver_config = solver_config or {}
    lns_config = lns_config or {}
    
    t0 = time.perf_counter()
    
    # 1. Initialize
    solver = solver_class(problem, **solver_config)
    
    # 2. Run GA
    # Solver.max_generations is set during init (dynamic or fixed)
    for _ in range(solver.max_generations):
        solver.step_generation()
        
    ga_best_cost = solver.global_best.cost
    current_ind = solver.global_best
    
    # 3. LNS Refinement
    if lns_config.get('enable', True):
        iters = lns_config.get('iters')
        if iters is None:
            # Dynamic default
            budget = solver_class.get_budget(problem.graph.number_of_nodes())
            iters = budget['lns_iters']
            
        current_ind = solver.improve_with_lns(current_ind, iters=iters, destroy_frac=(0.15, 0.35))
        
    # 4. Exact Split (Evaluation)
    # Re-split to be sure we have the trips corresponding to best genome
    _, trips = solver.split_route(current_ind.genome, win_scale=None)
    
    # 5. Expand
    action_list = solver.expand_solution_to_action_list(trips)
    
    # 6. Evaluate
    final_cost = evaluate_solution(problem, action_list)
    
    runtime = time.perf_counter() - t0
    
    # Diagnostics
    stats = solver.get_solution_diagnostics(current_ind)
    
    return {
        'cost': final_cost,
        'trips': trips,
        'runtime': runtime,
        'ga_best': ga_best_cost,
        'diagnostics': stats,
        'solver_ref': solver # Return reference if caller needs regime info etc.
    }

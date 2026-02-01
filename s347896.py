from problem import Problem
from src.solver import GA_Solver
from src.utils import reconstruct_route_for_plotting
import numpy as np

def solution(p: Problem):
    """
    Solves VRP using Giant Tour + Split DP.
    """    
    # Simple, defensible configuration based on N
    N = p.graph.number_of_nodes()
    
    # Scale effort with problem size
    # 200 cities -> ~2-3k evals? 
    # 500 cities -> ~5k evals?
    # User asked for "up to 20000" in total_evals logic before, let's keep it reasonable.
    
    base_evals = 5000
    if N > 200:
        base_evals = 10000
    if N > 1000:
        base_evals = 15000
        
    pop_size = 60
    generations = base_evals // pop_size
    
    print(f"Running GA Solver: N={N}, Pop={pop_size}, Gens={generations}")
    
    # Initial seeds
    seeds = []
    # 1. Identity
    seeds.append(list(range(1, N)))
    
    sim = GA_Solver(p, pop_size_per_island=pop_size // 3, max_generations=generations, initial_individuals=seeds)
    
    for _ in range(generations):
        sim.step_generation()
        
    best_ind = sim.global_best
    best_cost, best_trips = best_ind.cost, best_ind.trips
    
    # Format output for the system
    # Needs to be a list of (node, gold_collected)
    # The system expects the full sequence including intermediate nodes if we want to show them?
    # Or just the visit sequence? 
    # The problem.evaluate check likely iterates the list and checks adjacency.
    # If adjacency check is strict (must be edges), we MUST expand shortest paths.
    
    # Reconstruct visualization route (node sequence)
    full_route_nodes = reconstruct_route_for_plotting(p, best_trips)
    
    # Construct the final expected format: vector of (node_id, gold_collected)
    # Note: "gold_collected" is only non-zero at the specific city visit.
    # Intermediate nodes have 0 gold.
    
    formatted_solution = []
    
    golds = p.graph.nodes(data='gold', default=0)
    
    # We must be careful: reconstruct_route_for_plotting expands trips.
    # A trip 0 -> A -> B -> 0 becomes 0..n..A..n..B..n..0
    # Gold is collected ONLY at A and B.
    # But reconstruct_route_for_plotting just returns IDs. It loses the "target" info.
    # However, gold is fixed per node.
    # BUT: If we pass through C to get to B, we don't pick up C's gold if C was not in the tour?
    # Actually, the problem says "gold[i] > 0 collected when visited".
    # If we visit C as an intermediate node, we technically visit it.
    # BUT the Giant Tour optimizes a permutation.
    # If the shortest path 0->B goes through A, do we pick up A?
    # If A is elsewhere in the tour, we pick it up then.
    # If we pick it up now, we violate "exactly once".
    # Standard VRP on general graphs assumption: Intermediate nodes are "transit". You only "service" the node (collect gold) if it's the target.
    # The 'formatted_solution' format usually implies (node, transaction).
    # If we just output (intermediate, 0), it's fine.
    # Only output (target, gold) when we intend to service it.
    
    # Refined Construction:
    # Iterate trips again.
    
    formatted_solution = [(0, 0)]
    current_node = 0
    
    for trip in best_trips:
        for target in trip:
            if target == current_node: continue
            
            # Move current -> target
            try:
                path = nx.shortest_path(p.graph, current_node, target, weight='dist')
                # Path includes [start, ..., end]
                # We emit intermediates with 0 gold
                for node in path[1:-1]:
                    formatted_solution.append((node, 0))
                
                # Emit target with its gold
                amount = golds[target]
                formatted_solution.append((target, amount))
                current_node = target
                
            except:
                # Disconnected? Jump.
                amount = golds[target]
                formatted_solution.append((target, amount))
                current_node = target
                
        # Return to depot
        if current_node != 0:
            try:
                path = nx.shortest_path(p.graph, current_node, 0, weight='dist')
                for node in path[1:]:
                    formatted_solution.append((node, 0))
                current_node = 0
            except:
                formatted_solution.append((0, 0))
                current_node = 0

    return formatted_solution
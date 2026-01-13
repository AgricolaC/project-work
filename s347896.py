from problem import Problem
from src.solver import GA_Solver
import numpy as np
import networkx as nx

def solution(p: Problem):
    """
    Solves VRP using a Budget-Constrained Adaptive GA.
    Includes Granular Path Expansion to exploit Beta > 1 physics.
    """    
    # 1. Analyze Instance Complexity
    # Optimization: Cache graph to avoid repeated copy overhead
    g = p.graph
    
    N = g.number_of_nodes()
    beta = p.beta
    alpha = p.alpha
    
    max_edges = (N * (N - 1) / 2)
    density = g.number_of_edges() / max_edges if max_edges > 0 else 1.0
    density = max(density, 0.05)
    
    # --- Gravity-Aware Budgeting ---
    # Estimate Effective Problem Size (Virtual Nodes)
    # Replicate log-stable formula from solver.py to predict N_virt where Alpha increases genome size
    # Calculate EXACT averages from the specific instance to ensure Gravity Check matches Reality
    # Gold
    golds = nx.get_node_attributes(g, 'gold')
    # Use only real cities (1 to N-1) for gold stats (node 0 has 0 gold)
    real_golds = [golds[i] for i in range(1, N)]
    avg_gold = np.mean(real_golds) if real_golds else 500.0
    
    # Distance to Depot (Node 0)
    # Since graph might be sparse, we compute it from positions for estimation accuracy
    pos = nx.get_node_attributes(g, 'pos')
    depot_pos = np.array(pos[0])
    dists = []
    for i in range(1, N):
        node_pos = np.array(pos[i])
        d = np.linalg.norm(node_pos - depot_pos)
        dists.append(d)
    avg_dist = np.mean(dists) if dists else 0.5
    
    if beta > 1.001:
        # Match the Analytic Formula from solver.py
        # Estimated factor per node
        term1 = (beta - 1.0) ** (1.0 / beta)
        term2 = alpha * avg_gold
        term3 = avg_dist ** (1.0 - 1.0 / beta)
        
        # Scaling factor matching solver.py (0.5 dampener)
        estimated_split_factor = 0.5 * term1 * term2 * term3
        
        # Helper to keep estimate rational (max 40 splits per node)
        estimated_split_factor = max(1.0, min(estimated_split_factor, 40.0))
    else:
        estimated_split_factor = 1.0

    # N_virt is the "Real" size of the problem for the GA sequence
    N_virt = N * estimated_split_factor

    # Heuristic: Estimate computational complexity based on N_virt
    size_factor = np.log10(1 + N_virt) * 1.5  
    beta_factor = 1.0 + (min(beta, 5.0) * 0.2)
    # Alpha is now implicitly handled by N_virt expansion
    density_penalty = 1.0 + (max(0, 0.4 - density) * 1.5)
    
    complexity = size_factor * beta_factor * density_penalty
    
    # 2. Define Computational Budget
    # Base budget scaled by complexity
    raw_budget = 2000 * complexity

    # RUNTIME PROTECTION:
    # As N_virt grows, the cost per evaluation grows linearly.
    # To prevent runtime explosion, we scale DOWN the number of evaluations as N_virt increases.
    # Logic: Keep roughly constant "total operations" rather than constant "quality".
    # Factor = 200 / (200 + N_virt) => 1.0 at N=0, 0.5 at N=200, 0.2 at N=800
    runtime_scaler = 200.0 / (200.0 + N_virt)
    
    total_evals = raw_budget * runtime_scaler
    # Clamp to safe limits (min 3000 to ensure convergence, max 20000)
    total_evals = int(np.clip(total_evals, 3000, 20000))

    # 3. Determine Population Geometry
    # Adjust aspect ratio (Pop vs Gens) based on Density and Beta
    # Sparse/Concave -> Deeper Search (More Gens); Dense/Convex -> Wider Search (More Pop)
    gamma = 0.8 
    delta = 1.0 
    base_ratio = 0.45 
    
    ratio = base_ratio * (1.0 / (density ** gamma)) * (1.0 / (beta ** delta))
    
    pop_target = np.sqrt(total_evals * ratio)
    
    num_islands = 3 
    total_pop = int(np.clip(pop_target, 21, 150))
    generations = max(int(total_evals / total_pop), 30)

    pop_size_per_island = int(total_pop / num_islands)

    print(f"\n{'='*30} SOLVER CONFIGURATION {'='*30}")
    print(f"[Instance]  N={N:<4} | Alpha={alpha:<4.1f} | Beta={beta:<4.1f} | Density={density:<4.2f}")
    print(f"[Stats]     Avg Dist={avg_dist:<.3f} | Avg Gold={avg_gold:<.1f}")
    print(f"[Estimate]  Complexity Factor: {complexity:.2f}")
    print(f"[Planning]  Est. Virtual Nodes: ~{int(N_virt)} (Split Factor: {estimated_split_factor:.2f}x)")
    print(f"[Budget]    Total Evals: {total_evals} | Pop: {total_pop} ({pop_size_per_island}/isl) | Gens: {generations}")
    print(f"{'='*82}\n")

    # Prepare Seeds (Real Nodes)
    # 1. Identity
    seeds = [list(range(1, N))]
    
    # 2. Cheapest First
    golds = nx.get_node_attributes(g, 'gold')
    # Use array lookup if N is large? No, dict is fine for N=2000 setup cost.
    cheapest = sorted(range(1, N), key=lambda x: golds.get(x, 0))
    seeds.append(cheapest)

    # 3. Nearest Neighbor (Concave only: Beta < 1.0)
    # Concave problems benefit from short edges (TSP-like)
    if beta < 1.0:
        nn_seed = []
        unvisited = set(range(1, N))
        
        # Start from a node (heuristic: pick one with low ID)
        curr = 1 
        nn_seed.append(curr)
        unvisited.remove(curr)
        
        while unvisited:
            # Find nearest unvisited
            candidates = []
            # Optimization: don't iterate all neighbors if dense.
            # But graph might be sparse.
            # For N=2000, looking at all unvisited is O(N). Total O(N^2). Acceptable.
            
            # Use greedy heuristic
            # Just scan unvisited directly if graph is dense or neighbors if sparse
            # but getting neighbors in dense graph is O(N).
            
            # Simple implementation
            # Since p.graph might be dense, neighbors() is large.
            # But we only care about UNVISITED neighbors.
            
            # Simple greedy using sparse graph structure:
            found_neighbor = False
            best_n, best_d = None, float('inf')
            
            for neighbor in g[curr]:
                if neighbor in unvisited:
                    w = g[curr][neighbor].get('dist', float('inf'))
                    if w < best_d:
                        best_d = w
                        best_n = neighbor
                        found_neighbor = True
            
            if found_neighbor:
                curr = best_n
            else:
                # Disconnected or no unvisited neighbors (shouldn't happen in dense)
                curr = list(unvisited)[0]
            
            nn_seed.append(curr)
            unvisited.remove(curr)
            
        seeds.append(nn_seed)

    # Run Solver with Seeds
    sim = GA_Solver(p, pop_size_per_island=pop_size_per_island, max_generations=generations, initial_individuals=seeds, seed=42) 

    for _ in range(generations):
        sim.step_generation()
            
    best_ind = sim.global_best
    
    # --- POST-PROCESSING: Granular Path Expansion ---
    # Use shared utility
    # Convert global best genome (virtual) to tour sequence of trips
    
    # best_ind.trips is populated? 
    # Yes, evaluate_population calls split_route which populates trips.
    # best_ind.trips is list of lists (trips).
    
    formatted_solution = []
    
    # Helper to process a trip
    from src.utils import granular_path_expansion
    
    # Reconstruct full tour sequence with 0 delimiters
    full_tour = [0]
    for tr in best_ind.trips:
        full_tour.extend(tr)
        full_tour.append(0)
    
    # Granular expansion
    cost, expanded_path = granular_path_expansion(p, full_tour, sim.virtual_map, sim.node_golds)
    
    use_granular = beta > 1.0
    
    current_node = 0
    formatted_solution = []
    
    for trip in best_ind.trips:
        for v_node in trip:
            target_node = sim.virtual_map[v_node]
            gold_amount = sim.node_golds[v_node]
            
            if use_granular and current_node != target_node:
                try:
                     path = nx.shortest_path(g, current_node, target_node, weight='dist')
                     for intermediate in path[1:-1]:
                         formatted_solution.append((intermediate, 0))
                except:
                    pass
            
            formatted_solution.append((target_node, gold_amount))
            current_node = target_node
            
        if use_granular and current_node != 0:
             try:
                path = nx.shortest_path(g, current_node, 0, weight='dist')
                for intermediate in path[1:-1]:
                    formatted_solution.append((intermediate, 0))
             except:
                 pass
                 
        formatted_solution.append((0, 0))
        current_node = 0

    return formatted_solution
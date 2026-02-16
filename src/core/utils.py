def validate_encoded_solution(solution, n_nodes, expected_total_gold, golds_by_node=None):
    """
    Validates constraints:
    1. Starts/Ends with (0,0)
    2. GOLD constraints:
       - Each city 1..N-1 must yield its gold exactly once.
       - A city can be visited multiple times, but gold_taken > 0 only once.
       - Pass-through visits must have gold_taken = 0.
       - Depot visits must always have gold_taken = 0.
       - If golds_by_node provided, checks exact gold amount.
    3. Total gold matches expected_total_gold.
    4. All cities 1..N-1 effectively collected.
    """
    if not solution:
        return False, "Empty solution"
    
    if solution[0] != (0, 0):
        return False, "Must start at depot (0,0)"
    if solution[-1] != (0, 0):
        return False, "Must end at depot (0,0)"
        
    cities_picked = set()
    collected_gold = 0.0
    
    # Cities that MUST be collected
    cities_range = set(range(1, n_nodes))
    
    for i, (node, gold) in enumerate(solution):
        # Bounds check
        if not (0 <= node < n_nodes):
            return False, f"Node ID {node} out of bounds [0, {n_nodes-1}] (step {i})"
            
        # Depot check
        if node == 0:
            if gold != 0: return False, f"Depot cannot have gold (step {i})"
            continue
            
        # City check
        if gold > 0:
            if node in cities_picked:
                return False, f"City {node} gold collected multiple times (step {i})"
            
            # Exact Gold Check (if map available)
            if golds_by_node is not None:
                true_gold = golds_by_node.get(node)
                # Float comparison tolerance
                if abs(gold - true_gold) > 1e-6:
                    return False, f"City {node} gold mismatch: taken {gold} != real {true_gold}"
            
            cities_picked.add(node)
            collected_gold += gold
        else:
            # Pass-through (gold=0) is allowed for any node
            pass
            
    # Check if all required cities had gold picked
    # All cities 1..N-1 must be collected exactly once
    if cities_picked != cities_range:
        missing = cities_range - cities_picked
        return False, f"Missing Pickups: {sorted(list(missing))}"
        
    if abs(collected_gold - expected_total_gold) > 1e-6:
        return False, f"Total Gold Mismatch: {collected_gold} != {expected_total_gold}"
        
    return True, "Valid"

def evaluate_solution(problem, solution):
    """
    Evaluates the total cost of a solution using strict edge-by-edge charging.
    
    Args:
        problem: The Problem instance
        solution: List of (node_id, gold_taken) tuples, e.g. [(0,0), (1, 10), (0,0)]
        
    Returns:
        Total cost (float)
    """
    total_cost = 0.0
    current_load = 0.0
    
    # Iterate through transitions (u -> v)
    for i in range(len(solution) - 1):
        u, gold_u = solution[i]
        v, gold_v = solution[i+1]
        
       
        # problem.cost(path, w) -> "dist + (alpha * dist * weight) ** beta" 
        # When moving u->v, weight is the load CARRIED.
        # Load changes at the node.
        # If we pickup at u, we carry (current + gold_u) to v.
        
        if u == 0:
            current_load = 0.0
        else:
            current_load += gold_u
            
        if not problem.graph.has_edge(u, v):
             raise ValueError(f"Invalid transition {u}->{v} in solution: Edge does not exist.")
            
        # Calculate cost for this single edge
        edge_cost = problem.cost([u, v], current_load)
        total_cost += edge_cost
                
    return total_cost
import networkx as nx

def granular_path_expansion(problem, tour_sequence, virtual_map, node_golds):
    """
    Refines a tour by expanding edges into atomic steps based on the physical graph.
    
    This is necessary for Beta > 1.0 (Convex) where:
    Cost(A->B) > Cost(A->...->B). Granular steps minimize the beta penalty.
    
    Args:
        problem: The Problem instance.
        tour_sequence: List of virtual node IDs (visited sequence).
        virtual_map: Dict mapping virtual_node_id -> real_node_id.
        node_golds: Dict mapping virtual_node_id -> gold_amount.
        
    Returns:
        (total_cost, expanded_sequence): The calculated cost and the sequence of real nodes visited.
    """
    if problem.beta <= 1.0:
        return None, None 

    # 1. Segment tour into trips (Depot -> ... -> Depot)
    trips = []
    current_trip = []
    for node in tour_sequence:
        if node == 0:
            if current_trip:
                trips.append(current_trip)
                current_trip = []
        else:
            current_trip.append(node)
    
    final_sequence = [0]
    total_cost = 0.0
    
    for trip in trips:
        current_load = 0.0
        current_pos = 0 
        
        for next_visit_virtual in trip:
            next_visit = virtual_map.get(next_visit_virtual, next_visit_virtual)
            
            if next_visit == current_pos:
                continue

            # 2. Find path between nodes (Real Graph)
            try:
                path_segment = nx.shortest_path(problem.graph, source=current_pos, target=next_visit, weight='dist')
            except nx.NetworkXNoPath:
                path_segment = [current_pos, next_visit] 

            # 3. Calculate Cost for Atomic Steps
            # Path segment: [u, v1, v2...]
            for i in range(len(path_segment)-1):
                u, v = path_segment[i], path_segment[i+1]
                edge_cost = problem.cost([u, v], current_load)
                total_cost += edge_cost
                
            final_sequence.extend(path_segment[1:])
            
            # 4. Update State
            current_pos = next_visit
            current_load += node_golds.get(next_visit_virtual, 0)
            
        # 5. Return to Depot
        try:
            path_back = nx.shortest_path(problem.graph, source=current_pos, target=0, weight='dist')
        except:
            path_back = [current_pos, 0]
            
        for i in range(len(path_back)-1):
            u, v = path_back[i], path_back[i+1]
            edge_cost = problem.cost([u, v], current_load)
            total_cost += edge_cost
            
        final_sequence.extend(path_back[1:])
        
    return total_cost, final_sequence


def calculate_metrics(p, formatted_solution, baseline_cost):
    """Calculates requested mechanics metrics."""
    split_count = 0
    distances = []
    gold_carried = []
    
    current_gold = 0
    
    path_nodes = []
    for node_id, gold_amt in formatted_solution:
         path_nodes.append(node_id)
         
         if node_id == 0:
             split_count += 1
             current_gold = 0 # Reset load at base
         else:
             current_gold += gold_amt
         
         gold_carried.append(float(current_gold))

    g = p.graph
    
    # Calculate distances
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        
        if g.has_edge(u, v):
            dist = g[u][v]['dist']
        else:
            try:
                dist = nx.shortest_path_length(g, u, v, weight='dist')
            except nx.NetworkXNoPath:
                dist = float('inf')
        
        distances.append(float(dist))

    # Approximate GA cost for improvement calc
    ga_cost = 0
    curr_load = 0
    current_node = 0 
    
    for (next_node, gold_amt) in formatted_solution:
        try:
             step_path = nx.shortest_path(g, current_node, next_node, weight='dist')
             step_cost = p.cost(step_path, curr_load)
             ga_cost += step_cost
        except:
             pass 
             
        if next_node == 0:
            curr_load = 0 
        else:
            curr_load += gold_amt 
            
        current_node = next_node

    improvement = baseline_cost / ga_cost if ga_cost > 0 else 0.0

    return split_count, distances, gold_carried, improvement
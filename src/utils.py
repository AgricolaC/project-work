import networkx as nx

def reconstruct_route_for_plotting(problem, tour_trips):
    """
    Reconstructs the full node-by-node path for visualization purposes.
    EXPANDS shortest paths between cities.
    
    Args:
        problem: The Problem instance.
        tour_trips: List of trips, where each trip is a list of city IDs.
                   e.g. [[1, 2], [3, 4, 5]]
    
    Returns:
        full_route: List of city IDs visited in order, including depot (0) and intermediate nodes.
                    e.g. [0, ..., 1, ..., 2, ..., 0, ..., 3, ..., 4, ..., 5, ..., 0]
    """
    g = problem.graph
    full_route = [0]
    
    current_node = 0
    
    for trip in tour_trips:
        # Trip starts from depot (implicitly at current_node=0)
        
        for city in trip:
            if city == current_node: continue
            
            # Find path from current -> city
            try:
                path = nx.shortest_path(g, current_node, city, weight='dist')
                # Append path (excluding start which is already added)
                full_route.extend(path[1:])
                current_node = city
            except nx.NetworkXNoPath:
                # Fallback if disconnected (shouldn't happen in connected graph)
                full_route.append(city)
                current_node = city
                
        # Return to depot
        if current_node != 0:
            try:
                path = nx.shortest_path(g, current_node, 0, weight='dist')
                full_route.extend(path[1:])
                current_node = 0
            except nx.NetworkXNoPath:
                full_route.append(0)
                current_node = 0
                
    return full_route

def calculate_metrics(p, formatted_solution, baseline_cost):
    """Calculates basic metrics from the formatted solution [(node, gold), ...]."""
    split_count = 0
    distances = []
    gold_carried = []
    
    current_gold = 0
    
    path_nodes = []
    for node_id, gold_amt in formatted_solution:
         path_nodes.append(node_id)
         
         if node_id == 0:
             split_count += 1
             current_gold = 0 
         else:
             current_gold += gold_amt
         
         gold_carried.append(float(current_gold))

    g = p.graph
    
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        
        dist = 0
        if g.has_edge(u, v):
            dist = g[u][v]['dist']
        else:
            try:
                dist = nx.shortest_path_length(g, u, v, weight='dist')
            except:
                dist = 0
        distances.append(float(dist))
        
    # Cost improvement is less relevant if baseline is not comparable, 
    # but we keep the signature for compatibility.
    return split_count, distances, gold_carried, 0.0
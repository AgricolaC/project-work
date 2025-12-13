import logging
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from icecream import ic

class Problem:
    _graph: nx.Graph
    _alpha: float
    _beta: float

    def __init__(
        self,
        num_cities: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        density: float = 0.5,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._beta = beta
        cities = rng.random(size=(num_cities, 2))
        cities[0, 0] = cities[0, 1] = 0.5

        self._graph = nx.Graph()
        self._graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0)
        for c in range(1, num_cities):
            self._graph.add_node(c, pos=(cities[c, 0], cities[c, 1]), gold=(1 + 999 * rng.random()))

        tmp = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        d = np.sqrt(np.sum(np.square(tmp), axis=-1))
        for c1, c2 in combinations(range(num_cities), 2):
            if rng.random() < density or c2 == c1 + 1:
                self._graph.add_edge(c1, c2, dist=d[c1, c2])

        assert nx.is_connected(self._graph)

    @property
    def graph(self) -> nx.Graph:
        return nx.Graph(self._graph)

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def cost(self, path, weight):
        dist = nx.path_weight(self._graph, path, weight='dist')
        return dist + (self._alpha * dist * weight) ** self._beta

    def baseline(self):
        cost = 0
        for dest, path in nx.single_source_dijkstra_path(
            self._graph, source=0, weight='weight'
        ).items():
            if dest == 0:
                continue
            logging.debug(
                f"dummy_solution: go to {dest} ({' > '.join(str(n) for n in path)}) -- cost: {self.cost(path, 0):.2f}"
            )
            logging.debug(f"dummy_solution: grab {self._graph.nodes[dest]['gold']:.2f}kg of gold")
            logging.debug(
                f"dummy_solution: return to 0 ({' > '.join(str(n) for n in reversed(path))}) -- cost: {self.cost(path, self._graph.nodes[dest]['gold']):.2f}"
            )
            cost += self.cost(path, 0) + self.cost(path, self._graph.nodes[dest]['gold'])
        logging.info(f"dummy_solution: total cost: {cost:.2f}")
        return cost

    def plot(self):
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self._graph, 'pos')
        size = [100] + [self._graph.nodes[n]['gold'] for n in range(1, len(self._graph))]
        color = ['red'] + ['lightblue'] * (len(self._graph) - 1)
        return nx.draw(self._graph, pos, with_labels=True, node_color=color, node_size=size)
    
    def solution(self):
        """
        Solves the problem using a Genetic Algorithm with Optimal Splitting (Route First, Cluster Second).
        """
        solver = GA_Solver(self)
        best_solution = solver.solve()
        return best_solution


class GA_Solver:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.num_cities_real = problem.graph.number_of_nodes()
        self.alpha = problem.alpha
        self.beta = problem.beta
        
        # Precompute all-pairs shortest path distances
        self.real_dists = dict(nx.all_pairs_dijkstra_path_length(problem.graph, weight='dist'))
        self.real_golds = nx.get_node_attributes(problem.graph, 'gold')
        
        # Virtual Node Mapping
        # If beta > 1, large gold piles are penalized heavily. We split them.
        self.virtual_cities = []
        self.virtual_map = {} # virtual_id -> real_id
        self.node_golds = {}  # virtual_id -> amount
        
        self._create_virtual_cities()
        
        # The list of cities to be visited by the GA (virtual cities)
        self.cities = self.virtual_cities # This is the list of virtual node IDs

    def _create_virtual_cities(self):
        # Heuristic for splitting
        # If beta <= 1, splitting always increases cost (d penalty > 0, convex/linear load)
        # If beta > 1, splitting reduces load penalty.
        
        # Simple heuristic: max_load
        # If beta is high, we want small chunks.
        # Let's verify with a simple check or just use a safe chunk size.
        
        if self.beta > 1.0:
            if self.beta >= 2:
                 chunk_size = 50.0 # Aggressive
            else:
                 chunk_size = 200.0 # Mild
        else:
            chunk_size = 999999.0 # No split
            
        vid_counter = 1
        
        for c in range(1, self.num_cities_real):
            gold = self.real_golds[c]
            if gold > chunk_size and self.beta > 1.0:
                # Split
                num_chunks = int(np.ceil(gold / chunk_size))
                gold_per_chunk = gold / num_chunks
                for _ in range(num_chunks):
                    self.virtual_cities.append(vid_counter)
                    self.virtual_map[vid_counter] = c
                    self.node_golds[vid_counter] = gold_per_chunk
                    vid_counter += 1
            else:
                # Keep as is
                self.virtual_cities.append(vid_counter)
                self.virtual_map[vid_counter] = c
                self.node_golds[vid_counter] = gold
                vid_counter += 1
                
        self.num_nodes_solver = vid_counter # Total nodes + 1 logic?
        
    def get_dist(self, u, v):
        # u, v are virtual IDs. 0 is base (map to 0)
        u_real = 0 if u == 0 else self.virtual_map[u]
        v_real = 0 if v == 0 else self.virtual_map[v]
        return self.real_dists[u_real][v_real]

    def calculate_trip_cost(self, tour_segment):
        """
        Calculates the cost of a single trip: 0 -> c1 -> c2 ... -> ck -> 0
        tour_segment is a list of cities [c1, c2, ..., ck]
        """
        if not tour_segment:
            return 0.0
            
        total_cost = 0.0
        current_load = 0.0
        
        # 0 -> c1
        # Load is 0
        curr = 0
        first = tour_segment[0]
        d = self.get_dist(curr, first)
        # Cost = dist + (alpha * dist * weight) ** beta. Weight=0 => Cost = dist
        total_cost += d 
        
        # Intermediate Legs
        for i in range(len(tour_segment) - 1):
            curr = tour_segment[i]
            next_city = tour_segment[i+1]
            
            # Pick up gold at curr
            current_load += self.node_golds[curr]
            
            d = self.get_dist(curr, next_city)
            penalty = (self.alpha * d * current_load) ** self.beta
            total_cost += d + penalty
            
        # Final Leg: ck -> 0
        last = tour_segment[-1]
        current_load += self.node_golds[last] # Pick up last gold
        d = self.get_dist(last, 0)
        penalty = (self.alpha * d * current_load) ** self.beta
        total_cost += d + penalty
        
        return total_cost

    def split(self, permutation):
        """
        Splits a giant tour (permutation of cities) into optimal trips using the Split algorithm.
        Returns (total_cost, trips)
        where trips is a list of lists: [[c1, c2], [c3], ...]
        """
        n = len(permutation)
        # implicit graph nodes 0 to n
        # V[i] is the cost of shortest path from node 0 to i in the split graph
        V = [float('inf')] * (n + 1)
        V[0] = 0
        P = [-1] * (n + 1) # Predecessor to reconstruct path
        
        # Window size for optimization (limit max trip length)
        # If beta is high, trips will be short. If beta is low, maybe long. 
        # Safe upper bound 50-100? Or dynamic?
        # For now, use N (no window) or a large window to be safe, unless speed is issue.
        # Given project constraints, let's try a reasonable window to speed up O(N^2)
        WINDOW = min(n, 100) 

        for i in range(n):
            if V[i] == float('inf'):
                continue
                
            current_load = 0.0
            cost_accum = 0.0
            
            # Try to extend trip from i
            # The trip will cover permutation[i...j-1]
            # So the split edge is i -> j, representing trip visiting cities P[i]...P[j-1] (1-based in math, 0-based in list)
            
            # We build the trip incrementally to avoid re-calculating full cost
            # Trip: 0 -> p[i] -> p[i+1] ... -> p[j-1] -> 0
            
            # Start of trip: 0 -> p[i]
            u = permutation[i]
            d_0_u = self.get_dist(0, u)
            cost_leg_0 = d_0_u # weight 0
            
            # Initialize running values for the trip extension
            # In the inner loop, we are adding node 'v' at index 'j-1' to the trip
            # To extend efficiently:
            # We need to track the cost of the path SO FAR (0 -> ... -> u -> ... -> v)
            # AND add the return cost (v -> 0) at each step to check the candidate split
            
            # Re-thinking incremental update:
            # It's tricky because adding a node 'v' at the end changes the *return* leg, and the previous return leg is removed.
            # But the *forward* legs don't change their cost because visited nodes' gold is already picked up.
            # So: Trip cost(i, j) = Cost(part before v) + Cost(prev -> v) + Cost(v -> 0)
            
            forward_cost = d_0_u # Cost of 0 -> p[i]
            current_load = 0.0
             
            prev_city = u
            current_load += self.node_golds[prev_city]
            
            # Single node trip i -> i+1 (visiting p[i])
            # Cost = (0->p[i]) + (p[i]->0 with load)
            d_return = self.get_dist(prev_city, 0)
            penalty_return = (self.alpha * d_return * current_load) ** self.beta
            trip_cost = forward_cost + d_return + penalty_return
            
            if V[i] + trip_cost < V[i+1]:
                V[i+1] = V[i] + trip_cost
                P[i+1] = i
                
            # Extend to j
            for j in range(i + 2, min(i + WINDOW + 1, n + 1)):
                new_city = permutation[j-1]
                
                # Add leg prev -> new
                d_seg = self.get_dist(prev_city, new_city)
                penalty_seg = (self.alpha * d_seg * current_load) ** self.beta
                forward_cost += d_seg + penalty_seg
                
                # Pick up gold at new
                current_load += self.node_golds[new_city]
                prev_city = new_city
                
                # Return leg new -> 0
                d_return = self.get_dist(new_city, 0)
                penalty_return = (self.alpha * d_return * current_load) ** self.beta
                
                total_trip_cost = forward_cost + d_return + penalty_return
                
                if V[i] + total_trip_cost < V[j]:
                    V[j] = V[i] + total_trip_cost
                    P[j] = i
        
        # Reconstruct solution
        trips = []
        curr = n
        while curr > 0:
            prev = P[curr]
            trips.append(permutation[prev:curr])
            curr = prev
        trips.reverse()
        
        return V[n], trips

    def solve(self):
        import random
        
        POP_SIZE = 50
        GENERATIONS = 200 # Adjustable
        
        # Check time limit or simple iter count
        
        # Initialize Population
        population = []  
        # 1. Smart Initialization:
        # Sort virtual cities by distance to base (descending)
        # This encourages trips that go 0 -> Far -> Near -> 0
        # Minimizing the final leg distance where load is heaviest.
        try:
             # Virtual node to Real node mapping for distance lookup
             # Dist to 0 is self.get_dist(v_node, 0)
             sorted_perm = sorted(self.cities, key=lambda c: self.get_dist(c, 0), reverse=True)
             cost, trips = self.split(sorted_perm)
             population.append({'perm': sorted_perm, 'cost': cost, 'trips': trips})
        except Exception as e:
             logging.warning(f"Smart init failed: {e}")
        
        # 2. Random Initialization for the rest
        while len(population) < POP_SIZE:
            perm = self.cities[:]
            random.shuffle(perm)
            cost, trips = self.split(perm)
            population.append({'perm': perm, 'cost': cost, 'trips': trips})
        
        population.sort(key=lambda x: x['cost'])
        best_sol = population[0]
        
        initial_cost = best_sol['cost']
        logging.info(f"Initial best cost: {initial_cost:.2f}")

        # Evolution
        for gen in range(GENERATIONS):
            new_pop = []
            
            # Elitism
            new_pop.append(population[0])
            new_pop.append(population[1])
            
            while len(new_pop) < POP_SIZE:
                # Tournament Selection
                p1 = min(random.sample(population, 3), key=lambda x: x['cost'])
                p2 = min(random.sample(population, 3), key=lambda x: x['cost'])
                
                # Crossover (OX1 - Order Crossover)
                parent1, parent2 = p1['perm'], p2['perm']
                size = len(parent1)
                a, b = sorted(random.sample(range(size), 2))
                child_perm = [None] * size
                child_perm[a:b+1] = parent1[a:b+1]
                
                # Fill remaining
                current_pos = (b + 1) % size
                p2_pos = (b + 1) % size
                while None in child_perm:
                    if parent2[p2_pos] not in child_perm[a:b+1]:
                        child_perm[current_pos] = parent2[p2_pos]
                        current_pos = (current_pos + 1) % size
                    p2_pos = (p2_pos + 1) % size
                    
                # Mutation
                if random.random() < 0.2: # Mutation rate
                    # Swap
                    i, j = random.sample(range(size), 2)
                    child_perm[i], child_perm[j] = child_perm[j], child_perm[i]
                
                if random.random() < 0.05: # Inversion mutation
                     i, j = sorted(random.sample(range(size), 2))
                     child_perm[i:j+1] = reversed(child_perm[i:j+1])
                
                # Evaluate
                cost, trips = self.split(child_perm)
                new_pop.append({'perm': child_perm, 'cost': cost, 'trips': trips})
            
            population = new_pop
            population.sort(key=lambda x: x['cost'])
            
            if population[0]['cost'] < best_sol['cost']:
                best_sol = population[0]
                logging.info(f"Gen {gen}: New best cost {best_sol['cost']:.2f}")
                
        # Format output
        # [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
        # Actually formatted as trips returning to 0
        # Trips: [[c1, c2], [c3]]
        # Output: (c1, g1), (c2, g2), (0, 0), (c3, g3), (0, 0)
        formatted_sol = []
        for trip in best_sol['trips']:
            for city in trip:
                # Map virtual ID to Real ID and use partial gold amount
                real_city = self.virtual_map[city]
                gold_amount = self.node_golds[city]
                formatted_sol.append((real_city, gold_amount))
            formatted_sol.append((0, 0)) # Return to base
            
        return formatted_sol

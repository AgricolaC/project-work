import logging
import time
import numpy as np
import networkx as nx

# Logging config
logging.basicConfig(level=logging.INFO, format='%(message)s')

ISLAND_PRIORS = {
    # The "Finisher": High precision, gentle structural repair (Inversion)
    'Exploit': {
        'mut_rate': (0.01, 0.10),      # Very low mutation to preserve elite traits
        'win_scale': (1.5, 2.5),       # MAX window: Spend CPU here to find perfect splits
        'mut_mix': [0.1, 0.8, 0.1]     # 80% Inversion (2-opt) to untangle crossings
    },
    
    # The "Bridge": Standard genetic mixing
    'Balanced': {
        'mut_rate': (0.12, 0.26),      # Moderate mutation
        'win_scale': (0.8, 1.2),       # Standard window (1.0)
        'mut_mix': [0.3, 0.4, 0.3]     # Balanced mix
    },
    
    # The "Scout": Absorbs the Chaos role. Fast, disruptive, low-precision.
    'Explore': {
        'mut_rate': (0.35, 0.80),      # High mutation (approaching Chaos levels)
        'win_scale': (0.3, 0.6),       # TINY window: Very fast, approximate evaluation
        'mut_mix': [0.1, 0.1, 0.8]     # 80% Scramble: Massive structural disruption
    }
}

class Individual:
    """
    Represents a candidate solution in the population.
    Genome is a PERMUTATION of real city IDs (1..N-1).
    """
    def __init__(self, genome, params, problem_context=None):
        self.genome = genome 
        self.params = params 
        self.cost = float('inf')
        self.trips = []

    def clone(self):
        new_params = self.params.copy()
        new_params['mut_mix'] = list(self.params['mut_mix'])
        ind = Individual(list(self.genome), new_params)
        ind.cost = self.cost
        ind.trips = self.trips
        return ind

class Island:
    """
    An isolated population evolving with specific strategic priors.
    """
    def __init__(self, name, role, pop_size, simulation_ref):
        self.name = name
        self.role = role
        self.pop_size = pop_size
        self.sim = simulation_ref
        self.population = []
        self.priors = ISLAND_PRIORS[role].copy()
        
        # Concave Logic: Boost Mutation for Scramble (Explore)
        if self.sim.problem.beta < 1.0 and self.sim.problem.alpha < 3.0 and role == 'Explore':
            # Increase average mutation rate
            old_min, old_max = self.priors['mut_rate']
            self.priors['mut_rate'] = (min(0.9, old_min * 1.5), min(0.95, old_max * 1.5))
        
    def initialize(self, seeds=None):
        self.population = []
        # Inject Seeds if provided
        if seeds:
            for seed_genome in seeds:
                if len(self.population) < self.pop_size:
                    self.population.append(Individual(list(seed_genome), self._sample_params()))
                
        # Internal Heuristics
        heuristics_enabled = self.sim.ablation_config.get('seeding', True)
            
        if heuristics_enabled:
            heuristics = {
                'Exploit': ['nearest_neighbor', 'geometric', 'far_first'],
                'Balanced': ['cheapest_first', 'far_first']
            }.get(self.role, [])
        else:
            heuristics = []

        num_smart = max(1, int(self.pop_size * 0.2 / max(1, len(heuristics)))) if heuristics else 0
        
        for h_name in heuristics:
            for _ in range(num_smart):
                genome = self._generate_smart_genome(h_name)
                self.population.append(Individual(genome, self._sample_params()))

        # Fill remainder with random individuals
        while len(self.population) < self.pop_size:
            rng_genome = list(self.sim.cities)
            self.sim.rng.shuffle(rng_genome)
            self.population.append(Individual(rng_genome, self._sample_params()))
        
        self.evaluate_population()

    def _generate_smart_genome(self, strategy):
        cities = list(self.sim.cities)
        
        if strategy == 'far_first':
            # Sort by distance from depot (descending)
            return sorted(cities, key=lambda c: self.sim.dist_matrix[0, c], reverse=True)
            
        elif strategy == 'cheapest_first':
             # Sort by gold amount (ascending)
            return sorted(cities, key=lambda c: self.sim.real_golds[c])
            
        elif strategy == 'nearest_neighbor':
            # Greedy NN with candidate capping
            genome = []
            curr = 0
            unvisited = set(cities)
            while unvisited:
                candidates = unvisited
                if len(unvisited) > 200:
                    candidates = set(list(unvisited)[:200]) 
                
                # Find nearest to curr
                nn = min(candidates, key=lambda c: self.sim.dist_matrix[curr, c])
                genome.append(nn)
                unvisited.remove(nn)
                curr = nn
            return genome
            
        elif strategy == 'geometric':
            # Polar angle sort appropriate for VRP
            return sorted(cities, key=lambda c: self._get_angle(c))
            
        return cities

    def _get_angle(self, city_id):
        # We need positions for geometric sort
        # If graph has no positions, return 0
        if not hasattr(self.sim, 'node_positions'):
             return 0
        pos = self.sim.node_positions[city_id]
        return np.arctan2(pos[1] - 0.5, pos[0] - 0.5)

    def evaluate_population(self):
        for ind in self.population:
            if ind.cost == float('inf'):
                ind.cost, ind.trips = self.sim.split_route(ind.genome, ind.params['win_scale'])
        self.sort_pop()

    def _sample_params(self):
        return {
            'mut_rate': self.sim.rng.uniform(*self.priors['mut_rate']),
            'win_scale': self.sim.rng.uniform(*self.priors['win_scale']),
            'mut_mix': list(self.priors['mut_mix'])
        }

    def sort_pop(self):
        self.population.sort(key=lambda x: x.cost)

    def evolve_step(self):
        new_pop = self.population[:2]
        
        while len(new_pop) < self.pop_size:
            p1 = self._tournament()
            p2 = self._tournament()
            
            child_genome = self._crossover_genome(p1.genome, p2.genome)
            child_params = self._crossover_params(p1.params, p2.params)
            
            child_params = self._mutate_params(child_params)
            
            if self.sim.rng.random() < child_params['mut_rate']:
                child_genome = self._mutate_genome(child_genome, child_params['mut_mix'])
                
            if self.role == 'Exploit' and self.sim.ablation_config.get('local_search', True):
                child_genome = self._local_search_2opt(child_genome)
                
            child = Individual(child_genome, child_params, self.sim)
            new_pop.append(child)
            
        self.population = new_pop
        self.evaluate_population()

        # Diversity Check: Island Catastrophe
        best_cost = self.population[0].cost
        median_cost = self.population[len(self.population)//2].cost
        
        if abs(median_cost - best_cost) < 1e-4: 
            # Trigger Catastrophe (keep elite, reset others)
            survivors = [self.population[0]]
            
            while len(survivors) < self.pop_size:
                rng_genome = list(self.sim.cities)
                self.sim.rng.shuffle(rng_genome)
                survivors.append(Individual(rng_genome, self._sample_params()))
                
            self.population = survivors
            self.evaluate_population()

    def _tournament(self, k=3):
        candidates = [self.population[i] for i in self.sim.rng.integers(0, len(self.population), k)]
        return min(candidates, key=lambda x: x.cost)

    def _crossover_genome(self, p1, p2):
        # Ordered Crossover (OX1) or similar permutation-preserving operator
        size = len(p1)
        # Random slice
        a, b = sorted(self.sim.rng.choice(range(size), 2, replace=False))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        
        child_set = set(child[a:b+1])
        curr = (b + 1) % size
        p2_idx = (b + 1) % size
        
        count = size - (b - a + 1) 
        
        while count > 0:
            candidate = p2[p2_idx]
            if candidate not in child_set:
                child[curr] = candidate
                curr = (curr + 1) % size
                count -= 1
            p2_idx = (p2_idx + 1) % size
        return child

    def _crossover_params(self, p1, p2):
        new_params = {}
        alpha = self.sim.rng.random()
        for k in ['mut_rate', 'win_scale']:
            new_params[k] = alpha * p1[k] + (1 - alpha) * p2[k]
        
        mix1 = np.array(p1['mut_mix'])
        mix2 = np.array(p2['mut_mix'])
        new_mix = alpha * mix1 + (1 - alpha) * mix2
        new_params['mut_mix'] = list(new_mix / new_mix.sum())
        return new_params

    def _mutate_params(self, params):
        tau = 0.1
        params['mut_rate'] = params['mut_rate'] * np.exp(tau * self.sim.rng.normal())
        params['mut_rate'] = np.clip(params['mut_rate'], 0.05, 0.95)
        
        params['win_scale'] = params['win_scale'] * np.exp(tau * self.sim.rng.normal())
        params['win_scale'] = np.clip(params['win_scale'], 0.3, 2.0)
        
        mix = np.array(params['mut_mix'])
        noise = self.sim.rng.normal(0, 0.1, size=len(mix))
        mix = np.abs(mix + noise)
        params['mut_mix'] = list(mix / mix.sum())
        
        return params

    def _mutate_genome(self, genome, mix):
        op = self.sim.rng.choice(['swap', 'inv', 'scramble'], p=mix)
        size = len(genome)
        
        if op == 'swap':
            i, j = self.sim.rng.choice(range(size), 2, replace=False)
            genome[i], genome[j] = genome[j], genome[i]
        elif op == 'inv':
            i, j = sorted(self.sim.rng.choice(range(size), 2, replace=False))
            genome[i:j+1] = genome[i:j+1][::-1]
        elif op == 'scramble':
            i, j = sorted(self.sim.rng.choice(range(size), 2, replace=False))
            sub = genome[i:j+1]
            self.sim.rng.shuffle(sub)
            genome[i:j+1] = sub
        return genome

    def _local_search_2opt(self, genome):
        """
        Steepest Ascent 2-opt Local Search proxying 'Macro Distance'.
        Optimizes the permutation based on pure distances between cities.
        (Split DP handles the returns, so minimizing giant tour length is a good proxy).
        """
        size = len(genome)
        improved = True
        steps = 0
        max_steps = 50 
        
        while improved and steps < max_steps:
            improved = False
            # Stochastic 2-opt
            for _ in range(20): 
                i, j = sorted(self.sim.rng.choice(range(size), 2, replace=False))
                if j == i + 1: continue 
                
                # Nodes:
                # prev -> u -> ... -> v -> next
                # Reconnect: prev -> v -> ... -> u -> next
                
                u, v = genome[i], genome[j]
                
                # Predecessors/Successors in the tour check
                prev = genome[i-1] if i > 0 else 0
                next_ = genome[j+1] if j+1 < size else 0
                
                d_old = self.sim.dist_matrix[prev, u] + self.sim.dist_matrix[v, next_]
                d_new = self.sim.dist_matrix[prev, v] + self.sim.dist_matrix[u, next_]
                
                if d_new < d_old - 1e-6:
                    genome[i:j+1] = genome[i:j+1][::-1]
                    improved = True
                    steps += 1
                    break
        return genome

class GA_Solver:
    """
    Refactored solver using Giant Tour + Split DP.
    NO CHUNKING. NO VIRTUAL NODES. STRICT ATOMIC PICKUP.
    """
    def __init__(self, problem, pop_size_per_island=30, max_generations=100, initial_individuals=None, ablation_config=None, seed=42):
        self.problem = problem
        self.pop_size = pop_size_per_island
        self.max_generations = max_generations
        self.initial_individuals = initial_individuals or []
        self.rng = np.random.default_rng(seed)
        
        # Ablation Config
        self.ablation_config = {
            'seeding': True,
            'local_search': True
        }
        if ablation_config:
            self.ablation_config.update(ablation_config)
            
        self.cached_graph = problem.graph

        # Precompute APSP (All-Pairs Shortest Path)
        n_nodes = self.cached_graph.number_of_nodes()
        self.dist_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Check if we can use Euclidean optimization
        num_edges = problem.graph.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1) // 2
        
        # Positions for heuristics
        self.node_positions = nx.get_node_attributes(problem.graph, 'pos')

        if num_edges == max_edges and n_nodes > 100: # Dense/Complete
            # Vectorized Euclidean
            coords = np.array([self.node_positions[i] for i in range(n_nodes)])
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            self.dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        else:
            # Sparse - Dijkstra
            # nx.all_pairs_dijkstra_path_length is reliable
            raw_dists = dict(nx.all_pairs_dijkstra_path_length(problem.graph, weight='dist'))
            for u in range(n_nodes):
                for v in range(n_nodes):
                    self.dist_matrix[u, v] = raw_dists[u].get(v, float('inf'))

        # Cache Golds
        golds_map = nx.get_node_attributes(problem.graph, 'gold')
        # Ensure array access
        self.real_golds = np.zeros(n_nodes, dtype=np.float32)
        for i, g in golds_map.items():
            self.real_golds[i] = g

        # Define Cities (1..N-1)
        self.cities = list(range(1, n_nodes))
        self.win_base = 20 # Window for Split DP
        
        # Configure Islands
        self.islands = []
        for i, role in enumerate(['Exploit', 'Balanced', 'Explore']):
            isl = Island(role, role, self.pop_size, self)
            
            seeds = None
            if self.ablation_config['seeding'] and i == 0: 
                seeds = self.initial_individuals
            
            isl.initialize(seeds=seeds)
            self.islands.append(isl)

        self.global_best = self.islands[0].population[0]
        self.generation_count = 0

    def get_dist(self, u, v):
        return self.dist_matrix[u, v]

    def split_route(self, permutation, win_scale):
        """
        Optimal Split DP for VRP.
        Decides where to insert returns to depot (0).
        
        Physics:
        - Atomic pickup: Load increases by gold[v] upon leaving v.
        - Macro-leg cost: cost(u, v, w) = d + (alpha * d * w)^beta
        """
        n = len(permutation)
        win = int(self.win_base * win_scale)
        win = max(5, win)
        
        # V[i] = min cost to service first i cities in permutation
        # P[j] = start index of the trip that ENDS at j
        # Indices in V correspond to number of cities serviced.
        # V[0] = 0 (0 cities serviced)
        # V[n] = total cost
        
        V = [float('inf')] * (n + 1)
        V[0] = 0.0
        P = [-1] * (n + 1) 
        
        # Optimization: Localize access
        d_mat = self.dist_matrix
        golds = self.real_golds
        depot = 0
        
        alpha = self.problem.alpha
        beta = self.problem.beta
        
        def cost_fn(dist, w):
            # Optimized cost calculation
            if alpha == 0: return dist
            return dist + (alpha * dist * w) ** beta

        for i in range(n):
            if V[i] == float('inf'): continue
            
            # Start a new trip from depot to permutation[i]
            # Trip serves cities permutation[i] ... permutation[j-1]
            
            # Leg 1: Depot -> permutation[i]
            u = permutation[i]
            d_out = d_mat[depot, u]
            current_trip_cost = cost_fn(d_out, 0)
            
            current_load = golds[u]
            prev = u
            
            # Try extending the trip up to window limit
            limit = min(i + win + 1, n)
            
            for j in range(i + 1, limit + 1):
                # Currently at 'prev' (which is permutation[j-2] roughly, or u if j=i+1)
                # Trip ends at permutation[j-1].
                
                # Option 1: CLOSE TRIP here. (prev -> depot)
                d_in = d_mat[prev, depot]
                return_cost = cost_fn(d_in, current_load)
                
                total = V[i] + current_trip_cost + return_cost
                
                if total < V[j]:
                    V[j] = total
                    P[j] = i
                    
                # Option 2: EXTEND TRIP to next city (if available)
                # Next city is permutation[j] (shifted because j is 1-based index limit)
                # Wait, j is the index in V. V[j] means j cities served.
                # So next city to add is permutation[j].
                
                if j < n and j < limit:
                    next_city = permutation[j]
                    d_inter = d_mat[prev, next_city]
                    
                    step_cost = cost_fn(d_inter, current_load)
                    current_trip_cost += step_cost
                    current_load += golds[next_city]
                    prev = next_city
        
        # Reconstruct Trips
        trips = []
        curr = n
        while curr > 0:
            start = P[curr]
            sub_route = permutation[start:curr]
            trips.append(list(sub_route)) 
            curr = start
        trips.reverse()
        
        return V[n], trips

    def step_generation(self):
        """Advances the simulation by one generation."""
        self.generation_count += 1
        
        for island in self.islands:
            island.evolve_step()
            if island.population[0].cost < self.global_best.cost:
                self.global_best = island.population[0].clone()
        
        log_interval = max(1, self.max_generations // 10)
        
        if self.generation_count % log_interval == 0:
            self._migrate()
        return {
            "gen": self.generation_count,
            "best_cost": self.global_best.cost,
            "island_stats": {isl.name: isl.population[0].cost for isl in self.islands}
        }
    
    def _migrate(self):
        """Ring topology migration."""
        migrants = [isl.population[0].clone() for isl in self.islands]
        for i in range(len(self.islands)):
            target = self.islands[(i + 1) % len(self.islands)]
            migrant = migrants[i]
            target.population[-1] = migrant
            target.evaluate_population()

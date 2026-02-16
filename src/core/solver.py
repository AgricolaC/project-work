import logging
import time
import numpy as np
import networkx as nx
import heapq

# Logging config
logging.basicConfig(level=logging.INFO, format='%(message)s')

ISLAND_PRIORS = {
    'Exploit': {
        'mut_rate': (0.01, 0.10),      # Very low mutation to preserve elite traits
        'win_scale': (1.5, 2.5),       # MAX window: Spend CPU here to find perfect splits
        'mut_mix': [0.1, 0.8, 0.1]     # 80% Inversion (2-opt) to untangle crossings
    },
    
    'Balanced': {
        'mut_rate': (0.12, 0.26),      # Moderate mutation
        'win_scale': (0.8, 1.2),       # Standard window (1.0)
        'mut_mix': [0.3, 0.4, 0.3]     # Balanced mix
    },
    
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
        # Check if solver has computed regime yet, if not compute it
        regime = getattr(self.sim, 'regime', None)
        if regime is None:
             regime = self.sim.describe_regime()  
        curv = regime['curvature']
        rho_bucket = regime['rho_global_bucket']
        if curv == 'concave' and rho_bucket != 'distance' and role == 'Explore':
            # Boost Mutation for Scramble (Explore) in Concave regimes
            old_min, old_max = self.priors['mut_rate']
            self.priors['mut_rate'] = (min(0.9, old_min * 1.5), min(0.95, old_max * 1.5))
        elif curv == 'convex' and rho_bucket == 'penalty' and role == 'Explore':
            # Reduce Scramble in Convex Penalty regimes (structure matters more than chaos)
            # Bias toward Inversion/Swap which preserve more structure
            self.priors['mut_mix'] = [0.2, 0.5, 0.3] # Swap, Inv, Scramble

    def initialize(self, seeds=None):
        self.population = []
        # Inject Seeds if provided
        if seeds:
            for seed_genome in seeds:
                if len(self.population) < self.pop_size:
                    self.population.append(Individual(list(seed_genome), self._sample_params()))
                
        # Internal Heuristics
        heuristics_enabled = self.sim.ablation_config.get('seeding', True)
        seeding_mode = self.sim.ablation_config.get('seeding_mode', 'full')
            
        if heuristics_enabled:
            if seeding_mode == 'minimal':
                # Minimal: Just NN, Cheapest Insertion, Random
                heuristics = ['nearest_neighbor', 'cheapest_insertion', 'gold_aware_greedy']
            else:
                # Full mode (Role specific)
                heuristics = {
                    'Exploit': ['cheapest_insertion', 'gold_aware_greedy', 'nearest_neighbor', 'geometric', 'cluster_cleanup'],
                    'Balanced': ['cheapest_first', 'far_first', 'gold_aware_greedy', 'heavy_last'],
                    'Explore': [], # Explore usually random, maybe 1 simple
                    'Single': ['cheapest_insertion', 'gold_aware_greedy', 'nearest_neighbor'] # For single island
                }.get(self.role, [])
        else:
            heuristics = []

        # Limit seeds to 1 per heuristic to avoid overwhelming the diversity
        num_smart = 1 if heuristics else 0
        
        for h_name in heuristics:
            for _ in range(num_smart):
                genome = self._generate_smart_genome(h_name)
                self.population.append(Individual(genome, self._sample_params()))

        # inject a near-singleton seed (heaviest gold first)
        regime = getattr(self.sim, 'regime', {})
        rho_bucket = regime.get('rho_global_bucket', 'mixed')
        if rho_bucket == 'penalty' and len(self.population) < self.pop_size:
            # Gold-sorted descending: heaviest cities first means each gets its own trip
            penalty_seed = sorted(self.sim.cities, key=lambda c: self.sim.real_golds[c], reverse=True)
            self.population.append(Individual(penalty_seed, self._sample_params()))

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

        elif strategy == 'heavy_first':
             # Sort by gold amount (descending)
            return sorted(cities, key=lambda c: self.sim.real_golds[c], reverse=True)
            
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
            
        elif strategy == 'gold_aware_greedy':
            return self._gold_aware_greedy_genome()
            
        elif strategy == 'cheapest_insertion':
             return self._cheapest_insertion_genome()
             
        elif strategy == 'geometric':
            # Polar angle sort appropriate for VRP
            return sorted(cities, key=lambda c: self._get_angle(c))

        elif strategy == 'heavy_last':
            return self._heavy_last_seed()

        elif strategy == 'cluster_cleanup':
             return self._cluster_cleanup_seed()
            
        return cities

    def _heavy_last_seed(self):
        """
        Sort cities into quantiles by gold. Visit lighter quantiles first, using NN within each.
        """
        cities = list(self.sim.cities)
        # Sort by gold
        cities.sort(key=lambda c: self.sim.real_golds[c])
        
        # 3 Quantiles
        n = len(cities)
        q1 = cities[:n//3]
        q2 = cities[n//3:2*n//3]
        q3 = cities[2*n//3:]
        
        genome = []
        curr = 0
        
        for q in [q1, q2, q3]:
            unvisited = set(q)
            while unvisited:
                 nn = min(unvisited, key=lambda c: self.sim.dist_matrix[curr, c])
                 genome.append(nn)
                 unvisited.remove(nn)
                 curr = nn
        return genome

    def _cluster_cleanup_seed(self):
        """
        Geometric sort + Short refinement.
        """
        cities = sorted(self.sim.cities, key=lambda c: self._get_angle(c))
        # Create temp individual
        ind = Individual(cities, self._sample_params())
        # Refine briefly 
        ind.cost, ind.trips = self.sim.split_route(cities, win_scale=None)
        refined = self.sim.refine_solution(ind, max_steps=400, max_passes=1)
        return refined.genome

    def _gold_aware_greedy_genome(self):
        # Deterministic greedy with gold/dist ratio
        # For simplicity, we use k=0.5
        k = 0.5 
        cities = list(self.sim.cities)
        unvisited = set(cities)
        
        # Start with max gold node
        current = max(cities, key=lambda c: self.sim.real_golds[c])
        genome = [current]
        unvisited.remove(current)
        
        while unvisited:
            # Candidates restriction for speed if large
            candidates = unvisited
            if len(unvisited) > 100:
                candidates = set(list(unvisited)[:100]) # approximate
            
            # Minimize dist / (gold^k)
            # Avoid div by zero. Gold > 0 always?
            best_next = -1
            best_score = float('inf')
            
            for cand in candidates:
                d = self.sim.dist_matrix[current, cand]
                g = self.sim.real_golds[cand]
                # If gold is 0 (depot?), but these are cities.
                # If g < 1, might boost score.
                score = d / ( (g ** k) + 1e-6 )
                if score < best_score:
                    best_score = score
                    best_next = cand
            
            genome.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        return genome

    def _cheapest_insertion_genome(self):
        # Start with a small tour (3 cities)
        cities = list(self.sim.cities)
        # Sort by distance from depot to get a nucleus
        sorted_by_depot = sorted(cities, key=lambda c: self.sim.dist_matrix[0, c])
        
        # Nucleus: first 3
        tour = sorted_by_depot[:3]
        unvisited = set(sorted_by_depot[3:])
        
        # Insert remaining
        while unvisited:
            best_city = -1
            best_pos = -1
            min_added_cost = float('inf')
            
            # Sample subset for speed if large
            candidates = list(unvisited)
            if len(candidates) > 50:
                 candidates = candidates[:50] # Heuristic speedup
                 
            for city in candidates:
                for i in range(len(tour)):
                    # Insert city between tour[i] and tour[i+1] (circular)
                    u = tour[i]
                    v = tour[(i + 1) % len(tour)]
                    
                    cost_increase = self.sim.dist_matrix[u, city] + self.sim.dist_matrix[city, v] - self.sim.dist_matrix[u, v]
                    
                    if cost_increase < min_added_cost:
                        min_added_cost = cost_increase
                        best_city = city
                        best_pos = i + 1
            
            if best_city != -1:
                tour.insert(best_pos, best_city)
                unvisited.remove(best_city)
            else:
                 # Should not happen unless empty
                 break
                 
        return tour

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
                child_genome = self.local_search(child_genome)
                
            child = Individual(child_genome, child_params, self.sim)
            new_pop.append(child)
            
        self.population = new_pop
        self.evaluate_population()

        # Diversity Check: Island Catastrophe
        best_cost = self.population[0].cost
        median_cost = self.population[len(self.population)//2].cost
        # Relative diversity check (scale-aware)
        relative_gap = (median_cost - best_cost) / (abs(best_cost) + 1e-9)
        if relative_gap < 1e-3:  # Less than 0.1% spread 
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
        if not self.sim.ablation_config.get('adaptive_mutation', True):
            return params
            
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

    def local_search(self, genome):
        """
        True Cost Local Search using Split DP.
        Runs a few stochastic moves and accepts ONLY if splitting yields cheaper total cost.
        Includes: Swap, Relocate, 2-Opt.
        Budget: 40 attempts.
        """
        current_genome = list(genome)
        current_cost, _ = self.sim.split_route(current_genome, win_scale=None)
        
        n = len(current_genome)
        max_attempts = 15 # Reduced from 40 for speed
        
        for _ in range(max_attempts):
            op = self.sim.rng.choice(['relocate', 'swap', '2opt'])
            
            # Create Candidate
            candidate = list(current_genome)
            
            if op == 'relocate':
                # Move i to j
                i = self.sim.rng.integers(0, n)
                j = self.sim.rng.integers(0, n)
                if i == j: continue
                val = candidate.pop(i)
                candidate.insert(j, val)
                
            elif op == 'swap':
                # Swap i and j
                i, j = self.sim.rng.choice(range(n), 2, replace=False)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
            elif op == '2opt':
                # Reverse segment i..j
                i, j = sorted(self.sim.rng.choice(range(n), 2, replace=False))
                candidate[i:j+1] = candidate[i:j+1][::-1]
            
            # Evaluate using Exact Split
            new_cost, _ = self.sim.split_route(candidate, win_scale=None)
            
            if new_cost < current_cost - 1e-6:
                current_genome = candidate
                current_cost = new_cost
                
        return current_genome

class GA_Solver:
    """
    Refactored solver using Giant Tour + Split DP.
    """
    @staticmethod
    @staticmethod
    def get_budget(n_nodes):
        """
        Returns time budget parameters scaled inversely with problem size.
        Conservative scaling to penalize large instances and save compute.
        """
        # 1. Generations (GA)
        # Scale: 3000/N. Floor: 30.
        # N=20 -> 75. N=50 -> 30. N=100 -> 15.
        max_gens = max(15, int(1500 / max(1, n_nodes)))
        
        # 2. LNS Iterations
        # Scale: 15000/N. Floor: 100.
        # N=20 -> 750. N=50 -> 300. N=100 -> 150.
        lns_iters = max(100, int(15000 / max(1, n_nodes)))
        
        # 3. Population Size
        # Previously constant 21. Now dynamic.
        # Scale: 600/N. Floor: 15. Cap: 50.
        # N=20 -> 60. N=50 -> 24. N=100 -> 12 (->15).
        # Small problems get denser checks, large problems minimal robust pop.
        raw_pop = int(1200 / max(1, n_nodes))
        pop_size = max(15, min(50, raw_pop))
        
        return {
            'pop_size': pop_size,
            'max_generations': max_gens,
            'lns_iters': lns_iters
        }

    def __init__(self, problem, pop_size_per_island=None, max_generations=None, initial_individuals=None, ablation_config=None, seed=42):
        self.problem = problem
        
        # Dynamic Budget Defaults
        n_nodes = problem.graph.number_of_nodes()
        defaults = self.get_budget(n_nodes)
        
        self.pop_size = pop_size_per_island if pop_size_per_island is not None else defaults['pop_size']
        self.max_generations = max_generations if max_generations is not None else defaults['max_generations']
        
        self.initial_individuals = initial_individuals or []
        self.rng = np.random.default_rng(seed)


        
        # Ablation Config
        self.ablation_config = {
            'seeding': True,  # Main toggle
            'seeding_mode': 'full', # 'full' or 'minimal'
            'local_search': False,
            'island_mode': '3-island', # '3-island' or 'single'
            'adaptive_mutation': True # True or False
        }
        if ablation_config:
            self.ablation_config.update(ablation_config)
            
        self.cached_graph = problem.graph

        # Precompute APSP (All-Pairs Shortest Path)
        n_nodes = self.cached_graph.number_of_nodes()
        self.dist_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64) # Use float64 for safety
        
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

        # Precompute k-NN for targeted search
        self._precompute_knn(n_nodes)


        # Cache Golds
        golds_map = nx.get_node_attributes(problem.graph, 'gold')
        # Ensure array access
        self.real_golds = np.zeros(n_nodes, dtype=np.float64)
        for i, g in golds_map.items():
            self.real_golds[i] = g
            
        # Precompute L and S matrices for exact cost alignment
        self._precompute_LS(problem)
        
        # Precompute Spatial Clusters for LNS
        self._precompute_clusters(n_nodes)
        
        # Determine Regime (Curvature & Penalty Dominance)
        # Define Cities (1..N-1)
        self.cities = list(range(1, n_nodes))
        self.mean_gold = np.mean(self.real_golds[self.cities]) if self.cities else 1.0
        
        # Determine Regime (Curvature & Penalty Dominance)
        self.regime = self.describe_regime()
        logging.info(f"Regime: {self.regime['curvature']} | RhoGlobal: {self.regime['rho_global']:.2e} ({self.regime['rho_global_bucket']})")
        self.win_base = 20 # Window for Split DP
        
        # Configure Islands
        self.islands = []
        
        if self.ablation_config['island_mode'] == 'single':
            # To keep evaluation budget (evals/gen) consistent, we trip population.
            # 3 islands of N -> 1 island of 3N
            single_pop = self.pop_size * 3
            isl = Island('Single', 'Balanced', single_pop, self)
            
            seeds = None
            if self.ablation_config['seeding']:
                seeds = self.initial_individuals
            
            isl.initialize(seeds=seeds)
            self.islands.append(isl)
            
        else:
            # Standard 3-Island
            for i, role in enumerate(['Exploit', 'Balanced', 'Explore']):
                isl = Island(role, role, self.pop_size, self)
                
                seeds = None
                if self.ablation_config['seeding'] and i == 0: 
                    seeds = self.initial_individuals
                
                isl.initialize(seeds=seeds)
                self.islands.append(isl)

        self.global_best = self.islands[0].population[0]
        self.generation_count = 0

    def _precompute_knn(self, n_nodes, k=20):
        """
        Precompute k-Nearest Neighbors for each node to speed up local search.
        """
        self.nn = {}
        # We can use self.dist_matrix since it's fully populated
        for i in range(n_nodes):
            # argsort the i-th row
            # exclude self (dist 0 usually at index i)
            dists = self.dist_matrix[i]
            # Get indices of top k+1 smallest
            sorted_indices = np.argsort(dists)
            # Filter out i itself and depot if needed? 
            # Usually depot (0) is a valid neighbor for routing but maybe not for swap target if we swap cities only.
            # Cities are 1..N-1.
            neighbors = []
            for idx in sorted_indices:
                if idx != i:
                    neighbors.append(idx)
                    if len(neighbors) >= k:
                        break
            self.nn[i] = neighbors

    def _precompute_LS(self, problem):
        """
        Precomputes L (shortest path dist) and Sbeta (weighted sum) using
        vectorized Floyd-Warshall (O(N^3)).
        """
        n_nodes = problem.graph.number_of_nodes()
        self.L = np.full((n_nodes, n_nodes), float('inf'), dtype=np.float64)
        self.Sbeta = np.full((n_nodes, n_nodes), float('inf'), dtype=np.float64)
        self.parent = np.full((n_nodes, n_nodes), -1, dtype=np.int32)
        
        beta = problem.beta
        
        # 1. Initialize with direct edges
        np.fill_diagonal(self.L, 0.0)
        np.fill_diagonal(self.Sbeta, 0.0)
        
        for u, v, data in problem.graph.edges(data=True):
            d = data.get('dist', float('inf'))
            sb = d ** beta
            
            if d < self.L[u, v]:
                self.L[u, v] = self.L[v, u] = d
                self.Sbeta[u, v] = self.Sbeta[v, u] = sb
                self.parent[u, v] = u
                self.parent[v, u] = v
                
        # 2. Vectorized Floyd-Warshall
        for k in range(n_nodes):
            # Broadcast row/col k
            dist_ik = self.L[:, k][:, np.newaxis]
            dist_kj = self.L[k, :]
            new_dist = dist_ik + dist_kj
            
            sb_ik = self.Sbeta[:, k][:, np.newaxis]
            sb_kj = self.Sbeta[k, :]
            new_sb = sb_ik + sb_kj
            
            pred_kj = self.parent[k, :]
            
            # Optimization Masks
            with np.errstate(invalid='ignore'):
                mask_better = new_dist < self.L - 1e-9
                mask_equal = (np.abs(new_dist - self.L) < 1e-9) & (new_sb < self.Sbeta - 1e-9)
            
            mask_update = mask_better | mask_equal
            
            if np.any(mask_update):
                self.L[mask_update] = new_dist[mask_update]
                self.Sbeta[mask_update] = new_sb[mask_update]
                
                # Update parents: Parent[i, j] becomes Parent[k, j]
                pred_broadcast = np.tile(pred_kj, (n_nodes, 1))
                self.parent[mask_update] = pred_broadcast[mask_update]



    def expand_solution(self, trips):
        """
        Expands a list of trips (sequences of pickup cities) into a full solution 
        that includes all intermediate transitive nodes visited.
        
        Args:
            trips: List of lists, e.g. [[1, 2], [3, 4]]
            
        Returns:
            List of tuples [(node_id, gold_taken), ...]
            - Starts and ends with (0,0) (implicitly handled by trip loop)
            - Intermediate nodes have gold=0
            - Pickup nodes have gold=real_gold
        """
        solution = []
        
        # Start at depot
        curr_node = 0
        solution.append((0, 0)) # Start
        
        for trip in trips:
            # Trip is a sequence of target pickups: [t1, t2, ...]
            # We go curr -> t1 -> t2 ... -> last -> 0
            full_sequence = trip + [0] # Must return to depot 0 at end
            
            for target in full_sequence:
                # Reconstruct path from curr_node to target
                # Using self.parent[curr_node, target] is NOT sufficient directly because 
                # parent matrix stores 'predecessor of v in path FROM source'.
                # So we need to trace BACK from target to curr_node using self.parent[curr_node, ...]
                path_stack = []
                trace = target
                # Verify reachability
                if self.L[curr_node, target] == float('inf'):
                    # Should not happen in connected components, but fail safe
                    path_stack = [target]
                else:
                    # Attempt parent tracing
                    temp_trace = trace
                    temp_stack = []
                    valid_trace = True    
                    # Safety counter to prevent infinite loops (N nodes max depth)
                    steps = 0
                    max_steps = len(self.cached_graph.nodes) + 1
                    
                    while temp_trace != curr_node and steps < max_steps:
                        temp_stack.append(temp_trace)
                        prev = self.parent[curr_node, temp_trace]
                        if prev == -1: 
                            valid_trace = False
                            break
                        temp_trace = prev
                        steps += 1
                        
                    if valid_trace and temp_trace == curr_node:
                        # Tracing successful
                        path_stack = temp_stack
                    else:
                        # Fallback to NetworkX shortest path
                        try:
                            # Returns [curr, v1, v2, ... target]
                            nx_path = nx.shortest_path(self.cached_graph, source=curr_node, target=target, weight='dist')
                            # Stack needs [target, ..., v2, v1] (popped LIFO)
                            path_stack = nx_path[1:][::-1]
                        except Exception:
                            # Last resort: direct jump (should imply edge exists or penalty will be huge later)
                            path_stack = [target]
                # Pops trace path: (v1, v2, ..., target)
                while path_stack:
                    node = path_stack.pop()
                    
                    taken_gold = 0
                    # If this node is the 'target' (planned pickup), take its gold
                    if node == target and node != 0: 
                        taken_gold = self.real_golds[node]
                        
                    if node == 0:
                        taken_gold = 0 # Drop off
                        
                    solution.append((node, taken_gold))
                    
                curr_node = target
                
        return solution

    def get_dist(self, u, v):
        return self.dist_matrix[u, v]

    def split_route(self, permutation, win_scale):
        """
        Optimal Split DP for VRP.
        Decides where to insert returns to depot (0) to minimize total cost.
        Returns: (min_cost, trips)
        """
        n = len(permutation)
        if win_scale is None:
            win = n
        else:
            win = max(5, int(self.win_base * win_scale))
        
        V = [float('inf')] * (n + 1)
        V[0] = 0.0
        P = [-1] * (n + 1) 
        
        # Optimization: Localize access
        d_mat = self.dist_matrix
        golds = self.real_golds
        depot = 0
        
        for i in range(n):
            if V[i] == float('inf'): continue
            
            # Case: Start trip for customer permutation[i]
            # Trip serves cities permutation[i] ... permutation[j-1]
            u = permutation[i]
            
            # Initial leg: Depot -> u (load=0)
            current_trip_cost = self.leg_cost(depot, u, 0)
            
            current_load = golds[u]
            prev = u
            
            limit = min(i + win + 1, n)
            
            for j in range(i + 1, limit + 1):
                # Currently at 'prev' (end of sequence so far)
                
                # Option 1: CLOSE TRIP here (return to depot)
                return_cost = self.leg_cost(prev, depot, current_load)
                total = V[i] + current_trip_cost + return_cost
                
                if total < V[j]:
                    V[j] = total
                    P[j] = i
                    
                # Option 2: EXTEND TRIP to next city (if valid)
                if j < n and j < limit:
                    next_city = permutation[j]
                    step_cost = self.leg_cost(prev, next_city, current_load)
                    current_trip_cost += step_cost
                    current_load += golds[next_city]
                    prev = next_city
        
        # Reconstruct Trips
        trips = []
        curr = n
        while curr > 0:
            start = P[curr]
            trips.append(list(permutation[start:curr])) 
            curr = start
        trips.reverse()
        
        return V[n], trips

    def leg_cost(self, u: int, v: int, w: float) -> float:
        """
        Calculates macro-leg cost: L[u,v] + (alpha*w)^beta * Sbeta[u,v].
        Matches summing per-edge Problem.cost along shortest path.
        """
        dist = self.L[u, v]
        if w <= 0:
            return dist
        
        alpha = self.problem.alpha
        beta = self.problem.beta
        
        if np.isinf(self.Sbeta[u, v]):
             return float('inf')
        
        # Penalty = (alpha * w)^beta * Sbeta
        try:
            total_penalty = ((alpha * w) ** beta) * self.Sbeta[u, v]
        except OverflowError:
            return float('inf')
            
        if np.isinf(total_penalty):
            return float('inf')
            
        return dist + total_penalty

    def curvature_class(self, beta: float, eps: float = 1e-12) -> str:
        if beta < 1.0 - eps:
            return "concave"
        if abs(beta - 1.0) <= eps:
            return "linear"
        return "convex"

    @staticmethod
    def _log_rho(alpha, beta, L, Sbeta, w):
        """
        Computes log(rho) using exact Sbeta (sum of edge powers) for numerical stability.
        rho = ((alpha * w)^beta * Sbeta) / L
        log(rho) = beta * log(alpha * w) + log(Sbeta) - log(L)
        """
        if w <= 1e-9 or alpha <= 1e-9:
            return -float('inf') 
        if L <= 1e-9:
            return float('inf') # Infinite density if distance is zero
        if Sbeta <= 1e-18:
            return -float('inf')
            
        # log(rho) = beta * (log(alpha) + log(w)) + log(Sbeta) - log(L)
        val = beta * (np.log(alpha) + np.log(w)) + np.log(Sbeta) - np.log(L)
        return val

    def compute_rho_global(self, mode="knn", sample_pairs=2000, w_quantiles=[0.25, 0.5, 0.75, 0.9]) -> float:
        """
        Computes global rho using robust sampling and log-space aggregation.
        Cached after first computation.
        """
        if hasattr(self, '_cached_rho_global') and self._cached_rho_global is not None:
            return self._cached_rho_global
            
        alpha = self.problem.alpha
        beta = self.problem.beta
        cities = self.cities
        n = len(cities)
        
        if n < 2: return 0.0

        # 1. Sample MACRO Edges (Shortest Paths u->v)
        # We need L[u,v] and Sbeta[u,v]
        samples = []
        
        if mode == "knn" and hasattr(self, 'nn') and self.nn:
            # Sample random cities and their random neighbors
            for _ in range(sample_pairs):
                u = self.rng.choice(cities)
                if u in self.nn and self.nn[u]:
                    v = self.rng.choice(self.nn[u])
                    samples.append((self.L[u, v], self.Sbeta[u, v]))
        else:
            # Fallback to random pairs
            for _ in range(sample_pairs):
                u = self.rng.choice(cities)
                v = self.rng.choice(cities)
                if u != v:
                    samples.append((self.L[u, v], self.Sbeta[u, v]))
                    
        # Filter invalid samples
        samples = [(l, s) for l, s in samples if l > 1e-9 and np.isfinite(l) and s > 1e-18]
        if not samples:
            return 0.0
            
        # 2. Sample Load (w)
        golds = [self.real_golds[c] for c in cities]
        w_vals = np.quantile(golds, w_quantiles)
        w_vals = [w for w in w_vals if w > 0]
        if not w_vals: w_vals = [1.0]
        
        # 3. Compute log(rho) for all combinations
        log_rhos = []
        
        for L_val, S_val in samples:
            for w in w_vals:
                lr = self._log_rho(alpha, beta, L_val, S_val, w)
                if np.isfinite(lr):
                    log_rhos.append(lr)
                    
        if not log_rhos:
            res = 0.0
        else:
            # 4. Aggregate
            median_log_rho = np.median(log_rhos)
            
            # 5. Return exp (clamped)
            if median_log_rho > 700: res = float('inf')
            elif median_log_rho < -700: res = 0.0
            else: res = float(np.exp(median_log_rho))
            
        self._cached_rho_global = res
        return res

    def compute_rho_solution(self, solution) -> float:
        """
        Computes solution-aware rho using executed solution.
        Respects pickup semantics: load only increases at pickup nodes.
        Expects solution = [(city, gold), ...] where consecutve tuples are EDGES.
        For a direct edge, Sbeta = dist^beta.
        """
        if not solution:
            return 0.0
            
        alpha = self.problem.alpha
        beta = self.problem.beta
        
        log_rhos = []
        current_load = 0.0
        
        for i in range(len(solution) - 1):
            u_node, u_gold = solution[i]
            v_node, v_gold = solution[i+1]
            
            if u_node == 0:
                current_load = 0.0 # Reset at depot
            else:
                current_load += u_gold # Pickup at u_node
            
            # Explicit Edge Distance from Graph
            # Robust to macro/L difference if any
            if self.cached_graph.has_edge(u_node, v_node):
                dist = self.cached_graph[u_node][v_node].get('dist', 1.0)
            else:
                # Fallback to L if edge missing (e.g. implicitly fully connected)
                dist = self.L[u_node, v_node]
            
            # For a single edge, Sbeta is just dist^beta
            if dist > 1e-9 and np.isfinite(dist):
                S_beta = dist ** beta
                lr = self._log_rho(alpha, beta, dist, S_beta, current_load)
                if np.isfinite(lr):
                    log_rhos.append(lr)
                    
        if not log_rhos:
            return 0.0
            
        median_log_rho = np.median(log_rhos)
        
        if median_log_rho > 700: return float('inf')
        if median_log_rho < -700: return 0.0
        
        return float(np.exp(median_log_rho))


    def describe_regime(self, trips=None, solution=None) -> dict:
        """
        Determines the problem regime (Concave/Linear/Convex) and 
        Penalty Dominance (Distance/Mixed/Penalty).
        
        Args:
            trips: Optional list of trips (will be expanded to solution if solution not provided).
            solution: Optional solution list [(city, gold), ...] for precise calculation.
            
        Returns:
            dict with keys: curvature, rho_global, rho_solution, rho_bucket, etc.
        """
        beta = self.problem.beta
        curvature = self.curvature_class(beta)
        
        # 1. Global Baseline
        rho_g = self.compute_rho_global(mode="knn")
        
        # 2. Solution Specific (precise)
        rho_s = None
        if solution is not None:
             rho_s = self.compute_rho_solution(solution)
        elif trips is not None:
             # Expand trips to solution to get correct pickup semantics
             solution = self.expand_solution(trips)
             rho_s = self.compute_rho_solution(solution)
             
        def bucket(rho):
            if rho is None: return None
            if rho < 1.0/9.0: return "distance"
            if rho > 9.0:     return "penalty"
            return "mixed"

        bucket_g = bucket(rho_g)
        bucket_s = bucket(rho_s) if rho_s is not None else None
        
        final_bucket = bucket_s if bucket_s else bucket_g
        
        return {
            "curvature": curvature,
            "rho_global": rho_g,
            "rho_global_bucket": bucket_g,
            "rho_solution": rho_s,
            "rho_solution_bucket": bucket_s,
            "final_bucket": final_bucket # Unified bucket for logic
        }

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
            
        # Evaluation Accuracy: Frequent re-check for elites
        recheck_interval = 10
        if self.generation_count % recheck_interval == 0:
            self._recheck_elites_exact()
            
        return {
            "gen": self.generation_count,
            "best_cost": self.global_best.cost,
            "island_stats": {isl.name: isl.population[0].cost for isl in self.islands}
        }
    
    def _recheck_elites_exact(self):
        """
        Periodically re-evaluate top individuals in each island with EXACT split 
        to ensure windowed approximation isn't misleading.
        """
        k_elites = 3
        for isl in self.islands:
            # Sort first
            isl.sort_pop()
            # Recheck top k
            for i in range(min(len(isl.population), k_elites)):
                ind = isl.population[i]
                real_cost, real_trips = self.split_route(ind.genome, win_scale=None)
                if abs(real_cost - ind.cost) > 1e-6:
                    ind.cost = real_cost
                    ind.trips = real_trips
            # Re-sort after updates
            isl.sort_pop()

    def _migrate(self):
        """Ring topology migration."""
        migrants = [isl.population[0].clone() for isl in self.islands]
        for i in range(len(self.islands)):
            target = self.islands[(i + 1) % len(self.islands)]
            migrant = migrants[i]
            target.population[-1] = migrant
            target.evaluate_population()

    def refine_solution(self, individual, max_steps=800, max_passes=5):
        """
        Refines a single individual using Targeted Elite Local Search.
        Includes:
        1. Objective-Aware Scoring (Distance + Gold)
        2. Relocate/Swap with k-NN
        3. Heavy-Late Relocation
        4. Ruin-and-Recreate (Escape)
        """
        genome = list(individual.genome)
        current_cost = individual.cost
        
        # Ensure we start with exact cost and FRESH trips
        cost, trips = self.split_route(genome, win_scale=None)
        # Critical: assign to individual so describe_regime uses current trips
        individual.cost = cost
        individual.trips = trips
        current_cost = cost
        
        n = len(genome)
        improved_any = True
        total_steps = 0
        pass_idx = 0
        
        mean_gold = self.mean_gold
        
        while improved_any and pass_idx < max_passes and total_steps < max_steps:
            improved_any = False
            pass_idx += 1
            
            # Current regime status
            if pass_idx == 1 or total_steps % 50 == 0:
                current_regime = self.describe_regime(individual.trips)
                rho_bucket = current_regime['rho_solution_bucket'] or current_regime['rho_global_bucket']
            
            # Identify High-Impact Cities
            # Score = (L_prev_u + L_u_next) * (1 + gold_u / mean_gold)
            contributions = []
            for i in range(n):
                u = genome[i]
                prev = genome[i-1] if i > 0 else 0
                next_ = genome[i+1] if i + 1 < n else 0
                
                dist_impact = self.L[prev, u] + self.L[u, next_]
                gold_factor = 1.0 + (self.real_golds[u] / mean_gold)
                
                score = dist_impact * gold_factor
                contributions.append((score, i))
            
            contributions.sort(key=lambda x: x[0], reverse=True)
            
            # Top 30% candidates
            limit = max(1, int(0.3 * n))
            candidates = [x[1] for x in contributions[:limit]]
            self.rng.shuffle(candidates)
            
            # Targeted Relocate & Swap
            for idx_src in candidates:
                if total_steps >= max_steps: break
                u = genome[idx_src]
                neighbors = self.nn.get(u, [])
                
                # Relocate near neighbors
                for v in neighbors[:10]: # Check closest 10
                    if total_steps >= max_steps: break
                    if v == 0: continue
                    try:
                        idx_v = genome.index(v)
                    except ValueError: continue
                    
                    # Try inserting u at idx_v or idx_v+1
                    # Logic: remove u first
                    temp_genome = list(genome)
                    temp_genome.pop(idx_src) 
                    # new index of v might have shifted
                    try: new_idx_v = temp_genome.index(v)
                    except ValueError: continue
                    
                    for offset in [0, 1]:
                        cand = list(temp_genome)
                        cand.insert(new_idx_v + offset, u)
                        c, t = self.split_route(cand, win_scale=None)
                        total_steps += 1
                        if c < current_cost - 1e-6:
                            current_cost, individual.cost, individual.trips = c, c, t
                            genome = cand
                            improved_any = True
                            break
                    if improved_any: break
                if improved_any: break
                
                # Swap with neighbors
                for v in neighbors[:5]:
                    if total_steps >= max_steps: break
                    if v == 0: continue
                    try: idx_dest = genome.index(v)
                    except ValueError: continue
                    
                    genome[idx_src], genome[idx_dest] = genome[idx_dest], genome[idx_src]
                    c, t = self.split_route(genome, win_scale=None)
                    total_steps += 1
                    if c < current_cost - 1e-6:
                        current_cost, individual.cost, individual.trips = c, c, t
                        improved_any = True
                        break
                    else: # Revert
                        genome[idx_src], genome[idx_dest] = genome[idx_dest], genome[idx_src]
                if improved_any: break

            if improved_any: continue

            # Heavy-Late Relocation (Regime Gated
            # Condition: Concave OR (Convex but Distance-Dominated) OR (Significant Bundling Present)
            # If strictly convex and penalty-dominated with singleton trips, this operator is harmful/useless.
            should_run_heavy = False
            if self.problem.beta <= 1:
                should_run_heavy = True
            else:
                 # Convex case:
                 # Run if distance dominated (penalty low) OR if we have multi-city trips (bundling active)
                 # Note: individual.trips is kept up to date by splits
                 multi_frac = sum(1 for t in individual.trips if len(t)>1) / len(individual.trips) if individual.trips else 0

                 if 'rho_bucket' not in locals():
                      rho_bucket = getattr(self, 'regime', {}).get('rho_global_bucket', 'mixed')
                      
                 if rho_bucket == 'distance' or multi_frac > 0.2:
                     should_run_heavy = True
            
            if should_run_heavy and pass_idx % 2 == 0:
                # Pick a heavy city
                heavy_candidates = sorted(range(n), key=lambda i: self.real_golds[genome[i]], reverse=True)[:10]
                self.rng.shuffle(heavy_candidates)
                
                for idx_src in heavy_candidates:
                    if total_steps >= max_steps: break
                    u = genome[idx_src]
                    # Target zone: [0.8*n, n]
                    start_zone = int(0.8 * n)
                    if idx_src >= start_zone: continue # Already late
                    
                    target_idx = self.rng.integers(start_zone, n)
                    
                    temp_genome = list(genome)
                    temp_genome.pop(idx_src)
                    temp_genome.insert(target_idx, u)
                    
                    c, t = self.split_route(temp_genome, win_scale=None)
                    total_steps += 1
                    if c < current_cost - 1e-6:
                         current_cost, individual.cost, individual.trips = c, c, t
                         genome = temp_genome
                         improved_any = True
                         break
            
            if improved_any: continue

            # Penalty-Aware De-bundle (for rho=penalty, beta>=1
            # Target: break costly internal legs by relocating cities to early positions
            if rho_bucket == 'penalty' and self.problem.beta >= 1 and pass_idx % 2 == 1:
                # Find worst internal legs by penalty contribution
                worst_legs = []
                for trip in individual.trips:
                    if len(trip) < 2: continue
                    w = float(self.real_golds[trip[0]])
                    for k in range(len(trip) - 1):
                        u, v = trip[k], trip[k+1]
                        # Score = penalty component = leg_cost - base_distance
                        score = self.leg_cost(u, v, w) - self.L[u, v]
                        worst_legs.append((score, u, v, w))
                        w += float(self.real_golds[v])
                
                if worst_legs:
                    worst_legs.sort(key=lambda x: x[0], reverse=True)
                    # Pick from top K
                    K = min(5, len(worst_legs))
                    candidates = worst_legs[:K]
                    self.rng.shuffle(candidates)
                    
                    for (_, u, v, w_leg) in candidates[:3]:  # Try up to 3
                        if total_steps >= max_steps: break
                        
                        # Try relocating v to early positions (break the expensive leg)
                        try:
                            idx_v = genome.index(v)
                        except ValueError:
                            continue
                        
                        # Target: first 30% of genome (low-load positions)
                        early_zone = max(1, int(0.3 * n))
                        target_positions = list(range(early_zone))
                        self.rng.shuffle(target_positions)
                        
                        for target_idx in target_positions[:5]:  # Try up to 5 positions
                            if total_steps >= max_steps: break
                            
                            temp_genome = list(genome)
                            temp_genome.pop(idx_v)
                            temp_genome.insert(target_idx, v)
                            
                            c, t = self.split_route(temp_genome, win_scale=None)
                            total_steps += 1
                            
                            if c < current_cost - 1e-6:
                                current_cost, individual.cost, individual.trips = c, c, t
                                genome = temp_genome
                                improved_any = True
                                break
                        
                        if improved_any: break
            
            if improved_any: continue
            # Stronger LNS as requested: 15-20% ruin, Exact Cost Recreate
            if not improved_any and total_steps < max_steps * 0.9:
                # Ruin: Remove 15% (min 4)
                n_remove = max(4, int(0.15 * n))
                temp_genome = list(genome)
                removed = []
                
                # Random spatial chunk (preserve some structure) 
                # or Random scattering. Chunk is often better for VRP.
                if n > n_remove:
                    start = self.rng.integers(0, n - n_remove)
                    # Slicing
                    chunk = temp_genome[start : start + n_remove]
                    del temp_genome[start : start + n_remove]
                    removed = chunk
                
                # Recreate: "Best Insertion" using Exact Split
                # For small N (30-100), this is computationally feasible.
                # (approx 150-500 evals per cycle)
                
                # Shuffle removed to avoid deterministic bias in insertion order
                self.rng.shuffle(removed)
                
                for city in removed:
                    best_pos = -1
                    best_cost_increase = float('inf')
                    
                    # Baseline cost of partial solution
                    for k in range(len(temp_genome) + 1):
                        curr_cand = list(temp_genome)
                        curr_cand.insert(k, city)
                        
                        # Exact Eval
                        c, _ = self.split_route(curr_cand, win_scale=None)
                        total_steps += 1
                        
                        if c < best_cost_increase:
                            best_cost_increase = c
                            best_pos = k
                        
                        # Budget check
                        if total_steps >= max_steps:
                            # Abort LNS, revert to pre-ruin if strictly enforcing budget?
                            # Or just finish current insertion sub-optimally?
                            # Let's break loops
                            break
                    
                    if total_steps >= max_steps: break
                    if best_pos != -1:
                        temp_genome.insert(best_pos, city)
                
                # Check outcome if completed
                if len(temp_genome) == n:
                    c, t = self.split_route(temp_genome, win_scale=None)
                    if c < current_cost - 1e-6:
                         current_cost, individual.cost, individual.trips = c, c, t
                         genome = temp_genome
                         improved_any = True
                         # Reset pass count to encourage local polish of new basin
                         pass_idx = max(0, pass_idx - 1) 
        
        individual.genome = list(genome)
        return individual

    def refine_top_k(self, k=3, max_evals_per=800):
        """

        Refines the top k individuals from the global pool (or island bests).
        Returns the best refined individual.
        """
        # Gather bests from islands
        candidates = []
        for isl in self.islands:
            candidates.extend(isl.population[:2]) # Take top 2 from each
        
        # Sort and take top k unique genomes (by cost) to avoid redundant work
        candidates.sort(key=lambda x: x.cost)
        unique_candidates = []
        seen_costs = set()
        
        for cand in candidates:
            if cand.cost not in seen_costs:
                unique_candidates.append(cand.clone())
                seen_costs.add(cand.cost)
            if len(unique_candidates) >= k:
                break
        
        best_refined = self.global_best.clone()
        
        for cand in unique_candidates:
            refined = self.refine_solution(cand, max_steps=max_evals_per)
            if refined.cost < best_refined.cost:
                best_refined = refined
                
        return best_refined
        

    def _precompute_clusters(self, n_nodes):
        """Precomputes spatial clusters (polar sectors) for LNS."""
        self.clusters = {}
        self.cluster_members = {}
        
        if not self.node_positions:
            return 
            
        # Calc angles relative to depot (0,0 assumed or Node 0)
        depot_pos = self.node_positions[0]
        
        angles = []
        for i in range(1, n_nodes):
            pos = self.node_positions[i]
            theta = np.arctan2(pos[1] - depot_pos[1], pos[0] - depot_pos[0])
            angles.append((theta, i))
            
        angles.sort()
        
        # K sectors ~ sqrt(N)
        k = max(2, int(np.sqrt(n_nodes - 1)))
        chunk_size = len(angles) // k
        
        for i in range(k):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < k - 1 else len(angles)
            members = [x[1] for x in angles[start:end]]
            self.cluster_members[i] = members
            for city in members:
                self.clusters[city] = i

    def improve_with_lns(self, individual, iters=200, destroy_frac=(0.10, 0.30)):
        """
        Robust Large Neighborhood Search (LNS) to escape local optima.
        """
        best_ind = individual.clone()
        current_ind = individual.clone()
        
        # Ensure exact cost
        c, t = self.split_route(current_ind.genome, win_scale=None)
        current_ind.cost, current_ind.trips = c, t
        if c < best_ind.cost:
            best_ind = current_ind.clone()
            
        n = len(current_ind.genome)
        
        # Operators
        destroy_ops = [
            'random', 'segment', 'cluster', 'related', 'heavy'
        ]
        # Weights (simple adaptive)
        weights = {op: 1.0 for op in destroy_ops}
        alpha = 0.1 
        
        for i in range(iters):
            # Select operator
            keys = list(weights.keys())
            probs = np.array(list(weights.values()))
            probs /= probs.sum()
            op_name = self.rng.choice(keys, p=probs)
            
            # Destroy
            f = self.rng.uniform(destroy_frac[0], destroy_frac[1])
            k = max(4, int(f * n)) 
            k = min(k, n - 2)
            
            removed = []
            partial_genome = list(current_ind.genome)
            
            if op_name == 'random':
                removed, partial_genome = self._destroy_random(partial_genome, k)
            elif op_name == 'segment':
                removed, partial_genome = self._destroy_segment(partial_genome, k)
            elif op_name == 'cluster':
                 removed, partial_genome = self._destroy_cluster(partial_genome, k)
            elif op_name == 'related':
                 removed, partial_genome = self._destroy_related(partial_genome, k)
            elif op_name == 'heavy':
                 removed, partial_genome = self._destroy_heavy(partial_genome, k)
                 
            # Repair (Regret-Proxy) - pass solution-aware rho_bucket
            sol_regime = self.describe_regime(current_ind.trips)
            rho_bucket = sol_regime['rho_solution_bucket'] or sol_regime['rho_global_bucket'] or self.regime.get('rho_global_bucket', 'mixed')
            new_genome = self._repair_regret(partial_genome, removed, rho_bucket=rho_bucket)
            
            # Evaluate Exact
            new_cost, new_trips = self.split_route(new_genome, win_scale=None)
            
            if new_cost < current_ind.cost - 1e-6:
                current_ind.genome = new_genome
                current_ind.cost = new_cost
                current_ind.trips = new_trips
                
                if new_cost < best_ind.cost - 1e-6:
                    best_ind = current_ind.clone()
                    weights[op_name] *= (1 + alpha) 
            
        return best_ind

    # Destroy Operators
    
    def _destroy_random(self, genome, k):
        # Indices to remove
        if k >= len(genome): k = len(genome) - 1
        indices = sorted(self.rng.choice(range(len(genome)), k, replace=False), reverse=True)
        removed = []
        for i in indices:
            removed.append(genome.pop(i))
        return removed, genome
        
    def _destroy_segment(self, genome, k):
        if k >= len(genome): k = len(genome) - 1
        start = self.rng.integers(0, len(genome) - k) if len(genome) > k else 0
        removed = genome[start : start + k]
        del genome[start : start + k]
        return removed, genome

    def _destroy_cluster(self, genome, k):
        if not self.clusters:
             return self._destroy_random(genome, k)
             
        cid = self.rng.choice(list(self.cluster_members.keys()))
        members = set(self.cluster_members[cid])
        
        new_genome = []
        removed = []
        for city in genome:
            if city in members and len(removed) < k:
                removed.append(city)
            else:
                new_genome.append(city)
                
        if len(removed) < k:
             rem, new_genome = self._destroy_random(new_genome, k - len(removed))
             removed.extend(rem)
             
        return removed, new_genome

    def _destroy_related(self, genome, k):
        seed_idx = self.rng.integers(0, len(genome))
        seed_city = genome[seed_idx]
        lam = 0.5 
        
        candidates = []
        for i, city in enumerate(genome):
            if city == seed_city: continue
            d = self.L[seed_city, city]
            g_diff = abs(self.real_golds[seed_city] - self.real_golds[city])
            r = d + lam * g_diff
            candidates.append((r, city))
            
        candidates.sort(key=lambda x: x[0])
        limit = min(len(candidates), 2*k)
        pool = candidates[:limit]
        self.rng.shuffle(pool)
        
        targets = set([x[1] for x in pool[:k-1]])
        targets.add(seed_city)
        
        new_genome = []
        removed = []
        for city in genome:
            if city in targets:
                removed.append(city)
            else:
                new_genome.append(city)
        
        return removed, new_genome

    def _destroy_heavy(self, genome, k):
        candidates = []
        for c in genome:
            candidates.append((self.real_golds[c], c))
            
        candidates.sort(key=lambda x: x[0], reverse=True)
        limit = min(len(candidates), 2*k)
        pool = candidates[:limit]
        self.rng.shuffle(pool)
        
        targets = set([x[1] for x in pool[:k]])
        
        new_genome = []
        removed = []
        for city in genome:
            if city in targets:
                removed.append(city)
            else:
                new_genome.append(city)
        return removed, new_genome

    # Repair Operators 
    
    def _repair_regret(self, genome, removed, rho_bucket=None):
        """
        Regret-based repair: insert removed cities in heaviest-first order.
        Uses load-aware proxy for convex/penalty regimes.
        """
        # Fallback to global if no solution-specific rho provided
        if rho_bucket is None:
            rho_bucket = getattr(self, 'regime', {}).get('rho_global_bucket', 'mixed')
        
        # Heaviest First
        removed.sort(key=lambda c: self.real_golds[c], reverse=True)
        temp_genome = list(genome)
        
        for city in removed:
            best_val = float('inf')
            best_pos = -1
            
            n_curr = len(temp_genome)
            for i in range(n_curr + 1):
                prev = temp_genome[i-1] if i > 0 else 0
                next_ = temp_genome[i] if i < n_curr else 0
                
                # Use load-aware proxy if:
                # - Beta > 1 AND rho_bucket != 'distance', OR
                # - Beta == 1 AND rho_bucket == 'penalty' (linear but penalty-dominated)
                use_load_aware = False
                if self.problem.beta > 1 and rho_bucket != 'distance':
                    use_load_aware = True
                elif self.problem.beta == 1 and rho_bucket == 'penalty':
                    use_load_aware = True
                
                if use_load_aware:
                    # Load-Aware Proxy: estimate load as mean + city's gold
                    w_hat = self.mean_gold + self.real_golds[city]
                    cost_add = self.leg_cost(prev, city, w_hat) + self.leg_cost(city, next_, w_hat)
                    cost_rem = self.leg_cost(prev, next_, w_hat)
                    delta = cost_add - cost_rem
                else:
                    # Distance-Only Proxy for Concave/Linear or Distance-Dominated
                    delta = self.L[prev, city] + self.L[city, next_] - self.L[prev, next_]
                
                if delta < best_val:
                    best_val = delta
                    best_pos = i
            
            temp_genome.insert(best_pos, city)
            
        return temp_genome



    def get_solution_diagnostics(self, individual):
        """
        Computes detailed diagnostics for a solution.
        Includes trip structure, depot detours, and bundling advantage.
        """
        # Ensure trips are up to date
        cost, trips = self.split_route(individual.genome, win_scale=None)
        
        # 1. Trip Structure
        n_trips = len(trips)
        trip_lengths = [len(t) for t in trips]
        max_trip = max(trip_lengths) if trip_lengths else 0
        avg_trip = sum(trip_lengths) / n_trips if n_trips else 0
        multi_city_fraction = sum(1 for l in trip_lengths if l > 1) / n_trips if n_trips else 0
        
        # 2. Bundling Efficiency
        crossing_legs = 0
        total_internal_legs = 0
        near_depot_legs = 0
        marginal_bundle_legs = 0
        
        offending_examples = []
        
        for trip in trips:
            if len(trip) < 2: continue
            
            # Track ACTUAL load exactly like split DP
            # After picking up from first city, load = gold[first]
            w = float(self.real_golds[trip[0]])
            
            for k in range(len(trip) - 1):
                u = trip[k]
                v = trip[k+1]
                total_internal_legs += 1
                
                # Check 1: Depot detour ratio (distance only)
                via0 = self.L[u, 0] + self.L[0, v]
                direct = self.L[u, v]
                detour_ratio = via0 / (direct + 1e-9)
                
                if detour_ratio <= 1.02:  # 2% tolerance
                    near_depot_legs += 1
                    
                # Check 2: Bundling Advantage with ACTUAL load
                bundled_cost = self.leg_cost(u, v, w)
                split_cost = self.leg_cost(u, 0, w) + self.leg_cost(0, v, 0)  # return u->0 with w, start 0->v empty
                
                if split_cost > 0:
                    adv = bundled_cost / split_cost
                    if adv > 0.99:  # Minimal or negative advantage
                        marginal_bundle_legs += 1
                          
                # Check 3: Strict Path (Legacy check)
                try:
                    path = nx.shortest_path(self.problem.graph, u, v, weight='dist')
                    if 0 in path[1:-1]:
                        crossing_legs += 1
                        if len(offending_examples) < 3:
                            offending_examples.append((u, v))
                except nx.NetworkXNoPath:
                    pass
                
                # Update load: after visiting v, we carry more gold
                w += float(self.real_golds[v])
        crossing_ratio = crossing_legs / total_internal_legs if total_internal_legs > 0 else 0.0
        near_depot_ratio = near_depot_legs / total_internal_legs if total_internal_legs > 0 else 0.0
        marginal_bundle_ratio = marginal_bundle_legs / total_internal_legs if total_internal_legs > 0 else 0.0
        
        # Regime Description
        regime = self.describe_regime(trips)
        
        return {
            'n_trips': n_trips,
            'avg_len': avg_trip,
            'max_len': max_trip,
            'multi_frac': multi_city_fraction,
            'crossing_legs': crossing_legs,
            'total_internal': total_internal_legs,
            'crossing_ratio': crossing_ratio,
            'offending_samples': offending_examples,
            'near_depot_ratio': near_depot_ratio,
            'marginal_bundle_ratio': marginal_bundle_ratio,
            'curvature': regime['curvature'],
            'rho_solution': regime['rho_solution'],
            'rho_bucket': regime['rho_solution_bucket'],
            'rho_global': regime['rho_global']
        }

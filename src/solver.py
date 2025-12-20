import logging
from itertools import combinations
import time
import numpy as np
import networkx as nx

# Logging removed from step_generation but config kept for other potential uses
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
        'mut_rate': (0.20, 0.40),      # Moderate mutation
        'win_scale': (0.8, 1.2),       # Standard window (1.0)
        'mut_mix': [0.3, 0.4, 0.3]     # Balanced mix
    },
    
    # The "Scout": Absorbs the Chaos role. Fast, disruptive, low-precision.
    'Explore': {
        'mut_rate': (0.50, 0.80),      # High mutation (approaching Chaos levels)
        'win_scale': (0.3, 0.6),       # TINY window: Very fast, approximate evaluation
        'mut_mix': [0.1, 0.1, 0.8]     # 80% Scramble: Massive structural disruption
    }
}

class Individual:
    """
    Represents a candidate solution in the population.
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
        self.priors = ISLAND_PRIORS[role]
        
    def initialize(self):
        t0 = time.time()
        self.population = []
                
        if self.role == 'Exploit':
            heuristics = ['nearest_neighbor', 'geometric', 'far_first']
        elif self.role == 'Balanced':
            heuristics = ['cheapest_first', 'far_first']
        else: 
            # Explore and Chaos use pure random to preserve diversity
            heuristics = [] 

        num_smart_per_type = max(1, int(self.pop_size * 0.2 / max(1, len(heuristics)))) if heuristics else 0
        
        for h_name in heuristics:
            for _ in range(num_smart_per_type):
                genome = self._generate_smart_genome(h_name)
                self.population.append(Individual(genome, self._sample_params()))

        while len(self.population) < self.pop_size:
            rng_genome = list(self.sim.virtual_cities)
            np.random.shuffle(rng_genome)
            self.population.append(Individual(rng_genome, self._sample_params()))
        
        self.evaluate_population()

    def _generate_smart_genome(self, strategy):
        v_nodes = list(self.sim.virtual_cities)
        
        if strategy == 'far_first':
            return sorted(v_nodes, key=lambda v: self.sim.get_dist_to_base(v), reverse=True)
            
        elif strategy == 'cheapest_first':
            return sorted(v_nodes, key=lambda v: self.sim.node_golds[v])
            
        elif strategy == 'nearest_neighbor':
            genome = []
            curr = 0
            unvisited = set(v_nodes)
            while unvisited:
                candidates = unvisited
                if len(unvisited) > 200:
                    candidates = set(list(unvisited)[:200]) 
                
                nn = min(candidates, key=lambda v: self.sim.get_dist(curr, v))
                genome.append(nn)
                unvisited.remove(nn)
                curr = nn
            return genome
            
        elif strategy == 'geometric':
            genome = sorted(v_nodes, key=lambda v: self._get_angle(v))
            return genome
            
        return v_nodes

    def _get_angle(self, v_node):
        real_id = self.sim.virtual_map[v_node]
        pos = self.sim.problem.graph.nodes[real_id]['pos']
        return np.arctan2(pos[1] - 0.5, pos[0] - 0.5)

    def evaluate_population(self):
        for ind in self.population:
            if ind.cost == float('inf'):
                ind.cost, ind.trips = self.sim.split_route(ind.genome, ind.params['win_scale'])
        self.sort_pop()

    def _sample_params(self):
        return {
            'mut_rate': np.random.uniform(*self.priors['mut_rate']),
            'win_scale': np.random.uniform(*self.priors['win_scale']),
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
            
            if np.random.random() < child_params['mut_rate']:
                child_genome = self._mutate_genome(child_genome, child_params['mut_mix'])
                
            child = Individual(child_genome, child_params, self.sim)
            new_pop.append(child)
            
        self.population = new_pop
        self.evaluate_population()

        # Diversity Check: Island Catastrophe
        # If the island has converged (all individuals are clones/similar)
        # reset 90% of the population to random to force exploration.
        best_cost = self.population[0].cost
        median_cost = self.population[len(self.population)//2].cost
        
        if abs(median_cost - best_cost) < 1e-4: # Tolerance for float equality
            # Trigger Catastrophe
            logging.debug(f"Island {self.name} converged. Triggering Catastrophe.")
            # Keep only the elite, kill the rest
            survivors = [self.population[0]]
            
            while len(survivors) < self.pop_size:
                rng_genome = list(self.sim.virtual_cities)
                np.random.shuffle(rng_genome)
                survivors.append(Individual(rng_genome, self._sample_params()))
                
            self.population = survivors
            self.evaluate_population()

    def _tournament(self, k=3):
        candidates = [self.population[i] for i in np.random.randint(0, len(self.population), k)]
        return min(candidates, key=lambda x: x.cost)

    def _crossover_genome(self, p1, p2):
        size = len(p1)
        a, b = sorted(np.random.choice(range(size), 2, replace=False))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        
        # Optimization: Pre-calculate presence for O(1) lookups
        child_set = set(child[a:b+1])
        
        curr = (b + 1) % size
        p2_idx = (b + 1) % size
        
        count = size - (b - a + 1) # Number of slots to fill
        
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
        alpha = np.random.random()
        for k in ['mut_rate', 'win_scale']:
            new_params[k] = alpha * p1[k] + (1 - alpha) * p2[k]
        
        mix1 = np.array(p1['mut_mix'])
        mix2 = np.array(p2['mut_mix'])
        new_mix = alpha * mix1 + (1 - alpha) * mix2
        new_params['mut_mix'] = list(new_mix / new_mix.sum())
        return new_params

    def _mutate_params(self, params):
        tau = 0.1
        params['mut_rate'] = params['mut_rate'] * np.exp(tau * np.random.normal())
        params['mut_rate'] = np.clip(params['mut_rate'], 0.05, 0.95)
        
        params['win_scale'] = params['win_scale'] * np.exp(tau * np.random.normal())
        params['win_scale'] = np.clip(params['win_scale'], 0.3, 2.0)
        
        mix = np.array(params['mut_mix'])
        noise = np.random.normal(0, 0.1, size=len(mix))
        mix = np.abs(mix + noise)
        params['mut_mix'] = list(mix / mix.sum())
        
        return params

    def _mutate_genome(self, genome, mix):
        op = np.random.choice(['swap', 'inv', 'scramble'], p=mix)
        size = len(genome)
        
        if op == 'swap':
            i, j = np.random.choice(range(size), 2, replace=False)
            genome[i], genome[j] = genome[j], genome[i]
        elif op == 'inv':
            i, j = sorted(np.random.choice(range(size), 2, replace=False))
            genome[i:j+1] = genome[i:j+1][::-1]
        elif op == 'scramble':
            i, j = sorted(np.random.choice(range(size), 2, replace=False))
            sub = genome[i:j+1]
            np.random.shuffle(sub)
            genome[i:j+1] = sub
        return genome

class GA_Solver:
    """
    Main orchestration class. Handles the Problem, Virtual Node creation, and manages the Islands.
    """
    def __init__(self, problem, pop_size_per_island=30, max_generations=100):
        self.problem = problem
        self.pop_size = pop_size_per_island
        self.max_generations = max_generations
        
        # Optimization: Precompute distance matrix
        n_nodes = problem.graph.number_of_nodes()
        self.dist_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Flatten dictionary to matrix
        # Note: problem.graph node indices must match range(n_nodes) usually 0 to N-1
        raw_dists = dict(nx.all_pairs_dijkstra_path_length(problem.graph, weight='dist'))
        for u in range(n_nodes):
            for v in range(n_nodes):
                self.dist_matrix[u, v] = raw_dists[u].get(v, float('inf'))

        # To keep compat for _auto_tune_chunking which used self.real_dists[u][v]
        # We can simulate the subset needed, or update _auto_tune_chunking
        # Updating _auto_tune_chunking is better.
        self.real_golds = nx.get_node_attributes(problem.graph, 'gold')

        self.virtual_cities = []
        self.virtual_map = {} 
        self.node_golds = {}
        self.win_base = 20
        
        self._auto_tune_chunking()
        
        self.islands = []
        for role in ['Exploit', 'Balanced', 'Explore']:
            isl = Island(role, role, self.pop_size, self)
            isl.initialize()
            self.islands.append(isl)

        self.global_best = self.islands[0].population[0]
        self.generation_count = 0

    def _auto_tune_chunking(self):
        """
        Pre-check phase: Test different chunk scales and pick the best one.
        Optimized to prefer larger chunks (faster execution) effectively if cost penalty is low.
        """
        # Immediate shortcut for linear/sublinear cases
        if self.problem.beta <= 1.0:
            self._setup_virtual_nodes(target_chunk=float('inf')) 
            return


        # Granular scales from 0.50 to 1.0, plus larger steps (can go much lower but we have to save on time)
        # This replaces the hardcoded list with a generated one
        scales = list(np.arange(0.66, 1.00, 0.01)) 
        # Clean up floating point issues and ensure uniqueness/sorting
        scales = sorted(list(set([round(s, 2) for s in scales])))
        
        results = [] # Stores (cost, scale, num_nodes)
        
        # Heuristic Base Chunk Calculation using dist_matrix
        # Flatten upper triangle excluding diagonal
        n_nodes = self.dist_matrix.shape[0]
        iu = np.triu_indices(n_nodes, 1)
        all_dists = self.dist_matrix[iu]
        avg_dist = np.mean(all_dists) if all_dists.size > 0 else 1.0
        
        S = (self.problem.alpha ** self.problem.beta) * self.problem.beta * (avg_dist ** self.problem.beta) if self.problem.beta > 0 else 1.0
        
        n = self.problem.graph.number_of_nodes()
        m = self.problem.graph.number_of_edges()
        density = m / (n * (n - 1) / 2) if n > 1 else 1.0
        density_factor = 1.0 / max(density, 0.1)
        city_factor = np.log(n + 1)
        
        C = 1000.0
        denom = S * density_factor * city_factor
        chunk_base = C / denom if denom > 1e-6 else C
        chunk_base = np.clip(chunk_base, 20.0, 300.0)
        
        self.win_base = int(200.0 / (self.problem.beta * density_factor)) if self.problem.beta > 0 else 20
        self.win_base = np.clip(self.win_base, 20, 150)

        for scale in scales:
            target_chunk = chunk_base * scale
            self._setup_virtual_nodes(target_chunk)
            
            if not self.virtual_cities: continue
            
            curr = 0
            unvisited = set(self.virtual_cities)
            genome = []
            
            # Limit probe for speed
            step_limit = min(len(self.virtual_cities), 50) 
            
            for _ in range(step_limit):
                nn = min(unvisited, key=lambda v: self.get_dist(curr, v))
                genome.append(nn)
                unvisited.remove(nn)
                curr = nn
            
            genome.extend(list(unvisited))
            
            cost, _ = self.split_route(genome, 1.0)
            results.append((cost, scale, len(self.virtual_cities)))
            
        # Optimization Strategy: "Knee Point"
        if not results:
            best_scale = 1.0
        else:
            # Sort by cost primarily
            results.sort(key=lambda x: x[0])
            min_cost = results[0][0]
            
            # Allow 10% slack to find a larger scale (smaller problem size)
            threshold = min_cost * 1.10
            
            # Candidates that are within 10% of optimal cost
            candidates = [r for r in results if r[0] <= threshold]
            
            # Pick the candidate with the *largest* scale (index 1)
            # This directly minimizes execution time
            best_candidate = max(candidates, key=lambda x: x[1])
            
            best_cost = best_candidate[0]
            best_scale = best_candidate[1]
            best_nodes = best_candidate[2]
                            
        self._setup_virtual_nodes(chunk_base * best_scale)

    def _setup_virtual_nodes(self, target_chunk):
        self.virtual_cities = []
        self.virtual_map = {} 
        self.node_golds = {}
        
        n = self.problem.graph.number_of_nodes()
        vid_counter = 1
        for c in range(1, n):
            gold = self.real_golds[c]
            if gold > 1.2 * target_chunk and self.problem.beta > 1.0:
                num_chunks = int(np.ceil(gold / target_chunk))
                gold_per_chunk = gold / num_chunks
                for _ in range(num_chunks):
                    self.virtual_cities.append(vid_counter)
                    self.virtual_map[vid_counter] = c
                    self.node_golds[vid_counter] = gold_per_chunk
                    vid_counter += 1
            else:
                self.virtual_cities.append(vid_counter)
                self.virtual_map[vid_counter] = c
                self.node_golds[vid_counter] = gold
                vid_counter += 1

    def get_dist(self, u, v):
        # Fast Array Lookup
        # 0 is base, self.virtual_map maps virtual->real
        u_real = 0 if u == 0 else self.virtual_map[u]
        v_real = 0 if v == 0 else self.virtual_map[v]
        return self.dist_matrix[u_real, v_real]
    
    def get_dist_to_base(self, u_virt):
        return self.get_dist(0, u_virt)

    def split_route(self, permutation, win_scale):
        """
        Optimal Split Algorithm (Route First, Cluster Second).
        Takes a giant tour (permutation) and breaks it into valid trips returning to base.
        """
        n = len(permutation)
        win = int(self.win_base * win_scale)
        win = max(5, win)
        
        V = [float('inf')] * (n + 1)
        V[0] = 0.0
        P = [-1] * (n + 1)
        
        for i in range(n):
            if V[i] == float('inf'): continue
            
            u = permutation[i]
            d0 = self.get_dist(0, u)
            forward_cost = d0
            
            current_gold = 0.0
            prev = u
            current_gold += self.node_golds[prev]
            
            dr = self.get_dist(prev, 0)
            trip_cost = forward_cost + dr + (self.problem.alpha * dr * current_gold) ** self.problem.beta
            
            if V[i] + trip_cost < V[i+1]:
                V[i+1] = V[i] + trip_cost
                P[i+1] = i
            
            limit = min(i + win + 1, n + 1)
            for j in range(i + 2, limit):
                new_city = permutation[j-1]
                
                d_seg = self.get_dist(prev, new_city)
                forward_cost += d_seg + (self.problem.alpha * d_seg * current_gold) ** self.problem.beta
                
                current_gold += self.node_golds[new_city]
                prev = new_city
                
                dr = self.get_dist(new_city, 0)
                total_cost = forward_cost + dr + (self.problem.alpha * dr * current_gold) ** self.problem.beta
                
                if V[i] + total_cost < V[j]:
                    V[j] = V[i] + total_cost
                    P[j] = i
        
        trips = []
        curr = n
        while curr > 0:
            start = P[curr]
            sub_route = permutation[start:curr]
            trips.append(sub_route)
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
        
        # Migration every 10% of generations
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

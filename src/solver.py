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
        'mut_rate': (0.15, 0.30),      # Moderate mutation
        'win_scale': (0.8, 1.2),       # Standard window (1.0)
        'mut_mix': [0.3, 0.4, 0.3]     # Balanced mix
    },
    
    # The "Scout": Absorbs the Chaos role. Fast, disruptive, low-precision.
    'Explore': {
        'mut_rate': (0.40, 0.80),      # High mutation (approaching Chaos levels)
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
        self.priors = ISLAND_PRIORS[role].copy()
        
        # Concave Logic: Boost Mutation for Scramble (Explore)
        if self.sim.problem.beta < 1.0 and self.sim.problem.alpha < 3.0 and role == 'Explore':
            # Increase average mutation rate
            old_min, old_max = self.priors['mut_rate']
            self.priors['mut_rate'] = (min(0.9, old_min * 1.5), min(0.95, old_max * 1.5))
        
    def initialize(self, seeds=None):
        t0 = time.time()
        self.population = []
        # Inject Seeds if provided
        if seeds:
            for seed_genome in seeds:
                if len(self.population) < self.pop_size:
                    self.population.append(Individual(list(seed_genome), self._sample_params()))
                
        # Internal Heuristics (only enabled if Seeding not disabled via ablation)
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
            rng_genome = list(self.sim.virtual_cities)
            self.sim.rng.shuffle(rng_genome)
            self.population.append(Individual(rng_genome, self._sample_params()))
        
        self.evaluate_population()

    def _generate_smart_genome(self, strategy):
        v_nodes = list(self.sim.virtual_cities)
        
        if strategy == 'far_first':
            return sorted(v_nodes, key=lambda v: self.sim.get_dist_to_base(v), reverse=True)
            
        elif strategy == 'cheapest_first':
            return sorted(v_nodes, key=lambda v: self.sim.node_golds[v])
            
        elif strategy == 'nearest_neighbor':
            # Greedy NN with candidate capping for speed
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
            # Use utility function
            return sorted(v_nodes, key=lambda v: self._get_angle(v))
            
        return v_nodes

    def _get_angle(self, v_node):
        real_id = self.sim.virtual_map[v_node]
        pos = self.sim.cached_graph.nodes[real_id]['pos']
        # Local angle calculation
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
                
            child = Individual(child_genome, child_params, self.sim)
            new_pop.append(child)
            
        self.population = new_pop
        self.evaluate_population()

        # Diversity Check: Island Catastrophe
        # If the island has converged (all individuals are clones/similar)
        # reset 90% of the population to random to force exploration.
        best_cost = self.population[0].cost
        median_cost = self.population[len(self.population)//2].cost
        
        if abs(median_cost - best_cost) < 1e-4: 
            # Trigger Catastrophe
            # Keep only the elite, kill the rest
            survivors = [self.population[0]]
            
            while len(survivors) < self.pop_size:
                rng_genome = list(self.sim.virtual_cities)
                self.sim.rng.shuffle(rng_genome)
                survivors.append(Individual(rng_genome, self._sample_params()))
                
            self.population = survivors
            self.evaluate_population()

    def _tournament(self, k=3):
        candidates = [self.population[i] for i in self.sim.rng.integers(0, len(self.population), k)]
        return min(candidates, key=lambda x: x.cost)

    def _crossover_genome(self, p1, p2):
        size = len(p1)
        a, b = sorted(self.sim.rng.choice(range(size), 2, replace=False))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        
        # Optimization: Pre-calculate presence for O(1) lookups
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

class GA_Solver:
    """
    Main orchestration class. Handles the Problem, Virtual Node creation, and manages the Islands.
    """
    def __init__(self, problem, pop_size_per_island=30, max_generations=100, initial_individuals=None, ablation_config=None, seed=42):
        self.problem = problem
        self.pop_size = pop_size_per_island
        self.max_generations = max_generations
        self.initial_individuals = initial_individuals or []
        self.rng = np.random.default_rng(seed)
        
        # Ablation Config (Default: All Enabled)
        self.ablation_config = {
            'granular': True,
            'chunking': True,
            'seeding': True
        }
        
        # Threshold for usage of advanced physics (multi-hop approximation)
        self.multi_hop_threshold = 1.0
        if ablation_config:
            self.ablation_config.update(ablation_config)
            
        # Optimization: Cache graph to avoid expensive property copy in original problem.py
        self.cached_graph = problem.graph

        # Estimate average edge length for cost approximation
        # We can just sample edges if too many, but for N=2000 iterating all edges is fine.
        # But wait, problem.graph might be dense.
        # N=2000, max edges = 2 million. Iterating 2M edges is fast (fraction of second).
        
        edge_lengths = [d['dist'] for u, v, d in self.cached_graph.edges(data=True)]
        if len(edge_lengths) > 0:
            self.avg_edge_len = np.mean(edge_lengths)
            # If the graph is sparse but connected via long edges, this might be large.
            # But the granular expansion uses 'atomic' steps.
            # If avg_edge_len is too large, our k will be small.
            # Let's be conservative: use a smaller percentile or a fixed small value?
            # Problem says "unit square coordinates".
            # If we want to really approximate "atomic" cost, we should estimate
            # how many "hops" are in a shortest path of length L.
            # The hops are determined by the DENSITY of the graph.
            # If density is high, we can jump directly (1 hop).
            # But high beta penalizes 1 big jump.
            # So the agent WANTS multiple small jumps.
            # Are there intermediate nodes?
            # If density=1.0, direct edge exists. But triangular inequality holds for dist.
            # So dist(A,B) <= dist(A,C) + dist(C,B).
            # But Cost(A,B) > Cost(A,C) + Cost(C,B) if beta > 1.
            # So we WANT to find C.
            # The graph contains "random" points.
            # We can treat the field as a "sea" of points.
            # The expected distance to nearest neighbor is ~ 1/sqrt(N).
            # For N=1000, 1/31 ~= 0.03.
            # So we can assume we can find hops of length ~0.05 easily.
            self.avg_edge_len = max(0.05, float(np.percentile(edge_lengths, 10)))
        else:
             self.avg_edge_len = 0.1

        # Optimization: Precompute distance matrix
        n_nodes = self.cached_graph.number_of_nodes()
        self.dist_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Check density
        num_edges = problem.graph.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1) // 2
        is_dense = (num_edges == max_edges)
        
        if is_dense and n_nodes > 200:
            # Use direct Euclidean calculation for dense graphs (Triangle Inequality assumed)
            # Fetch positions
            pos = nx.get_node_attributes(problem.graph, 'pos')
            coords = np.array([pos[i] for i in range(n_nodes)])
            # Vectorized logical distance
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            self.dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        else:
            # Use Dijkstra for sparse graphs
            # Optimization: Use johnson if available or just all_pairs
            # NetworkX all_pairs_dijkstra is decent but slow for python
            raw_dists = dict(nx.all_pairs_dijkstra_path_length(problem.graph, weight='dist'))
            for u in range(n_nodes):
                for v in range(n_nodes):
                    self.dist_matrix[u, v] = raw_dists[u].get(v, float('inf'))

        self.real_golds = nx.get_node_attributes(problem.graph, 'gold')

        self.virtual_cities = []
        self.virtual_map = {} 
        self.node_golds = {}
        self.win_base = 20
        
        # Analytic Setup
        self._configure_virtual_nodes_analytic()
        
        # Robust Seeding: Ensure seeds match virtual nodes
        self._sanitize_seeds()
        
        self.islands = []
        for i, role in enumerate(['Exploit', 'Balanced', 'Explore']):
            isl = Island(role, role, self.pop_size, self)
            
            # Seeding Logic: pass external seeds only if enabled and role is right
            seeds = None
            if self.ablation_config['seeding']:
                if i == 0: # Only seed Exploit island
                    seeds = self.initial_individuals
            
            isl.initialize(seeds=seeds)
            self.islands.append(isl)

        self.global_best = self.islands[0].population[0]
        self.generation_count = 0

    def _configure_virtual_nodes_analytic(self):
        """
        Robust Virtual Node Generation.
        Uses logarithmic scaling for stability and importance-based capping to prevent bloat.
        """
        self.virtual_cities = []
        self.virtual_map = {} 
        self.node_golds = {}
        
        n_cities = self.problem.graph.number_of_nodes()
        beta = self.problem.beta
        alpha = self.problem.alpha

        # 1. Phase Transition Guard
        # Updated Logic: Disable below Beta=1.25 unless forcibly enabled by ablation
        chunking_enabled = self.ablation_config.get('chunking', True)
        
        if beta <= self.multi_hop_threshold or not chunking_enabled:
            vid_counter = 1
            for c in range(1, n_cities):
                self.virtual_cities.append(vid_counter)
                self.virtual_map[vid_counter] = c
                self.node_golds[vid_counter] = self.real_golds[c]
                vid_counter += 1
            
            self.win_base = 20
            # logging.info(f"Chunking Disabled (Beta={beta}, Config={chunking_enabled})")
            return

        # 2. Analytic Split Calculation
        requested_splits = [] 
        
        for c in range(1, n_cities):
            weight = self.real_golds[c]
            dist = self.dist_matrix[0, c]
            
            if dist < 1e-6: 
                k = 1
                imp = 0
            else:
                # Rigorous Calculus-Based Formula
                # C(k) = k*D + k*(alpha*D*g/k)^beta
                # Optimal k = (beta-1)^(1/beta) * alpha * g * D^(1 - 1/beta)
                if beta > 1.001:
                    term1 = (beta - 1.0) ** (1.0 / beta)
                    term2 = alpha * weight
                    term3 = dist ** (1.0 - 1.0 / beta)
                    # DAMPENING FACTOR: Scale optimal k by 0.5 to trade theoretical optimality for GA convergence speed.
                    raw_k = 0.5 * term1 * term2 * term3
                else:
                    raw_k = 1.0
                
                k = int(np.floor(raw_k))
                
                # Priority Score (Cost Reduction Potential)
                # How much do we save by chunking? Approx ~ (alpha*D*G)^beta
                imp = (alpha * weight * dist) ** beta
            
            # Reduce per-city cap from 50 to 40
            k = max(1, min(k, 40))
            requested_splits.append({'c': c, 'k': k, 'imp': imp})

        # 3. Global Safety Cap (Reduced to 400 for Runtime Performance)
        # We use a soft cap that scales down proportional to importance if we exceed budget.
        MAX_TOTAL_NODES = 500
        current_total = sum(x['k'] for x in requested_splits)
        
        if current_total > MAX_TOTAL_NODES:
            # If we exploded the budget (likely with high alpha), 
            # we scale down k proportional to the overshoot, rather than just filling from top.
            # This preserves the *relative* distribution of k_opt.
            scale_factor = MAX_TOTAL_NODES / current_total
            for item in requested_splits:
                if item['k'] > 1:
                    item['k'] = max(1, int(item['k'] * scale_factor))
            
        # 4. Generation
        requested_splits.sort(key=lambda x: x['c'])
        vid_counter = 1
        for item in requested_splits:
            c = item['c']
            k = item['k']
            weight = self.real_golds[c]
            gold_per_chunk = weight / k
            
            for _ in range(k):
                self.virtual_cities.append(vid_counter)
                self.virtual_map[vid_counter] = c
                self.node_golds[vid_counter] = gold_per_chunk
                vid_counter += 1
        
        avg_k = len(self.virtual_cities) / max(1, n_cities)
        self.win_base = int(30 / np.sqrt(avg_k))
        self.win_base = max(5, self.win_base)
        
        factor = len(self.virtual_cities) / max(1, n_cities - 1)
        logging.info(f"CHUNKING FINALIZED: {len(self.virtual_cities)} virtual nodes created (Expansion: {factor:.2f}x). Beta={beta}")

    def _sanitize_seeds(self):
        """
        Ensure initial_individuals match the current set of virtual_cities.
        If chunking is active, expand real-node seeds into virtual-node sequences.
        """
        if not self.initial_individuals:
            return
            
        # Build Reverse Map: Real -> [Virtuals]
        real_to_virtual = {}
        for v_id, r_id in self.virtual_map.items():
            if r_id not in real_to_virtual:
                real_to_virtual[r_id] = []
            real_to_virtual[r_id].append(v_id)
            
        for r_id in real_to_virtual:
            real_to_virtual[r_id].sort()
            
        new_seeds = []
        for seed in self.initial_individuals:
            new_genome = []
            for node in seed:
                if node in self.virtual_map:
                    new_genome.append(node)
                elif node in real_to_virtual:
                    new_genome.extend(real_to_virtual[node])
                else:
                    if node == 0: continue
            
            if len(new_genome) == len(self.virtual_cities):
                new_seeds.append(new_genome)
                
        self.initial_individuals = new_seeds

    def get_dist(self, u, v):
        u_real = 0 if u == 0 else self.virtual_map[u]
        v_real = 0 if v == 0 else self.virtual_map[v]
        return self.dist_matrix[u_real, v_real]
    
    def get_dist_to_base(self, u_virt):
        return self.get_dist(0, u_virt)

    def split_route(self, permutation, win_scale):
        """
        Optimal Split Algorithm with Hybrid Physics.
        """
        n = len(permutation)
        win = int(self.win_base * win_scale)
        win = max(5, win)
        
        # Physics Flags
        # The Solver now always uses the correct Granular Physics + Chunking.
        # "Optmistic" linear calculation (which caused Hallucination) is permanently removed.
        
        V = [float('inf')] * (n + 1)
        V[0] = 0.0
        P = [-1] * (n + 1)
        
        for i in range(n):
            if V[i] == float('inf'): continue
            
            # --- CORE LOGIC ---
            # We must carefully track:
            # 1. `path_cost`: The cost of moving 0 -> ... -> prev (Forward Only)
            # 2. `final_leg`: The cost of moving prev -> 0 (Return)
            # 3. `total_trip_cost`: path_cost + final_leg
            
            # Step 1: Initial Link (0 -> u)
            u = permutation[i]
            d0 = self.get_dist(0, u)
            
            # Initial Path Cost (0 -> u) is always Empty (Linear)
            path_cost = d0
            
            # Initial Return (u -> 0)
            dr = self.get_dist(u, 0)
            current_gold = self.node_golds[u]
            
            # --- START FIXED COST LOGIC ---
            # Multi-Hop Approximation for Beta > 1
            # If we just do (alpha*dr*g)**beta, it's huge.
            # Actual path will use k small hops.
            # Cost approx = k * (dr/k + (alpha * dr/k * g)**beta )
            #             = dr + k * (alpha * dr/k * g)**beta
            #             = dr + k^(1-beta) * (alpha * dr * g)**beta
            
            if self.problem.beta > self.multi_hop_threshold:
                k_approx = max(1.0, dr / self.avg_edge_len)
                penalty_factor = k_approx ** (1.0 - self.problem.beta)
                ret_cost = dr + penalty_factor * ((self.problem.alpha * dr * current_gold) ** self.problem.beta)
            else:
                ret_cost = dr + (self.problem.alpha * dr * current_gold) ** self.problem.beta
            # --- END FIXED COST LOGIC ---

            # Update Bellman for Single Trip
            trip_total = path_cost + ret_cost
            if V[i] + trip_total < V[i+1]:
                V[i+1] = V[i] + trip_total
                P[i+1] = i
            
            # Step 2: Chain Extension (u -> v -> ...)
            limit = min(i + win + 1, n + 1)
            prev = u
            
            for j in range(i + 2, limit):
                new_city = permutation[j-1]
                d_seg = self.get_dist(prev, new_city)
                
                # Segment Cost (prev -> new_city)
                # Carries `current_gold` (load from prev)
                
                if self.problem.beta > self.multi_hop_threshold:
                    k_approx = max(1.0, d_seg / self.avg_edge_len)
                    penalty_factor = k_approx ** (1.0 - self.problem.beta)
                    seg_cost = d_seg + penalty_factor * ((self.problem.alpha * d_seg * current_gold) ** self.problem.beta)
                else:
                    seg_cost = d_seg + (self.problem.alpha * d_seg * current_gold) ** self.problem.beta
                
                # Accumulate Forward Path
                path_cost += seg_cost
                
                # Prepare for next
                current_gold += self.node_golds[new_city]
                prev = new_city
                
                # Calculate New Return (new_city -> 0)
                dr = self.get_dist(new_city, 0)
                
                if self.problem.beta > self.multi_hop_threshold:
                    k_approx = max(1.0, dr / self.avg_edge_len)
                    penalty_factor = k_approx ** (1.0 - self.problem.beta)
                    ret_cost = dr + penalty_factor * ((self.problem.alpha * dr * current_gold) ** self.problem.beta)
                else:
                    ret_cost = dr + (self.problem.alpha * dr * current_gold) ** self.problem.beta
                
                # Update Bellman for Extended Trip
                trip_total = path_cost + ret_cost
                if V[i] + trip_total < V[j]:
                    V[j] = V[i] + trip_total
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

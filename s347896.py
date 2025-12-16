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


# --- constants ---
ISLAND_PRIORS = {
    'Exploit': {'mut_rate': (0.05, 0.15), 'win_scale': (1.0, 1.5), 'mut_mix': [0.1, 0.1, 0.8]}, # Mostly scramble/swap
    'Balanced': {'mut_rate': (0.15, 0.35), 'win_scale': (0.8, 1.2), 'mut_mix': [0.3, 0.3, 0.4]},
    'Explore': {'mut_rate': (0.35, 0.60), 'win_scale': (0.6, 1.0), 'mut_mix': [0.4, 0.4, 0.2]},
    'Chaos': {'mut_rate': (0.60, 0.90), 'win_scale': (0.3, 0.7), 'mut_mix': [0.33, 0.33, 0.33]}
}

class Individual:
    def __init__(self, genome, params, problem_context=None):
        self.genome = genome # Permutation of virtual nodes
        self.params = params # {mut_rate, mut_mix, chunk_scale, win_scale}
        self.cost = float('inf')
        self.trips = []
        self.problem_context = problem_context # Reference to solver for Split
        
        # If context is provided, evaluate immediately
        if self.problem_context:
            self.evaluate()

    def evaluate(self):
        # Use simple caching or just run split (split is O(N*W), reasonably fast now)
        self.cost, self.trips = self.problem_context.split(self.genome, self.params['win_scale'])

    def clone(self):
        # Deep copy params
        new_params = self.params.copy()
        new_params['mut_mix'] = list(self.params['mut_mix'])
        ind = Individual(list(self.genome), new_params, self.problem_context)
        ind.cost = self.cost
        ind.trips = self.trips # Shallow copy trips ok as they are re-generated on eval, but safer to just carry
        return ind

class Island:
    def __init__(self, name, role, pop_size, solver):
        self.name = name
        self.role = role
        self.pop_size = pop_size
        self.solver = solver
        self.population = []
        self.priors = ISLAND_PRIORS[role]
        
    def initialize(self):
        # Initialize population with island-specific priors
        self.population = []
        
        # 1. Smart Init (Far -> Near) for a few individuals
        for _ in range(max(1, int(self.pop_size * 0.1))):
            genome = sorted(self.solver.cities, key=lambda c: self.solver.get_dist(c, 0), reverse=True)
            params = self._sample_params()
            self.population.append(Individual(genome, params, self.solver))
            
        # 2. Random Init
        while len(self.population) < self.pop_size:
            genome = self.solver.cities[:]
            np.random.shuffle(genome)
            params = self._sample_params()
            self.population.append(Individual(genome, params, self.solver))
            
        self.sort_pop()

    def _sample_params(self):
        # Sample from priors
        mut_rate = np.random.uniform(*self.priors['mut_rate'])
        win_scale = np.random.uniform(*self.priors['win_scale'])
        # Start with island's bias but allow drift
        mut_mix = list(self.priors['mut_mix']) 
        return {
            'mut_rate': mut_rate,
            'mut_mix': mut_mix,
            'win_scale': win_scale
        }

    def sort_pop(self):
        self.population.sort(key=lambda x: x.cost)

    def evolve_step(self):
        # Elitism
        new_pop = self.population[:2]
        
        while len(new_pop) < self.pop_size:
            # Tournament
            p1 = self._tournament()
            p2 = self._tournament()
            
            # Crossover (Genome + Params)
            child_genome = self._crossover_genome(p1.genome, p2.genome)
            child_params = self._crossover_params(p1.params, p2.params)
            
            # Mutation (Self-Adaptive)
            # 1. Parameter Mutation (Always)
            child_params = self._mutate_params(child_params)
            
            # 2. Genome Mutation (Probabilistic based on child's mut_rate)
            if np.random.random() < child_params['mut_rate']:
                child_genome = self._mutate_genome(child_genome, child_params['mut_mix'])
                
            child = Individual(child_genome, child_params, self.solver)
            new_pop.append(child)
            
        self.population = new_pop
        self.sort_pop()

    def _tournament(self, k=3):
        candidates = [self.population[i] for i in np.random.randint(0, len(self.population), k)]
        return min(candidates, key=lambda x: x.cost)

    def _crossover_genome(self, p1, p2):
        # OX1
        size = len(p1)
        a, b = sorted(np.random.choice(range(size), 2, replace=False))
        child = [None] * size
        child[a:b+1] = p1[a:b+1]
        
        curr = (b + 1) % size
        p2_idx = (b + 1) % size
        while None in child:
            if p2[p2_idx] not in child[a:b+1]:
                child[curr] = p2[p2_idx]
                curr = (curr + 1) % size
            p2_idx = (p2_idx + 1) % size
        return child

    def _crossover_params(self, p1, p2):
        # Average or Random pick? Average is safer for continuous
        new_params = {}
        alpha = np.random.random()
        for k in ['mut_rate', 'win_scale']:
            new_params[k] = alpha * p1[k] + (1 - alpha) * p2[k]
        
        # Mix for vectors
        mix1 = np.array(p1['mut_mix'])
        mix2 = np.array(p2['mut_mix'])
        new_mix = alpha * mix1 + (1 - alpha) * mix2
        new_params['mut_mix'] = list(new_mix / new_mix.sum())
        return new_params

    def _mutate_params(self, params):
        # Log-normal update
        tau = 0.1
        params['mut_rate'] = params['mut_rate'] * np.exp(tau * np.random.normal())
        params['mut_rate'] = np.clip(params['mut_rate'], 0.05, 0.95)
        
        params['win_scale'] = params['win_scale'] * np.exp(tau * np.random.normal())
        params['win_scale'] = np.clip(params['win_scale'], 0.3, 2.0)
        
        # Mutate mix (dirichlet-like perturbation)
        mix = np.array(params['mut_mix'])
        noise = np.random.normal(0, 0.1, size=len(mix))
        mix = np.abs(mix + noise)
        params['mut_mix'] = list(mix / mix.sum())
        
        return params

    def _mutate_genome(self, genome, mix):
        # Mix: [p_swap, p_inv, p_scramble]
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
    def __init__(self, problem: Problem):
        self.problem = problem
        self.num_cities_real = problem.graph.number_of_nodes()
        self.alpha = problem.alpha
        self.beta = problem.beta
        
        self.real_dists = dict(nx.all_pairs_dijkstra_path_length(problem.graph, weight='dist'))
        self.real_golds = nx.get_node_attributes(problem.graph, 'gold')
        
        # Pre-calc Base Stats for Adaptivity
        all_dists = [d for u in self.real_dists for d in self.real_dists[u].values() if d > 0]
        self.avg_dist = np.mean(all_dists) if all_dists else 1.0
        
        # Penalty Sensitivity
        self.S = (self.alpha ** self.beta) * self.beta * (self.avg_dist ** self.beta) if self.beta > 0 else 1.0
        
        # Density Factor (Low density = 0.1 or so)
        # We were passed density in __init__ of Problem but it's not stored. 
        # Estimate: Edges / (N*(N-1)/2)
        n = self.num_cities_real
        m = problem.graph.number_of_edges()
        if n > 1:
            density = m / (n * (n - 1) / 2)
        else:
            density = 1.0
        self.density_factor = 1.0 / max(density, 0.1)
        
        self.city_factor = np.log(self.num_cities_real + 1)
        
        # Base Chunk Size
        # C ~ 1000
        C = 1000.0
        denom = self.S * self.density_factor * self.city_factor
        self.chunk_base = C / denom if denom > 1e-6 else C
        self.chunk_base = np.clip(self.chunk_base, 20.0, 300.0)
        
        # Base Window
        val = 200.0 / (self.beta * self.density_factor) if self.beta > 0 else 200.0
        self.win_base = int(np.clip(val, 20, min(150, n)))

        # Virtual Nodes Setup
        self.virtual_cities = []
        self.virtual_map = {} 
        self.node_golds = {}
        self._create_virtual_cities_fixed(scale=0.8) 
        self.cities = self.virtual_cities

    def _create_virtual_cities_fixed(self, scale=1.0):
        target_chunk = self.chunk_base * scale
        vid_counter = 1
        for c in range(1, self.num_cities_real):
            gold = self.real_golds[c]
            # Simple splitter
            if gold > 1.2 * target_chunk and self.beta > 1.0:
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
        u_real = 0 if u == 0 else self.virtual_map[u]
        v_real = 0 if v == 0 else self.virtual_map[v]
        return self.real_dists[u_real][v_real]

    def split(self, permutation, win_scale):
        # Dynamic Window logic:
        win = int(self.win_base * win_scale)
        win = max(5, win) # safety
        
        n = len(permutation)
        # 1-based indexing for V and P? No, 0 to n.
        V = [float('inf')] * (n + 1)
        V[0] = 0.0
        P = [-1] * (n + 1)
        
        # Max lookahead
        max_j = n
        
        for i in range(n):
            if V[i] == float('inf'):
                continue
            
            current_load = 0.0
            forward_cost = 0.0
            
            # Start from i
            u = permutation[i]
            # 0 -> u
            d0 = self.get_dist(0, u)
            forward_cost = d0 # weight 0
            
            prev = u
            current_load += self.node_golds[prev]
            
            # Return i->0
            dr = self.get_dist(prev, 0)
            cost_trip = forward_cost + dr + (self.alpha * dr * current_load) ** self.beta
            
            if V[i] + cost_trip < V[i+1]:
                V[i+1] = V[i] + cost_trip
                P[i+1] = i
                
            # Window check
            limit = min(i + win + 1, n + 1)
            
            for j in range(i + 2, limit):
                new_city = permutation[j-1]
                
                # prev -> new
                d_seg = self.get_dist(prev, new_city)
                # penalty for moving to new node
                forward_cost += d_seg + (self.alpha * d_seg * current_load) ** self.beta
                
                # Pick up
                current_load += self.node_golds[new_city]
                prev = new_city
                
                # Return new -> 0
                dr = self.get_dist(new_city, 0)
                total = forward_cost + dr + (self.alpha * dr * current_load) ** self.beta
                
                if V[i] + total < V[j]:
                    V[j] = V[i] + total
                    P[j] = i
                    
        # Reconstruct
        trips = []
        curr = n
        while curr > 0:
            start = P[curr]
            trips.append(permutation[start:curr])
            curr = start
        trips.reverse()
        return V[n], trips

    def solve(self):
        # Island Manager
        islands = []
        roles = ['Exploit', 'Balanced', 'Explore', 'Chaos']
        # Total Pop ~ 60? 15 per island
        pop_per_island = 15
        
        for role in roles:
            islands.append(Island(role, role, pop_per_island, self))
            
        for island in islands:
            island.initialize()
            
        global_best = islands[0].population[0]
        
        generations = 200
        migration_interval = 20
        
        for gen in range(generations):
            # Evolve All
            for island in islands:
                island.evolve_step()
                if island.population[0].cost < global_best.cost:
                    global_best = island.population[0] # Copy reference
                    logging.info(f"Gen {gen} [{island.name}]: New Global Best {global_best.cost:.2f}")
            
            # Migration
            if gen % migration_interval == 0 and gen > 0:
                self._migrate(islands)
                
        # Format output
        formatted_sol = []
        for trip in global_best.trips:
            for city in trip:
                real_city = self.virtual_map[city]
                gold_amount = self.node_golds[city]
                formatted_sol.append((real_city, gold_amount))
            formatted_sol.append((0, 0)) # Return to base
            
        return formatted_sol

    def _migrate(self, islands):
        # Ring topology or complete mixing?
        # Let's do a simple mix: Top 1 from each island goes to next island
        # replacing worst
        migrants = [island.population[0].clone() for island in islands]
        
        for i in range(len(islands)):
            target = islands[(i + 1) % len(islands)]
            migrant = migrants[i]
            
            # Parameter Perturbation post-migration
            # "Light parameter mutation"
            migrant.params = target._mutate_params(migrant.params) 
            
            # Replace worst
            target.population[-1] = migrant
            target.sort_pop()


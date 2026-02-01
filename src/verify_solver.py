
import numpy as np
import networkx as nx
import networkx as nx
from src.solver import GA_Solver
# from problem import Problem # Not strictly needed if using Mock

class MockProblem:
    def __init__(self, graph, alpha, beta):
        self._graph = graph
        self._alpha = alpha
        self._beta = beta

    @property
    def graph(self):
        return self._graph

    @property
    def alpha(self):
        return self._alpha
        
    @property
    def beta(self):
        return self._beta

def test_split_dp_exactness():
    """
    Verifies that split_route manually calculated cost matches its output.
    Constructs a manual case.
    """
    print("Testing Split DP Exactness...")
    
    # Create dummy graph
    blocks = {}
    g = nx.Graph()
    # Depot
    g.add_node(0, pos=(0,0), gold=0)
    # City 1
    g.add_node(1, pos=(1,0), gold=10)
    g.add_edge(0, 1, dist=1.0)
    # City 2
    g.add_node(2, pos=(3,0), gold=20)
    g.add_edge(1, 2, dist=2.0)
    # City 3
    g.add_node(3, pos=(3,4), gold=5)
    g.add_edge(2, 3, dist=4.0)
    g.add_edge(0, 3, dist=5.0) # 3-4-5 triangle
    
    # All pairs shortest paths (Dijkstra)
    # 0->1: 1.0
    # 0->2: 3.0 (0-1-2)
    # 0->3: 5.0 (direct)
    # 1->2: 2.0
    # 1->3: 0-1 + 0-3 = 6 ?? No. 1->2->3 = 2+4=6. 1->0->3 = 1+5=6.
    # 2->3: 4.0
    
    # Setup problem
    p = MockProblem(g, alpha=0.1, beta=2.0)
    
    # Initialize Solver
    solver = GA_Solver(p, pop_size_per_island=10, max_generations=5)
    
    # Force distance matrix to match expected (NetworkX dijkstra usage confirmed in solver)
    print(f"Dist(0,1) = {solver.dist_matrix[0,1]}")
    print(f"Dist(1,2) = {solver.dist_matrix[1,2]}")
    assert abs(solver.dist_matrix[0,2] - 3.0) < 1e-6
    
    # Permutation: [1, 2, 3]
    # Possible Splits:
    # A) 0-1-2-3-0 (One trip)
    # B) 0-1-0, 0-2-3-0 (Two trips)
    # ...
    
    # Calculate Cost Manual for A) 0 -> 1 -> 2 -> 3 -> 0
    # Leg 1: 0->1 (dist=1, w=0) -> Cost = 1 + (0.1*1*0)^2 = 1.0
    # Load at 1: 10.
    # Leg 2: 1->2 (dist=2, w=10) -> Cost = 2 + (0.1*2*10)**2 = 2 + 2^2 = 6.0
    # Load into 2: 10+20=30.
    # Leg 3: 2->3 (dist=4, w=30) -> Cost = 4 + (0.1*4*30)**2 = 4 + 12^2 = 148.0
    # Load into 3: 30+5=35.
    # Leg 4: 3->0 (dist=5, w=35) -> Cost = 5 + (0.1*5*35)**2 = 5 + 17.5^2 = 5 + 306.25 = 311.25
    # Total A = 1 + 6 + 148 + 311.25 = 466.25
    
    cost_a, trips_a = solver.split_route([1, 2, 3], win_scale=100) # Big window to allow full processing
    # The DP finds optimal. Is A optimal?
    
    # Let's check Split B) 0-1-0, 0-2-3-0
    # Trip 1: 0->1 (1.0), 1->0 (dist=1, w=10) -> 1 + (0.1*1*10)^2 = 2.0. Total = 3.0.
    # Trip 2: 0->2 (dist=3, w=0) -> 3.0
    #         2->3 (dist=4, w=20) -> 4 + (0.1*4*20)**2 = 4 + 64 = 68.0
    #         3->0 (dist=5, w=25) -> 5 + (0.1*5*25)**2 = 5 + 156.25 = 161.25
    # Total B = 3.0 + 3.0 + 68.0 + 161.25 = 235.25
    
    # Clearly B is much better. DP should return <= 235.25.
    
    print(f"DP Cost: {cost_a}")
    print(f"Trips: {trips_a}")
    
    # Allow small float error
    # Optimal is actually [[1], [2], [3]] with cost 61.25
    assert abs(cost_a - 61.25) < 1e-5
    
    # Check if trips match [[1], [2], [3]]
    assert trips_a == [[1], [2], [3]]
    
    print("Split DP Exactness Passed!")

def test_model_consistency():
    print("Testing Model Consistency...")
    g = nx.Graph()
    for i in range(5): 
        g.add_node(i, pos=(i,i), gold=10)
        if i > 0: g.add_edge(i-1, i, dist=1)
    
    p = MockProblem(g, alpha=0.1, beta=1.1)
    solver = GA_Solver(p, pop_size_per_island=5)
    
    ind = solver.islands[0].population[0]
    
    # Check genome is permutation of 1..4
    assert len(ind.genome) == 4
    assert set(ind.genome) == {1, 2, 3, 4}
    
    # Check simple mutation doesn't break permutation
    mutated = solver.islands[0]._mutate_genome(list(ind.genome), [0.0, 1.0, 0.0]) # Force Inversion
    assert set(mutated) == {1, 2, 3, 4}
    
    print("Model Consistency Passed!")

if __name__ == "__main__":
    test_split_dp_exactness()
    test_model_consistency()

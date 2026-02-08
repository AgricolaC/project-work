
import unittest
import sys
import os
import numpy as np

# Add project root to sys.path
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from problem import Problem
from src.core.solver import GA_Solver
from src.core.utils import validate_encoded_solution, evaluate_solution

class TestSolverContracts(unittest.TestCase):

    def setUp(self):
        # Create a real Problem instance (with seed for determinism)
        # N=5, complete graph (density=1.0)
        self.p = Problem(5, alpha=0.01, beta=1.0, density=1.0, seed=42)
        self.solver = GA_Solver(self.p, pop_size_per_island=10, max_generations=1, seed=42)

    # Test 1: Encoding Validity 

    def test_validate_encoded_solution_valid(self):
        n_test = 5
        golds = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
        total_gold = 100.0
        
        # 1. Standard Linear Tour
        sol = [(0,0), (1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0), (0,0)]
        valid, msg = validate_encoded_solution(sol, n_test, total_gold, golds)
        self.assertTrue(valid, f"Standard should be valid: {msg}")

        # 2. Pass-Through Semantics (Visit 1 with 0 gold first, then pickup later)
        # 0 -> 1(0) -> 2(20) -> 1(10) -> 3(30) -> 4(40) -> 0
        sol_pt = [(0,0), (1, 0), (2, 20.0), (1, 10.0), (3, 30.0), (4, 40.0), (0,0)]
        valid, msg = validate_encoded_solution(sol_pt, n_test, total_gold, golds)
        self.assertTrue(valid, f"Pass-through should be valid: {msg}")

    def test_validate_encoded_solution_errors(self):
        n_nodes = 4 
        golds = {1: 10.0, 2: 20.0, 3: 30.0}
        total_gold = 60.0
        
        # 1. Bad Terminals
        sol = [(1, 10), (2, 20), (3, 30), (0,0)]
        valid, msg = validate_encoded_solution(sol, n_nodes, total_gold, golds)
        self.assertFalse(valid)
        self.assertIn("must start at depot", msg.lower())
        
        # 2. Missing City
        sol = [(0,0), (1, 10), (2, 20), (0,0)]
        valid, msg = validate_encoded_solution(sol, n_nodes, total_gold, golds)
        self.assertFalse(valid)
        self.assertIn("missing", msg.lower())
        
        # 3. Duplicate City (Pickup)
        sol = [(0,0), (1, 10), (2, 20), (1, 10), (3, 30), (0,0)]
        valid, msg = validate_encoded_solution(sol, n_nodes, total_gold + 10, golds)
        self.assertFalse(valid)
        self.assertIn("multiple times", msg.lower())

        # 4. Wrong Gold Amount (Individual Mismatch)
        sol = [(0,0), (1, 10), (2, 20), (3, 35), (0,0)] # 35 != 30
        valid, msg = validate_encoded_solution(sol, n_nodes, total_gold + 5, golds)
        self.assertFalse(valid)
        self.assertIn("mismatch", msg.lower())

    # Test 2: Action List Correctness 

    def test_action_list_expansion(self):
        # We need to force a known graph structure to verify path expansion.
        # It's hard to rely on random 'p'.
        # Let's override the solver's internal path matrices for this specific test.
        # This is white-box testing of the 'expand_solution_to_action_list' method.
        
        solver = self.solver
        
        # Override Internals
        solver.cities = [1, 2]
        solver.real_golds = np.array([0, 10, 20], dtype=float) # Index 0 dummy
        
        # Distance Matrix & Parents
        # 0 -> 1 -> 2
        
        # L matrix
        solver.L = np.full((3, 3), float('inf'))
        solver.L[0,0] = 0; solver.L[1,1]=0; solver.L[2,2]=0
        # 0-1
        solver.L[0,1] = solver.L[1,0] = 1
        # 1-2
        solver.L[1,2] = solver.L[2,1] = 1
        # 0-2 (via 1)
        solver.L[0,2] = solver.L[2,0] = 2
        
        # Parent Matrix (Predecessor)
        solver.parent = np.full((3, 3), -1, dtype=int)
        
        # Path 0->2: 0->1->2. Parent[0, 2] = 1. Parent[0, 1] = 0.
        solver.parent[0, 2] = 1
        solver.parent[0, 1] = 0
        
        # Path 2->0: 2->1->0. Parent[2, 0] = 1. Parent[2, 1] = 2.
        solver.parent[2, 0] = 1
        solver.parent[2, 1] = 2
        
        # Neighbors (1->0, 1->2)
        solver.parent[1, 0] = 1 # ? No, parent[u, v] is pred of v. 
        # Path 1->0: 1->0. Pred of 0 is 1.
        solver.parent[1, 0] = 1 
        # Path 1->2: 1->2. Pred of 2 is 1.
        solver.parent[1, 2] = 1
        
        trips = [[2]] # Go to 2 only
        
        # Path: 0 -> 1 -> 2 -> 1 -> 0
        # Action list should capture this.
        # But 'expand_solution_to_action_list' uses 'real_golds' and graph connectivity?
        # It uses self.L to check reachability.
        
        actions = solver.expand_solution_to_action_list(trips)
        
        # Check sequence
        # We expect: (0,0), (1,0), (2,20), (1,0), (0,0)
        
        self.assertEqual(len(actions), 5)
        self.assertEqual(actions[0], (0, 0))
        self.assertEqual(actions[1], (1, 0)) # Passthrough 1
        self.assertEqual(actions[2], (2, 20)) # Visit 2 (Target)
        self.assertEqual(actions[3], (1, 0)) # Return via 1
        self.assertEqual(actions[4], (0, 0))

    # Test 3: Cost Invariants 

    def test_cost_evaluation_consistency(self):
        # Verify evaluate_solution against manual calculation
        # Problem: Linear (beta=1), Alpha=0.1
        
        # We can construct a mock problem object purely for this function
        # since it doesn't depend on complex internal state.
        
        class MockGraph:
            def has_edge(self, u, v): return True
        
        class MockProblem:
            graph = MockGraph()
            alpha = 0.1
            beta = 1.0
            def cost(self, path, weight):
                # Simple cost function for test
                # dist = 10
                dist = 10.0
                return dist + (self.alpha * dist * weight) ** self.beta

        p = MockProblem()
        action_list = [(0,0), (1, 5), (0,0)]
        # 0->1: w=0. Cost = 10 + 0 = 10
        # 1->0: w=5. Cost = 10 + (0.1*10*5)^1 = 15
        # Total = 25
        
        cost = evaluate_solution(p, action_list)
        self.assertEqual(cost, 25.0)

    # Test 4: Split DP Sanity

    def test_split_dp_logic(self):
        # Override solver to test exact split logic on known graph
        solver = self.solver
        
        # 2 Cities: 1, 2
        solver.cities = [1, 2]
        solver.real_golds = np.array([0, 10, 10], dtype=float)
        
        # Distances
        solver.L = np.zeros((3, 3))
        # Depot-1 = 10
        solver.L[0,1] = solver.L[1,0] = 10
        # Depot-2 = 10
        solver.L[0,2] = solver.L[2,0] = 10
        # 1-2 = 5
        solver.L[1,2] = solver.L[2,1] = 5
        
        # Sbeta (Penalties) - Zero for simplicity (Alpha=0 in problem)
        solver.Sbeta = np.zeros((3,3)) 
        
        # Mock problem in solver to control alpha/beta
        class MockP:
            alpha = 0.0
            beta = 1.0
        solver.problem = MockP()
        
        # Case 1: Route 1-2 cheap
        # Tour 1-2: 0->1->2->0 => 10 + 5 + 10 = 25
        # Split: (0->1->0) + (0->2->0) => 20 + 20 = 40
        cost, trips = solver.split_route([1, 2], win_scale=None)
        self.assertEqual(trips, [[1, 2]])
        self.assertEqual(cost, 25.0)
        
        # Case 2: Route 1-2 expensive
        solver.L[1,2] = solver.L[2,1] = 100
        # Tour 1-2: 120
        # Split: 40
        cost, trips = solver.split_route([1, 2], win_scale=None)
        self.assertEqual(trips, [[1], [2]])
        self.assertEqual(cost, 40.0)

if __name__ == "__main__":
    unittest.main()

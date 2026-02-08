
import unittest
import numpy as np
import networkx as nx
import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.solver import GA_Solver
from problem import Problem

class MockProblem:
    def __init__(self, alpha=0.1, beta=1.0, n=10):
        self.alpha = alpha
        self.beta = beta
        self.graph = nx.complete_graph(n)
        for i in range(n):
            self.graph.nodes[i]['pos'] = (np.random.rand(), np.random.rand())
            self.graph.nodes[i]['gold'] = 1.0
        for u, v in self.graph.edges():
            self.graph[u][v]['weight'] = 1.0
            self.graph[u][v]['dist'] = 1.0 # Solver uses 'dist'

class TestRhoLogic(unittest.TestCase):
    
    def test_log_rho_basic(self):
        """Verify log_rho matches direct calculation for safe values."""
        alpha = 0.1
        beta = 2.0
        L = 10.0
        w = 5.0
        Sbeta = L ** beta # For single edge
        
        # Direct: ((0.1 * 5)**2 * 10^2) / 10 = (0.25 * 100) / 10 = 2.5
        # Formula: ((alpha * w)^beta * Sbeta) / L
        # (0.5^2 * 100) / 10 = 25 / 10 = 2.5
        
        expected = ((alpha * w) ** beta * Sbeta) / L
        
        log_val = GA_Solver._log_rho(alpha, beta, L, Sbeta, w)
        computed = np.exp(log_val)
        
        self.assertAlmostEqual(computed, expected, places=7)

    def test_log_rho_stability(self):
        """Verify stability with large beta."""
        alpha = 1.0
        beta = 50.0 # Very large
        L = 2.0
        w = 2.0
        Sbeta = L ** beta
        
        # Direct: (1*2)^50 * 2^50 / 2 = HUGE
        # log_rho should be finite
        log_val = GA_Solver._log_rho(alpha, beta, L, Sbeta, w)
        self.assertTrue(np.isfinite(log_val))
        self.assertTrue(log_val > 0)

    def test_pickup_semantics(self):
        """
        Verify compute_rho_solution uses action list correctly.
        Pass-through nodes (gold=0) should NOT increase load.
        """
        problem = MockProblem(alpha=1.0, beta=1.0)
        solver = GA_Solver(problem)
        
        # Mock L matrix and other needs
        solver.L = np.ones((10, 10)) # All distances 1
        solver.problem = problem # Ensure link
        solver.cities = list(range(1, 10))
        solver.real_golds = np.zeros(10) # Just for fallback if needed
        solver.rng = np.random.default_rng(42)
        
        # Actions: 
        # 0 -> 1 (pickup 10) -> 2 (pickup 0/pass) -> 3 (pickup 10)
        actions = [
            (0, 0),
            (1, 10),
            (2, 0),
            (3, 10)
        ]
        
        rho = solver.compute_rho_solution(actions)
        
        # Expected:
        # Edge 0->1: L=1, load=0. rho = 0.
        # Edge 1->2: L=1, load=10. rho = (1*1*10)^1/1 = 10.
        # Edge 2->3: L=1, load=10 (picked 0). rho = 10.
        # Median([10, 10]) = 10. (Filtered out 0 if log_rho handles it well, or kept it?)
        # _log_rho with w=0 returns -inf. 
        # compute_rho_solution filters out non-finite log_rho?
        # Let's check implementation:
        # if np.isfinite(lr): log_rhos.append(lr)
        # So 0->1 is skipped (w=0 -> -inf).
        # We have [10, 10]. Median 10.
        
        self.assertAlmostEqual(rho, 10.0, places=7)

    def test_bucket_thresholds(self):
        """Verify regime buckets."""
        problem = MockProblem()
        solver = GA_Solver(problem)
        solver.problem = problem
        solver.cities = list(range(1, 5))
        solver.rng = np.random.default_rng(42)
        solver.L = np.ones((5,5))
        solver.real_golds = np.ones(5)
        
        # We can just test logic if we extracted it, but let's usage describe_regime
        # Mock compute_rho_global to control output
        solver.compute_rho_global = MagicMock()
        
        # distance
        solver.compute_rho_global.return_value = 0.05 # < 1/9
        regime = solver.describe_regime()
        self.assertEqual(regime['rho_global_bucket'], 'distance')
        
        # mixed
        solver.compute_rho_global.return_value = 1.0
        regime = solver.describe_regime()
        self.assertEqual(regime['rho_global_bucket'], 'mixed')
        
        # penalty
        solver.compute_rho_global.return_value = 100.0
        regime = solver.describe_regime()
        self.assertEqual(regime['rho_global_bucket'], 'penalty')

if __name__ == "__main__":
    unittest.main()

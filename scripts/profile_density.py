
import time
import sys
import os
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problem import Problem
from src.core.solver import GA_Solver

def benchmark_density(n=50, density=0.2):
    print(f"\n--- Benchmark N={n}, Density={density} ---")
    
    # 1. Create Problem
    t0 = time.time()
    p = Problem(n, density=density, seed=42)
    t_prob = time.time() - t0
    print(f"Problem Init: {t_prob:.4f}s | Edges: {p.graph.number_of_edges()}")
    
    # 2. Solver Init (Precomputations)
    t0 = time.time()
    solver = GA_Solver(p, pop_size_per_island=10, max_generations=1)
    t_init = time.time() - t0
    print(f"Solver Init (Precompute): {t_init:.4f}s")
    
    # 3. Evolution Step (Loop)
    t0 = time.time()
    solver.step_generation()
    t_step = time.time() - t0
    print(f"Generation Step: {t_step:.4f}s")
    
    return t_init, t_step

def main():
    # Warmup
    benchmark_density(n=20, density=0.5)
    
    # Compare
    t_sparse_init, t_sparse_step = benchmark_density(n=50, density=0.1)
    t_dense_init, t_dense_step = benchmark_density(n=50, density=0.9)
    
    print("\n--- Summary ---")
    print(f"Init (Precompute) Ratio (Dense/Sparse): {t_dense_init/t_sparse_init:.2f}x")
    print(f"Step Ratio (Dense/Sparse): {t_dense_step/t_sparse_step:.2f}x")

if __name__ == "__main__":
    main()

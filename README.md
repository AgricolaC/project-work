# Computational Intelligence Project - Golden VRP Solver

This repository contains a Genetic Algorithm (GA) implementation designed to solve a variant of the Vehicle Routing Problem (VRP) where the cost function is non-linear and dependent on the carried load.

## Project Description

The problem involves collecting gold from $N$ cities and returning it to a base. The cost of travel is defined as:
$$c(d, w) = d + (\alpha \cdot d \cdot w)^\beta$$
Where:
- $d$ is the distance traveled.
- $w$ is the weight (gold) carried.
- $\alpha$, $\beta$ are parameters that define the problem regime.

High values of $\beta$ create a "convex" cost landscape where carrying heavy loads over long distances is penalized exponentially. Our solver uses a multi-island Genetic Algorithm with specialized "Split-Delivery" strategies to handle these regimes effectively.

## Installation

This project requires Python 3.10+. Dependencies are listed in `requirements.txt`.

**Note for macOS/Linux Users:**
If you encounter an `externally-managed-environment` error, you can install packages to your user directory:

```bash
pip3 install --user --break-system-packages -r requirements.txt
```

Alternatively, use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Module Usage Guidelines

### 1. Main Solver (`s347896.py`)
This is the main entry point for the examination submission. 
**Usage:**
This file is typically imported by the evaluation system. To run it, use the 'solution(p:Problem)' function as a standalone script which receives as input an instance of the class Problem which generates the problem.

### 2. Unit Tests (`src/tests/test_solver.py`)
Ensures the core logic of the solver (encoding, decoding, cost calculation) is correct.

**Usage:**
```bash
python3 -m unittest src/tests/test_solver.py
```

### 3. Ablation Studies (`src/experiments/ablation_study.py`)
Runs a comparative study to evaluate the impact of specific solver components (e.g., LNS, Local Search, Multi-Island).

**Usage:**
```bash
python3 src/experiments/ablation_study.py --output_dir src/results/ablation
```
**Options:**
- `--alphas`, `--betas`, `--densities`: Parameter lists to sweep.
- `--seeds`: List of random seeds.
- `--ns`: List of problem sizes (N).
- `--debug`: Run a fast, minimal verification set.

### 4. Benchmark Grid (`src/experiments/benchmark_grid.py`)
Runs a comprehensive grid search benchmark comparing the GA solver against a baseline strategy.

**Usage:**
```bash
python3 src/experiments/benchmark_grid.py --output_dir src/results/benchmark
```
**Options:** Same as `ablation_study.py`.

### 5. Visualization Pipeline (`src/analysis/make_figures.py`)
Generates publication-ready figures and tables from benchmark results.

**Usage:**
```bash
python3 -m src.analysis.make_figures --csv src/results/tables/benchmark_results.csv --out src/results/figures
```
**Options:**
- `--format`: Output format (png, pdf, svg).
- `--show`: Display plots interactively.
- `--only`: Generate specific figures (e.g., `fig1,fig3`).
- `--latex-table`: Generate a LaTeX summary table.

**Figures Generated:**
1. Cost Breakdown vs Load (Convex Regime)
2. Solver Improvement vs Baseline
3. Trip Load Distribution
4. Runtime Scaling
5. Ablation Study Heatmap

## Code Structure
- **`problem.py`**: Defines the problem environment and cost function.
- **`src/core/solver.py`**: Core GA implementation (Population, Islands, Evolution).
- **`src/core/utils.py`**: Helper functions for metrics and validation.
- **`src/results/`**: Directory where logs and CSV reports are saved.

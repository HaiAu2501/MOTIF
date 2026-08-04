# [AAAI 2026 Oral] MOTIF: <ins>M</ins>ulti-strategy <ins>O</ins>ptimization via <ins>T</ins>urn-based <ins>I</ins>nteractive <ins>F</ins>ramework

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2508.03929-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2508.03929)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A competitive multi-agent framework that evolves combinatorial optimization strategies through LLM-powered turn-based interactions.*

</div>

---

## 📰 1. News

<div align="center">

| Date | Update |
|:----:|:-------|
| 🎉 **Dec. 2025** | Refactored codebase for better modularity and extensibility |
| 🏆 **Nov. 2025** | Paper accepted for **oral presentation** at AAAI 2026! |
| 🚀 **Aug. 2025** | Released first version of MOTIF |

</div>

---

> [!CAUTION]
> We discovered that, during a recent refactoring and prompt optimization effort, we unintentionally altered several core components of the MOTIF algorithm. As a result, the current implementation no longer faithfully matches the original method described in the paper, leading to degraded performance. **We are working to fix this quickly and restore the implementation to the best-performing version of MOTIF. Thank you to the community for bringing this to our attention, and we apologize for the inconvenience.**


---

## 🔍 2. Overview

**MOTIF** is a turn-based, multi-agent framework for improving combinatorial optimization solvers by jointly evolving multiple algorithmic strategies rather than tuning a single heuristic.

<div align="center">
<img src="./assets/overview.png" alt="MOTIF Overview"/>
</div>

### Key Features

- 🎮 **Competitive-Collaborative Learning**: Two LLM agents take alternating turns to refine components
- 📊 **Dynamic Baselines**: Performance guided by adaptive baseline comparisons
- 🔄 **Opponent Feedback**: Each agent learns from the other's improvements
- 🧩 **Structured Operators**: Modular prompts enable targeted strategy refinement
- 🌐 **Broad Search Space**: Competitive dynamics encourage diverse adaptations

---

## 🚀 3. Quick Start

### Step 1. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY="your-openai-api-key-here"
```

### Step 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3. Run Default Experiment

```bash
python main.py
```

### Step 4. Run Custom Experiments

**Specify a solver:**

```bash
python main.py solver=tsp_aco
python main.py solver=cvrp_dr_f1_f2_f3
```

**Configure MCTS parameters:**

```bash
python main.py \
       solver=tsp_aco \
       mcts.outer_iterations=20 \
       mcts.inner_iterations=10 \
       mcts.final_iterations=10
```

**Customize LLM settings:**

```bash
python main.py \
       llm.model=gpt-4o \
       llm.temperature=0.8
```

### Step 5. Results

Results are saved to `./results/`:

- `F*_final_best.py`: Final optimized strategy implementations
- `*_round_*.json`: Detailed experiment logs with performance metrics

---

## 📋 4. Supported Problems and Solvers

<div align="center">

| Solver | Problems | Strategies |
|:-------|:--------|:-----------|
| ACO | TSP, CVRP, MKP | - **F1**: Heuristic & Pheromone Initialization<br>- **F2**: Probabilistic Transition Rule<br>- **F3**: Pheromone Update Rule |
| ACO | OP, BPP | - **F1**: Heuristic & Pheromone Initialization<br>- **F2**: Pheromone Update Rule |
| GLS | TSP | - **F1**: Guide Matrix Initialization |
| DR  | TSP, CVRP, BPP | - **F1**: Initial Solution Construction Rule<br>- **F2**: Deconstruction Rule<br>- **F3**: Repair Rule |

</div>

---

## 🔧 5. Customization and Extension

MOTIF is designed for easy extension. Follow these guides to add new problems and solvers.

### Step 1. Create the problem directory:

```
problems/
└── problem_solver/
    ├── __init__.py
	├── solver.py        # Solver implementation
	├── eval.py          # Evaluation functions
	├── generator.py     # Instance generator
	├── prompts.py       # LLM prompts for each strategy
	├── F1.py            # Strategy component 1 (baseline)
	├── F2.py            # Strategy component 2 (baseline)
	├── F3.py            # Strategy component 3 (baseline)
	└── datasets/        # Training and test datasets
```

### Step 2. Implement core files:

- **`prompts.py`**: Define `PROBLEM_DESCRIPTION`, `CONSTRAINTS`, and strategy prompts (`F1`, `F2`, `F3`)
- **`F*.py`**: Provide baseline implementations for each strategy component

### Step 3. Create solver configuration:

```yaml
# @package _global_.solver
base_path: ${paths.problems_dir}/problem_solver # Path to problem-solver directory
functions: [F1, F2] # Strategy files to optimize
```

---

## 📚 6. Citation

If you find MOTIF useful in your research, please consider citing our paper:

```bibtex
@inproceedings{nguyen2026motif,
  title     = {MOTIF: Multi-Strategy Optimization via Turn-based Interactive Framework},
  author    = {Kiet, Nguyen Viet Tuan and Tung, Dao Van and Dao, Tran Cong and Binh, Huynh Thi Thanh},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026},
  address   = {Singapore},
  month     = {January},
  note      = {Oral Presentation}
}
```

---

<div align="center">

Made with ❤️ for the optimization research community!

</div>

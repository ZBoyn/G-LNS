# G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Abstract

While Large Language Models (LLMs) have recently shown promise in Automated Heuristic Design (AHD), existing approaches typically formulate AHD around constructive priority rules or parameterized local search guidance, thereby restricting the search space to fixed heuristic forms. Such designs offer limited capacity for structural exploration, making it difficult to escape deep local optima in complex Combinatorial Optimization Problems (COPs). 

In this work, we propose **G-LNS**, a generative evolutionary framework that extends LLM-based AHD to the automated design of **Large Neighborhood Search (LNS)** operators. Unlike prior methods that evolve heuristics in isolation, G-LNS leverages LLMs to co-evolve tightly coupled pairs of *destroy* and *repair* operators. A cooperative evaluation mechanism explicitly captures their interaction, enabling the discovery of complementary operator logic that jointly performs effective structural disruption and reconstruction. Extensive experiments on challenging COP benchmarks, such as **Traveling Salesman Problems (TSP)** and **Capacitated Vehicle Routing Problems (CVRP)**, demonstrate that G-LNS significantly outperforms LLM-based AHD methods as well as strong classical solvers. The discovered heuristics not only achieve near-optimal solutions with reduced computational budgets but also exhibit robust generalization across diverse and unseen instance distributions.

![Motivation](figures/introduction.png)

## 🚀 Key Features

- **Generative LNS Design**: Goes beyond parameter tuning or constructive rules by generating executable code for both *destroy* and *repair* operators, enabling structural algorithmic innovation.
- **Synergy-Aware Co-evolution**: Introduces a **Synergy Matrix** to explicitly model and exploit the coupling between destroy and repair operators during evolution.
- **Dual-Population Architecture**: Maintains distinct populations for destroy ($\mathcal{P}_d$) and repair ($\mathcal{P}_r$) operators, evolving them through adaptive selection and joint crossover.
- **Robust Generalization**: Discovered heuristics exhibit strong performance across diverse and unseen instance distributions.

## 🏗️ Methodology

![Method Overview](figures/method.png)

The G-LNS framework operates in a cyclic process consisting of four phases:

1.  **Initialization**: Dual populations are seeded with domain expertise and LLM-generated operators.
2.  **Evaluation**: Operators are dynamically selected and scored via an **Adaptive LNS (ALNS)** process with multi-episode evaluation to handle stochasticity.
3.  **Population Management**: Operators are ranked based on **Global Fitness**, and low-performing ones are pruned.
4.  **Evolution**: LLMs perform **Mutation** (Logic Evolution, Parameter Calibration) and **Crossover** (Homogeneous, Synergistic Joint) to replenish the population.

## 📂 Repository Structure

- **```base/```**: Base classes
- **```datasets/```**: Dataset generation
- **```examples/```**: Training scripts and configurations
- **```heuristics/```**: Huristics designed by G-LNS
- **```task/```**: Problem definitions and interfaces
- **```tools/```**: Utilities (LLM API, Profiler, etc.)

## 🛠️ Quick Start

### Configuration

Create a `.env` file in the root directory to configure your LLM API. You can check `tools/llm` for supported APIs.

```env
DEESEEK_API_HOST=https://api.deepseek.com
DEESEEK_API_KEY=your_api_key_here
DEESEEK_API_MODEL=deepseek-chat
```

### Running Experiments

To run G-LNS on the **Traveling Salesman Problem (TSP)**:

```bash
python examples/tsp/run_tsp.py
```

The logs and generated operators will be saved in `examples/tsp/logs/`.

## 🙏 Acknowledgments

This project is built upon the [LLM4AD](https://github.com/Optima-CityU/LLM4AD) platform. We thank the authors for their open-source contribution.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

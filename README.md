# Reinforcement Learning – Dynamic Programming in Maze Environments

This repository contains a project on **Dynamic Programming (DP)** applied to discrete Markov Decision Processes (MDPs), completed as part of the Symbolic AI course at Leiden University (2024).  
We implement **Value Iteration (VI)** and **Q-value Iteration (QI)** to solve maze environments with multiple goals, keys, and doors.

---

## 📖 Project Overview
- **Environment**: Maze defined in text files (e.g., `prison.txt`) with:
  - Free cells, walls, keys, doors, and goals
  - States represented by agent position and key possession
  - Actions: {up, down, left, right}
  - Rewards: −1 per step, positive rewards at goals
  - Multiple goals scenario to test optimal policy selection
- **Algorithms studied**:
  - **Value Iteration (VI)**: Computes optimal state-value function
  - **Q-value Iteration (QI)**: Computes optimal state-action value function
- **Experiments**:
  - Single and multiple goals
  - Observing agent behavior when multiple goals differ in reward and distance
  - Analysis of convergence, state space size, and computational efficiency
  - Comparison to iterative deepening tree search

---

## 🔑 Key Findings
- **Multiple goals**: Agent prioritizes higher reward or closer goals; unintended termination may occur if a lower-reward goal is on the path to a higher-reward goal due to environment definition.
- **Curse of dimensionality**: Number of states grows exponentially with number of keys; e.g., a 100×100 maze with 15 keys leads to ~3.28×10⁸ states.
- **Efficiency**: Dynamic Programming is empirically much faster than tree search approaches, solving problems in seconds versus hours.
- **Policy representation**: DP stores solutions as tables of values, allowing full analysis of all possible paths, whereas tree/graph search stores sequences of states for direct execution.

---

## 📂 Repository Contents
- `Report.pdf` and `Report.pages` – Full report of the assignnment
- `Assignment.pdf` - An assignment statement specifying the tasks to complete. 
- `src/` – Python code for environment utilities and policy execution.
  - `dynamic_programming.py` – Implementation of Value Iteration and Q-value Iteration.  
  - `prison_nr.txt` – Maze environment definitions with multiple goals.  

---

## 🚀 Getting Started
Clone the repository and install dependencies:

```bash
git clone https://github.com/Reymer249/Markov-Decision-Process.git
cd Markov-Decision-Process
pip install -r requirements.txt

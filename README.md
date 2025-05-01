# Gomoku AI with Renju Rules 🧠♟️

This is an AI-driven **Gomoku (Five in a Row)** game implemented with **Python 3.13.2** and a PyQt6 graphical interface. It supports human-vs-AI and AI-vs-AI modes under **Renju rules**, featuring three distinct AI models ranging from classic heuristics to deep reinforcement learning.

## ✨ Features

- ✅ **Renju rule enforcement** 
- 🎮 **PyQt6 GUI** for interactive play
- 🤖 Three powerful AI models:
  1. **Minimax with heuristic evaluation and Alphabeta pruning**
  2. **Monte Carlo Tree Search with heuristic rollout**
  3. **Neural Network (Policy + Value) trained via self-play using MCTS**

---

## 🧠 AI Models

### 1. Minimax + Heuristic
- Classic tree search
- Static board evaluation heuristics
- Depth-limited for tractability

### 2. MCTS + Heuristic
- Tree search with playout simulations
- Rollout policy based on simple board heuristics
- Balances exploration & exploitation

### 3. MCTS + Neural Network (Policy + Value)
- Unified neural network for move probability (policy) and win prediction (value)
- Trained through reinforcement learning from self-play
- Currently trained through 1,000 self played games
- Similar to AlphaZero-style architecture

---

## 📦 Dependencies

- Python `3.13.2`
- NumPy `2.2.2`
- PyQt6 `6.8.1`
- PyTorch `2.6.0`

Install them with:

```bash
pip install numpy==2.2.2 PyQt6==6.8.1 torch==2.6.0
```

## 🚀 Getting Started
```bash
python gomoku_gui.py
```


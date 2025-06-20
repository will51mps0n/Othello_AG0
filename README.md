# Othello_AG0

A reinforcement learning agent trained to play Othello using a simplified AlphaZero framework. It combines Monte Carlo Tree Search (MCTS) with a deep convolutional neural network for board evaluation and policy prediction.

## Features
- Full AlphaZero training loop with self-play
- MCTS for move planning
- Convolutional Neural Network (CNN) for board evaluation
- Game engine and Arena for match evaluation
- Command-line interface to pit trained agents

## Technologies
- Python
- PyTorch
- NumPy

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Train the model:
python pit.py

2. Evaluate matches:
python Arena.py

3. Folder Structure:
MCST.py: MCTS logic
NeuralNet.py: CNN model
Game.py: Othello logic
Arena.py: Match runner

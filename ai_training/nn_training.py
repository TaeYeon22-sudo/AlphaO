import os
import math
import torch
import torch.optim as optim
import numpy as np
from nn_deeplearning import GomokuNet
from nn_mcts import (
    MCTSNode,
    mcts_simulation,
    board_to_tensor,
    get_allowed_moves,
    move_to_index,
    make_move,
    is_terminal,
)

# Set device: use "cuda" if GPU is available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

##############################################
# Self-Play and Training Functions
##############################################

def select_move_from_root(root, temperature=1.0):
    """
    Select a move from the root node using the visit count distribution.
    temperature = 0 corresponds to deterministic (argmax), 
    higher temperatures yield a more probabilistic selection.
    """
    board_size = len(root.board)
    num_moves = board_size * board_size
    counts = np.zeros(num_moves, dtype=np.float32)
    for move, child in root.children.items():
        counts[move_to_index(move, root.board)] = child.visits

    if temperature == 0:
        move_index = np.argmax(counts)
    else:
        # Apply temperature to the counts.
        counts = counts ** (1.0 / temperature)
        total = np.sum(counts)
        if total == 0:
            probs = np.ones(num_moves, dtype=np.float32) / num_moves
        else:
            probs = counts / total
        move_index = np.random.choice(np.arange(num_moves), p=probs)

    row = move_index // board_size
    col = move_index % board_size
    return (row, col), counts / np.sum(counts)

def self_play_game(model, num_simulations=100):
    """
    Plays a single self-play game guided by MCTS and the neural network.
    Returns a list of training examples.
    
    Each example is a tuple (board_state, mcts_policy, outcome)
    where:
      - board_state is the board configuration (a 2D list),
      - mcts_policy is the improved move probabilities from MCTS (flattened to [num_moves]),
      - outcome is the game result from the perspective of the player (+1 win, -1 loss, 0 draw).
    """
    game_data = []   # Stores tuples: (board, mcts_policy, player)
    board_size = 15
    # Initialize an empty board: 0 for empty, 1 for player 1, -1 for player -1.
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    current_player = 1  # Let player 1 start (e.g., black).

    while True:
        # Create the MCTS root node for the current board state.
        root = MCTSNode(board, current_player)
        # Run a series of MCTS simulations.
        for _ in range(num_simulations):
            mcts_simulation(root, model, current_player)
        
        # Derive the improved move distribution (policy) from visit counts.
        board_moves = board_size * board_size
        move_counts = np.zeros(board_moves, dtype=np.float32)
        for move, child in root.children.items():
            move_index = move_to_index(move, board)
            move_counts[move_index] = child.visits

        if np.sum(move_counts) > 0:
            mcts_policy = move_counts / np.sum(move_counts)
        else:
            # Fallback: if no moves were visited, use a uniform distribution over allowed moves.
            allowed_moves = get_allowed_moves(board, current_player)
            mask = np.zeros(board_moves, dtype=np.float32)
            for move in allowed_moves:
                mask[move_to_index(move, board)] = 1.0
            mcts_policy = mask / mask.sum()

        # Record the current board state, policy, and current player.
        game_data.append((board, mcts_policy, current_player))
        
        # Select the next move from the root using the improved probabilities.
        move, _ = select_move_from_root(root, temperature=1.0)
        board = make_move(board, move, current_player)
        
        # Check if the game has ended.
        terminal, winner = is_terminal(board)
        if terminal:
            break

        # Switch the player for the next turn.
        current_player = -current_player

    # Convert the collected game data to training examples.
    training_examples = []
    for state, policy, player in game_data:
        # Outcome is assigned from the perspective of the player who made the move.
        if winner is None:
            outcome = 0
        else:
            outcome = 1 if winner == player else -1
        training_examples.append((state, policy, outcome))
    return training_examples

def play_self_games(model, num_games=10, num_simulations=100):
    """
    Run multiple self-play games to collect training examples.
    """
    all_examples = []
    for game in range(num_games):
        examples = self_play_game(model, num_simulations)
        all_examples.extend(examples)
        print(f"Completed game {game+1}/{num_games}")
    return all_examples

##############################################
# Training Loop
##############################################

def loss_func(pred_policy, pred_value, target_policy, target_value, model, l2_reg=1e-4):
    """
    Combined loss function:
      - Policy loss: Negative log-likelihood (cross-entropy) using target probabilities.
      - Value loss: Mean squared error between predicted value and the actual outcome.
      - Regularization: L2 penalty on the network parameters.
    """
    # pred_policy is in log probabilities.
    policy_loss = -torch.mean(torch.sum(target_policy * pred_policy, dim=1))
    value_loss = torch.mean((pred_value.view(-1) - target_value)**2)
    l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())
    return policy_loss + value_loss + l2_reg * l2_loss

def train_model(model, training_data, epochs=10, batch_size=32, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    # Shuffle training data.
    np.random.shuffle(training_data)
    num_examples = len(training_data)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = math.ceil(num_examples / batch_size)
        for batch_idx in range(num_batches):
            batch = training_data[batch_idx * batch_size : (batch_idx+1) * batch_size]
            state_batch = []
            policy_targets = []
            value_targets = []
            for board, mcts_policy, outcome in batch:
                # Convert board state to a tensor.
                # Here we use a canonical view (from player 1's perspective).
                state_tensor = board_to_tensor(board, current_player=1)
                state_batch.append(state_tensor)
                policy_targets.append(torch.tensor(mcts_policy, dtype=torch.float32))
                value_targets.append(torch.tensor(outcome, dtype=torch.float32))
            # Stack into batches and move to GPU.
            state_batch = torch.stack(state_batch).to(device)
            policy_targets = torch.stack(policy_targets).to(device)
            value_targets = torch.stack(value_targets).to(device)

            pred_policy, pred_value = model(state_batch)
            loss = loss_func(pred_policy, pred_value, policy_targets, value_targets, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}")

##############################################
# Main Training Routine
##############################################

if __name__ == '__main__':
    board_size = 15
    model = GomokuNet(board_size=board_size, input_channels=3, num_res_blocks=5, num_filters=64)
    model.to(device)

    ckpt_dir = os.path.join(os.path.dirname(__file__), "ai_training", "trained_data")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load latest checkpoint
    last_iter = 0
    for i in range(100, 0, -1):
        ckpt_path = os.path.join(ckpt_dir, f"model_checkpoint_iter_{i}.pth")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            last_iter = i
            print(f"✅ Loaded checkpoint from iteration {i}")
            break
    else:
        print("⚠️ No checkpoint found. Starting from scratch.")

    total_iterations = 100
    for iteration in range(last_iter, total_iterations):
        print(f"\nIteration {iteration+1}/{total_iterations}: Self-play phase")
        training_examples = play_self_games(model, num_games=10, num_simulations=100)

        print("Training phase")
        train_model(model, training_examples, epochs=10, batch_size=32, learning_rate=1e-3)

        ckpt_path = os.path.join(ckpt_dir, f"model_checkpoint_iter_{iteration+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint to {ckpt_path}")


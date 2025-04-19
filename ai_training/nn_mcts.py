import math
import copy
import torch
import numpy as np
from ai_training.nn_deeplearning import GomokuNet
from ai_training.nn_renju_rule import check_winner, is_allowed_move  # Import terminal-state functions

##############################################
# Helper functions for board encoding and moves
##############################################

def board_to_tensor(board, current_player):
    """
    Convert the board (2D list) to a tensor for the network.
    Assumes board values:
      - 1 for player 1,
      - -1 for player -1,
      - 0 for empty.
    Uses a 3-channel encoding:
      1. Current player's stones.
      2. Opponent's stones.
      3. A constant channel (e.g., all ones).
    """
    board_size = len(board)
    current_board = np.array([[1 if cell == current_player else 0 for cell in row] for row in board], dtype=np.float32)
    opponent_board = np.array([[1 if cell == -current_player else 0 for cell in row] for row in board], dtype=np.float32)
    turn_board = np.ones((board_size, board_size), dtype=np.float32)
    tensor = np.stack([current_board, opponent_board, turn_board])
    return torch.tensor(tensor)

def get_allowed_moves(board, stone):
    """
    Returns a list of allowed moves (as (row, col) tuples) based on the rules.
    A move is allowed if the cell is empty and passes the rules defined in nn_renju_rule.
    """
    allowed = []
    board_size = len(board)
    for y in range(board_size):
        for x in range(board_size):
            if board[y][x] == 0 and is_allowed_move(board, (y, x), stone):
                allowed.append((y, x))
    return allowed

def create_move_mask(board, allowed_moves):
    """
    Create a flattened mask (length board_size*board_size) for allowed moves.
    """
    board_size = len(board)
    mask = np.zeros((board_size, board_size), dtype=np.float32)
    for (y, x) in allowed_moves:
        mask[y][x] = 1.0
    return mask.flatten()

def move_to_index(move, board):
    """
    Convert a move (row, col) to a flat index.
    """
    board_size = len(board)
    y, x = move
    return y * board_size + x

def is_terminal(board):
    """
    Check for terminal state using the rules.
    Returns (True, winner) if terminal:
      - winner: 1 or -1 for a win; 0 for a draw.
    Returns (False, None) if the game is still in progress.
    """
    winner = check_winner(board)
    if winner is not None:
        return True, winner
    return False, None

def make_move(board, move, player):
    """
    Return a new board state after applying the move (row, col) for the given player.
    """
    new_board = [row.copy() for row in board]
    y, x = move
    new_board[y][x] = player
    return new_board

##############################################
# MCTS Node and Simulation Functions
##############################################

class MCTSNode:
    def __init__(self, board, current_player, parent=None, move=None):
        self.board = board              # Board state (2D list)
        self.current_player = current_player  # 1 or -1
        self.parent = parent            # Parent node
        self.move = move                # Move that led to this node
        self.children = {}              # Dictionary: move -> child node
        self.visits = 0
        self.value = 0.0
        self.priors = {}              # Prior probability for each move
        self.expanded = False

    def create_child(self, move, child_board, prior):
        """Create and add a child node for a given move."""
        child_node = MCTSNode(child_board, -self.current_player, parent=self, move=move)
        self.children[move] = child_node
        self.priors[move] = prior

    def select_child(self, c_puct=1.0):
        """Select a child using the PUCT formula."""
        best_score = -float('inf')
        best_move = None
        best_child = None
        for move, child in self.children.items():
            Q = child.value / (child.visits + 1e-6)
            U = c_puct * self.priors[move] * math.sqrt(self.visits + 1) / (1 + child.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def update(self, value):
        """Update statistics of the node."""
        self.visits += 1
        self.value += value

def evaluate_state(board, model, current_player):
    """
    Evaluate a board state using the neural network.
    Returns:
      - A probability distribution over moves (flattened and masked by allowed moves),
      - A value estimate for the state.
    """
    board_tensor = board_to_tensor(board, current_player)
    # Move the tensor to the same device as the model.
    board_tensor = board_tensor.to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        # Add a batch dimension.
        log_policy, value = model(board_tensor.unsqueeze(0))
    raw_policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
    allowed_moves = get_allowed_moves(board, current_player)
    mask = create_move_mask(board, allowed_moves)
    policy = raw_policy * mask
    if policy.sum() > 0:
        policy /= policy.sum()
    else:
        policy = mask / mask.sum()
    return policy, value.item()

def mcts_simulation(node, model, current_player):
    """
    Run a single MCTS simulation from the given node, guided by the neural network.
    """
    terminal, winner = is_terminal(node.board)
    if terminal:
        # Return simulation result from the perspective of the original current player.
        return 1 if winner == current_player else -1

    allowed_moves = get_allowed_moves(node.board, current_player)
    if not allowed_moves:
        return 0  # Consider it a draw if no moves remain.

    if not node.expanded:
        policy, value = evaluate_state(node.board, model, node.current_player)
        for move in allowed_moves:
            child_board = make_move(node.board, move, node.current_player)
            prior = policy[move_to_index(move, node.board)]
            node.create_child(move, child_board, prior)
        node.expanded = True
        return value

    selected_move, selected_child = node.select_child()
    simulation_result = mcts_simulation(selected_child, model, current_player)
    node.update(simulation_result)
    return simulation_result

##############################################
# Example: Running MCTS with Neural Guidance
##############################################

if __name__ == '__main__':
    # Create an empty 15×15 board: 0 for empty, 1 for player 1, -1 for player -1.
    board_size = 15
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    current_player = 1  # For example, player 1 starts.

    # Initialize the neural network model.
    model = GomokuNet(board_size=board_size, input_channels=3, num_res_blocks=5, num_filters=64)
    # Optionally load a pretrained model checkpoint:
    # model.load_state_dict(torch.load('model.pth'))
    
    # (Assuming the model has already been moved to the proper device by your training script,
    #  or you could call model.to(device) here if needed.)

    # Create the MCTS root node.
    root = MCTSNode(board, current_player)

    # Run a number of MCTS simulations.
    simulations = 100
    for _ in range(simulations):
        mcts_simulation(root, model, current_player)

    # Choose the best move based on visit count.
    best_move = None
    best_visits = -1
    for move, child in root.children.items():
        if child.visits > best_visits:
            best_visits = child.visits
            best_move = move

    print("Best move selected by MCTS:", best_move)
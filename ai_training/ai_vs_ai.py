import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import copy
import torch
from ai_training.minimax import Minimax
from ai_training.mcts.mcts_agent import MCTSAgent, GomokuState
from ai_training.nn_deeplearning import GomokuNet
from ai_training.nn_mcts import board_to_tensor
from renju_rule import check_if_win, is_double_three, is_double_four, is_overline

BOARD_SIZE = 15

def is_valid_move(board, row, col, player):
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return False
    if board[row][col] != 0:
        return False
    if player == 1:
        if (is_double_three(board, row, col, player) or
            is_double_four(board, row, col, player) or
            is_overline(board, row, col)):
            return False
    return True

class NNAIWrapper:
    def __init__(self, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.net = GomokuNet().to(device)
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.eval()

    def get_best_move(self, board, player):
        board_tensor = board_to_tensor(board, player).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.net(board_tensor)
            policy = torch.softmax(policy_logits, dim=1).view(-1)

        legal = [i for i in range(BOARD_SIZE * BOARD_SIZE)
                 if is_valid_move(board, i // BOARD_SIZE, i % BOARD_SIZE, player)]
        if not legal:
            return None

        best_idx = max(legal, key=lambda i: policy[i].item())
        return best_idx // BOARD_SIZE, best_idx % BOARD_SIZE
    
class MCTSWrapper:
    def __init__(self, iterations=10, max_playout_depth=7):
        self.agent = MCTSAgent(iterations=iterations, max_playout_depth=max_playout_depth)

    def get_best_move(self, board, player):
        state = GomokuState(copy.deepcopy(board), player)
        return self.agent.select_move(state)

def play_game(ai_black, ai_white):
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 1

    while True:
        move = ai_black.get_best_move(copy.deepcopy(board), current_player) if current_player == 1 else \
               ai_white.get_best_move(copy.deepcopy(board), current_player)

        if move is None or not is_valid_move(board, move[0], move[1], current_player):
            return 0  # Draw

        r, c = move
        board[r][c] = current_player

        if check_if_win(board, r, c, current_player):
            return current_player

        current_player *= -1

def evaluate(ai1, ai2, label1, label2, num_games):
    ai1_wins = 0
    ai2_wins = 0
    draws = 0

    for i in range(num_games):
        result = play_game(ai1, ai2) if i % 2 == 0 else play_game(ai2, ai1)
        if result == 1:
            winner = label1 if i % 2 == 0 else label2
        elif result == -1:
            winner = label2 if i % 2 == 0 else label1
        else:
            winner = "Draw"

        if winner == label1:
            ai1_wins += 1
        elif winner == label2:
            ai2_wins += 1
        else:
            draws += 1

        print(f"Game {i+1:02}: Winner - {winner}")

    print(f"\n===== Results: {label1} vs {label2} =====")
    print(f"{label1} Wins: {ai1_wins} / {num_games} ({ai1_wins / num_games:.1%})")
    print(f"{label2} Wins: {ai2_wins} / {num_games} ({ai2_wins / num_games:.1%})")
    print(f"Draws:        {draws} / {num_games} ({draws / num_games:.1%})")
    print("========================================\n")

def run_all_matchups(num_games=10):
    print(f"🔁 Running all matchups with {num_games} games each...\n")

    minimax = Minimax(depth=3)
    # mcts = MCTSAgent(iterations=10, max_playout_depth=7)
    mcts = MCTSWrapper(iterations=10, max_playout_depth=10)
    model_path = os.path.join(os.path.dirname(__file__), "trained_data", "model_checkpoint_iter_100.pth")
    nn_ai = NNAIWrapper(model_path)

    evaluate(minimax, mcts, "Minimax", "MCTS", num_games)
    evaluate(minimax, nn_ai, "Minimax", "NN", num_games)
    evaluate(mcts, nn_ai, "MCTS", "NN", num_games)

if __name__ == "__main__":
    # 게임 횟수를 여기서 지정
    run_all_matchups(num_games=10)

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import copy
import time
from ai_training.minimax import Minimax
from ai_training.mcts_agent import MCTSAgent, GomokuState
from renju_rule import check_if_win, is_overline, is_invalid
from ai_training.rule import Rule
import numpy as np

BOARD_SIZE = 15

def is_valid_move(board, move, stone):
    """
    Dummy example for move legality.
    You can expand this function according to your Renju rules.
    Returns True if the cell at move (y, x) is empty.
    """
    y, x = move
    rule = Rule(board)
    
    # The cell must be empty
    if board[y][x] != 0:
        return False

    # Only check for forbidden moves if the stone is black.
    if stone == 1:
        if rule.is_invalid(y, x) or rule.double_four(x, y, stone) or rule.double_three(x, y, stone):
            return False

    return True

def play_game():
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    current_player = 1  # Minimax starts first (Black)
    minimax_ai = Minimax(depth=3)
    mcts_ai = MCTSAgent(iterations=10)  # Adjust iteration for performance

    turn = 0
    while True:
        turn += 1
        print(f"\nTurn {turn}: Player {current_player}'s move")

        if current_player == 1:
            move = minimax_ai.get_best_move(copy.deepcopy(board), current_player)
        else:
            state = GomokuState(copy.deepcopy(board), current_player)
            move = mcts_ai.select_move(state)

        if move is None:
            print("No valid moves. It's a draw!")
            break

        row, col = move
        if not is_valid_move(board, move, current_player):
            print(f"Invalid move by player {current_player} at ({row}, {col}). Skipping turn.")
            current_player *= -1
            continue

        board[row][col] = current_player
        print(f"Player {current_player} placed at ({row}, {col})")

        if check_if_win(board, row, col, current_player):
            winner = "Minimax (Black)" if current_player == 1 else "MCTS (White)"
            print(f"\n🎉 Game Over! Winner: {winner}")
            break

        current_player *= -1

if __name__ == "__main__":
    play_game()
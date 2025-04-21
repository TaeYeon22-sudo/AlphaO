import sys
import os
# Compute the parent directory relative to this file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import numpy as np
from renju_rule import is_double_four, is_double_three, is_invalid, check_if_win, check_list


def check_winner(board):
    """
    Scan the entire board to determine if a winning condition exists.
    Returns:
      - 1 if player 1 wins,
      - -1 if player -1 wins,
      - 0 if the board is full and no win (draw),
      - None if the game is still in progress.
    """
    board_size = len(board)
    for y in range(board_size):
        for x in range(board_size):
            if board[y][x] != 0:
                color = board[y][x]
                if check_if_win(board, y, x, color):
                    return color
    # If board full and no winning sequence, declare a draw.
    if np.all(np.array(board) != 0):
        return 0
    return None

def is_allowed_move(board, move, stone):
    """
    Dummy example for move legality.
    You can expand this function according to your Renju rules.
    Returns True if the cell at move (y, x) is empty.
    """
    y, x = move
    
    # The cell must be empty
    if board[y][x] != 0:
        return False

    # Only check for forbidden moves if the stone is black.
    if stone == 1:
        if is_invalid(y, x) or is_double_four(board, y, x, stone) or is_double_three(board, y, x, stone):
            return False

    return True

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now you can import renju_rule
from renju_rule import board, ban, is_invalid


import copy
import random
# from renju_rule import board, ban, is_invalid
from ai_training.rule import Rule

BOARD_SIZE = 15
list_dx = [-1, 1, -1, 1, 0, 0, 1, -1]
list_dy = [0, 0, -1, 1, -1, 1, -1, 1]
r = Rule(board)

def heuristic_evaluation(state, move):
    """
    기본 휴리스틱 평가 함수:
    - move 위치의 8방향 인접 셀 중, 현재 플레이어의 돌이 몇 개 있는지 셉니다.
    """
    y, x = move
    bonus = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < BOARD_SIZE and 0 <= nx < BOARD_SIZE:
            if state.board[ny][nx] == state.current_player or state.board[ny][nx] == -state.current_player:
                bonus += 1
    return bonus



def threat_blocking_score(state, move):
    """
    상대 돌이 연속으로 3개 이상 있는 경우 또는 2개이고 open 상태인 경우,
    해당 move가 위협을 차단할 가능성이 있으므로 높은 보너스 점수를 부여합니다.

    4가지 주요 방향(수평, 수직, 두 대각선)마다 검사하여 각 방향의 bonus를 합산합니다.
    """
    # 중요도 순서
    # current_player 가 흑돌에 금수 위치(-100,000)
    # 내가 돌을 놓으면 5 (100,000)
    # 상대가 4, open4 (49,000)   -> X
    # 내가 돌을 놓으면 open4 (24,000)  -> X
    # 내가 돌을 놓으면 4 (10,000)
    # 상대가 open3 (4,000)
    # 상대가 3 (1,000)
    # 내가 돌을 놓으면 open3 (400)
    # 내가 돌을 놓으면 3 (100)
    # 인접 8방에 돌이 있으면 (1) -> heuristic_evaluation

    r = Rule(copy.deepcopy(state.board))
    current_player = state.current_player
    opponent = -current_player
    y,x = move
    bonus = 0

    # 금수 검사
    if current_player == 1 and r.forbidden_point(x, y, 1):
        return -100000

    # 나에게 유리한지 판단: 현재 돌을 놓은 후의 상태 검사
    r.set_stone(x, y, current_player)
    for direction in range(4):
        if r.is_five(x, y, current_player):
            return 100000
        if r.open_four(x, y, current_player, direction):
            bonus += 24000
        if r.four(x, y, current_player, direction):
            bonus += 5000
        if r.open_three(x, y, current_player, direction):
            bonus += 3000
        if r.three(x, y, current_player, direction):
            bonus += 1000
    # 원래 상태로 복원
    r.set_stone(x, y, 0)

    # 위험도 판단: 상대 돌에 대한 체크
    r.set_stone(x, y, opponent)
    for direction in range(4):
        if r.is_five(x, y, opponent):
            return 49000
        if r.open_four(x, y, opponent, direction):
            bonus += 10000
        if r.four(x, y, opponent, direction):
            bonus += 4000
        if r.open_three(x, y, opponent, direction):
            bonus += 2000
        if r.three(x, y, opponent, direction):
            bonus += 1000
    return bonus



def heuristic_policy(state):
    """
    휴리스틱 정책:
    - state의 유효한 후보 수(valid moves) 각각에 대해,
      기본 휴리스틱 점수와 위협 차단 점수(threat_blocking_score)를 합산합니다.
    - 최고 점수를 가진 후보들 중 하나를 랜덤하게 선택합니다.
    - 모든 후보의 점수가 0이면 단순 랜덤 선택합니다.
    """
    moves = state.get_valid_moves()
    list.sort(moves)
    if not moves:
        return None
    scores = []
    for move in moves:
        base_score = heuristic_evaluation(state, move)
        block_score = threat_blocking_score(state, move)
        total_score = base_score + block_score
        scores.append(total_score)
    max_score = max(scores)
    best_moves = [move for move, score in zip(moves, scores) if score == max_score]
    if best_moves:
        return random.choice(best_moves)
    return random.choice(moves)

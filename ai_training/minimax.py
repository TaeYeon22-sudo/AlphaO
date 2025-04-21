import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from typing import Tuple, List, Optional
from renju_rule import is_double_three, is_double_four, is_overline, check_if_win
import copy

class Minimax:
    def __init__(self, depth: int = 3, beam_width: Optional[int] = 10):
        """
        Gomoku AI를 초기화한다.
        
        Args:
            depth (int): 미니맥스 탐색 트리의 깊이 (기본값: 3)
            beam_width (Optional[int]): 각 노드에서 고려할 후보 수 (기본값: 10)
        
        돌 색상: 흑 = 1, 백 = -1, 빈칸 = 0
        
        Renju 규칙에 따른 8방향 중 실제 평가에서는 4방향(그리고 그 반대쪽)만 활용.
        """
        self.depth = depth
        self.beam_width = beam_width
        # 트랜스포지션 테이블: (board_hash, depth, maximizing_player, color) -> (eval, best_move)
        self.transposition_table = {}

        # 8방향 벡터 중 실제로 사용할 4방향과 그 반대쪽
        self.list_dx = [-1, 1, -1, 1, 0, 0, 1, -1]
        self.list_dy = [0, 0, -1, 1, -1, 1, -1, 1]

    def _board_hash(self, board: List[List[int]]) -> Tuple[Tuple[int]]:
        """보드를 해시 가능한 튜플 형태로 변환."""
        return tuple(tuple(row) for row in board)

    def evaluate_window(self, window: List[int], color: int) -> int:
        """
        5칸짜리 윈도우(정수 리스트: 1, -1, 0)를 평가하여 점수를 반환한다.
    
        Args:
            window: 1(흑), -1(백), 0(빈 칸)을 포함하는 길이 5의 리스트.
            color: 평가할 플레이어의 색 (1: 흑, -1: 백).
    
        Returns:
            해당 윈도우에 대한 누적 점수 (int)
        """
        score = 0
        pattern_scores = {
            'forbidden': -100000,
            'five': 100000,
            'open_four_attack': 24000,
            'open_four_defense': 49000,
            'closed_four': 10000,
            'four_completion': 5000,
            'opponent_open_three': 4000,
            'opponent_three': 1000,
            'three_strong_expandable': 600,
            'open_three_attack': 400,
            'three_attack': 100,
            'three_weak_expandable': 57,
            'two_expand_high': 55,
            'two_expand_mid': 50,
            'two_expand_low': 35,
            'two_unique_five': 30,
            'two_block_edge': 30,
            'one_expand_high': 13,
            'one_blank_middle': 10,
            'one_expand_low': 7,
            'one_unique_five': 5,
            'one_block_edge': 5,
            'adjacent_bonus': 1,
            'center_control': 500
        }

        opponent = -color
        if color in window and opponent in window:
            return 0

        count_self = window.count(color)
        if count_self > 0:
            if count_self == 5:
                score += pattern_scores['five']
            elif count_self == 4:
                empty_index = window.index(0)
                if empty_index in [0, 4]:
                    score += pattern_scores['open_four_attack']
                else:
                    score += pattern_scores['four_completion']
            elif count_self == 3:
                indices = [i for i, cell in enumerate(window) if cell == color]
                if indices == [1, 2, 3] and window[0] == 0 and window[4] == 0:
                    score += pattern_scores['open_three_attack']
                elif max(indices) - min(indices) == 2:
                    score += pattern_scores['three_attack']
                else:
                    score += pattern_scores['three_strong_expandable']
            elif count_self == 2:
                indices = [i for i, cell in enumerate(window) if cell == color]
                gap = indices[1] - indices[0]
                if gap == 1:
                    if indices[0] > 0 and indices[1] < 4:
                        score += pattern_scores['two_expand_high']
                    else:
                        score += pattern_scores['two_block_edge']
                elif gap == 2:
                    score += pattern_scores['two_expand_mid']
                else:
                    score += pattern_scores['two_expand_low']
            elif count_self == 1:
                index = window.index(color)
                if index == 2:
                    score += pattern_scores['one_blank_middle']
                elif index in [0, 4]:
                    score += pattern_scores['one_block_edge']
                elif index in [1, 3]:
                    score += pattern_scores['one_expand_high']
                else:
                    score += pattern_scores['one_expand_low']

        count_opp = window.count(opponent)
        if count_opp > 0:
            if count_opp == 5:
                score -= pattern_scores['five']
            elif count_opp == 4:
                empty_index = window.index(0)
                if empty_index in [0, 4]:
                    score -= pattern_scores['open_four_defense']
                else:
                    score -= pattern_scores['closed_four']
            elif count_opp == 3:
                indices = [i for i, cell in enumerate(window) if cell == opponent]
                if indices == [1, 2, 3] and window[0] == 0 and window[4] == 0:
                    score -= pattern_scores['opponent_open_three']
                elif max(indices) - min(indices) == 2:
                    score -= pattern_scores['opponent_three']
                else:
                    score -= pattern_scores['opponent_three']
            elif count_opp == 2:
                indices = [i for i, cell in enumerate(window) if cell == opponent]
                gap = indices[1] - indices[0]
                if gap == 1:
                    if indices[0] > 0 and indices[1] < 4:
                        score -= pattern_scores['two_expand_high']
                    else:
                        score -= pattern_scores['two_block_edge']
                elif gap == 2:
                    score -= pattern_scores['two_expand_mid']
                else:
                    score -= pattern_scores['two_expand_low']
            elif count_opp == 1:
                index = window.index(opponent)
                if index == 2:
                    score -= pattern_scores['one_blank_middle']
                elif index in [0, 4]:
                    score -= pattern_scores['one_block_edge']
                elif index in [1, 3]:
                    score -= pattern_scores['one_expand_high']
                else:
                    score -= pattern_scores['one_expand_low']
                    
        return score

    def evaluate_board(self, board: List[List[int]], color: int) -> int:
        """
        전체 보드 상태를 평가하여 총 점수를 반환한다.
        
        Args:
            board: 15x15 2차원 리스트 형태의 현재 게임 보드
            color: 평가할 플레이어의 색 (1: 흑, -1: 백)
        
        Returns:
            보드의 총 평가 점수 (int)
        """
        total_score = 0
        for direction in range(4):
            for i in range(2):
                idx = direction * 2 + i
                dx, dy = self.list_dx[idx], self.list_dy[idx]
                for row in range(15):
                    for col in range(15):
                        window = []
                        for k in range(5):
                            new_row = row + k * dy
                            new_col = col + k * dx
                            if 0 <= new_row < 15 and 0 <= new_col < 15:
                                window.append(board[new_row][new_col])
                            else:
                                break
                        if len(window) == 5:
                            total_score += self.evaluate_window(window, color)
        return total_score

    def get_valid_moves(self, board: List[List[int]], color: int) -> List[Tuple[int, int]]:
        """
        현재 보드에서 유효한 수(빈 칸 중 인접에 돌이 존재하는 칸)를 반환한다.
        흑돌(1)의 경우 Renju 규칙에 따른 금수를 배제한다.
        
        Args:
            board: 현재 게임 보드 (2차원 리스트)
            color: 현재 플레이어의 색 (1: 흑, -1: 백)
        
        Returns:
            유효한 수의 좌표 리스트 [(row, col), ...]
        """
        # returns (7,7) if empty board
        board_has_stone = any(cell != 0 for row in board for cell in row)
        if not board_has_stone:
            return [(7, 7)]

        valid_moves = []
        for i in range(15):
            for j in range(15):
                if board[i][j] == 0:
                    has_neighbor = False
                    for direction in range(4):
                        for side in range(2):
                            idx = direction * 2 + side
                            for dist in range(1, 3):
                                ni = i + self.list_dy[idx] * dist
                                nj = j + self.list_dx[idx] * dist
                                if 0 <= ni < 15 and 0 <= nj < 15 and board[ni][nj] != 0:
                                    has_neighbor = True
                                    break
                            if has_neighbor:
                                break
                        if has_neighbor:
                            break
                    if has_neighbor:
                        if color == 1:
                            if (is_double_three(board, i, j, color) or 
                                is_double_four(board, i, j, color) or 
                                is_overline(board, i, j)):
                                continue
                        valid_moves.append((i, j))
        return valid_moves

    def get_ordered_moves(self, board: List[List[int]], color: int, maximizing_player: bool) -> List[Tuple[int, int]]:
        """
        get_valid_moves에서 반환된 후보들을 간단한 평가 함수(evaluate_board)를 통해 정렬하고,
        beam_width에 따라 상위 후보만 반환한다.
        
        Args:
            board: 현재 보드 상태 (2차원 리스트)
            color: 이동할 돌의 색 (최대화 턴이면 기준 색, 최소화 턴이면 상대 색)
            maximizing_player: 최대화 턴이면 True, 최소화 턴이면 False
        
        Returns:
            (row, col) 형태의 후보 수 목록 (정렬된 순서)
        """
        valid_moves = self.get_valid_moves(board, color)
        moves_with_scores = []
        for move in valid_moves:
            r, c = move
            board[r][c] = color if maximizing_player else -color
            score = self.evaluate_board(board, color)
            board[r][c] = 0
            moves_with_scores.append((score, move))
        moves_with_scores.sort(key=lambda x: x[0], reverse=maximizing_player)
        if self.beam_width is not None:
            moves_with_scores = moves_with_scores[:self.beam_width]
        ordered_moves = [move for _, move in moves_with_scores]
        return ordered_moves

    def minimax(self,
                board: List[List[int]],
                depth: int,
                alpha: float,
                beta: float,
                maximizing_player: bool,
                color: int) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        알파-베타 가지치기를 적용한 미니맥스 알고리즘.
        트랜스포지션 테이블을 활용하여 중복 탐색을 방지한다.
        
        Args:
            board: 현재 보드 상태 (2차원 리스트)
            depth: 탐색 깊이
            alpha, beta: 가지치기용 경계값
            maximizing_player: 최대화 턴이면 True, 최소화 턴이면 False
            color: 기준 플레이어의 색 (1: 흑, -1: 백)
        
        Returns:
            (평가 점수, (row, col)) 튜플. (row, col)은 최적의 수이며, 없으면 None.
        """
        # 캐시 확인
        board_key = self._board_hash(board)
        tt_key = (board_key, depth, maximizing_player, color)
        if tt_key in self.transposition_table:
            return self.transposition_table[tt_key]

        if depth == 0:
            eval_val = self.evaluate_board(board, color)
            self.transposition_table[tt_key] = (eval_val, None)
            return eval_val, None

        if maximizing_player:
            valid_moves = self.get_ordered_moves(board, color, True)
        else:
            valid_moves = self.get_ordered_moves(board, -color, False)
        if not valid_moves:
            eval_val = self.evaluate_board(board, color)
            self.transposition_table[tt_key] = (eval_val, None)
            return eval_val, None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for (r, c) in valid_moves:
                board[r][c] = color
                if check_if_win(board, r, c, color):
                    board[r][c] = 0
                    self.transposition_table[tt_key] = (100000, (r, c))
                    return 100000, (r, c)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False, color)
                board[r][c] = 0
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (r, c)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # 가지치기
            self.transposition_table[tt_key] = (max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for (r, c) in valid_moves:
                board[r][c] = -color
                if check_if_win(board, r, c, -color):
                    board[r][c] = 0
                    self.transposition_table[tt_key] = (-100000, (r, c))
                    return -100000, (r, c)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True, color)
                board[r][c] = 0
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (r, c)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # 가지치기
            self.transposition_table[tt_key] = (min_eval, best_move)
            return min_eval, best_move

    def get_best_move(self, board: List[List[int]], color: int) -> Optional[Tuple[int, int]]:
        """
        현재 보드 상태에서 최적의 수를 구하여 반환한다.
        
        Args:
            board: 15x15 형태의 게임 보드 (2차원 리스트)
            color: 현재 플레이어의 색 (1: 흑, -1: 백)
        
        Returns:
            (row, col) 형태의 최적의 수. 단, 흑금수인 경우 다른 유효한 수를 반환.
        """
        # 매 검색 시작 시 트랜스포지션 테이블 초기화 (또는 필요에 따라 재사용)
        self.transposition_table.clear()
        best_score, best_move = self.minimax(
            board,
            self.depth,
            alpha=float('-inf'),
            beta=float('inf'),
            maximizing_player=True,
            color=color
        )
        
        if best_move:
            row, col = best_move
            if color == 1:
                if (is_double_three(board, row, col, color) or 
                    is_double_four(board, row, col, color) or 
                    is_overline(board, row, col)):
                    valid_moves = self.get_valid_moves(board, color)
                    if valid_moves:
                        return valid_moves[0]
                    return None
        return best_move

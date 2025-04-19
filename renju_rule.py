import copy
from itertools import count

import numpy as np
from numpy.matlib import empty

board = np.zeros((15, 15), dtype=int)
ban = []
list_dx = [-1, 1, -1, 1, 0, 0, 1, -1]
list_dy = [0, 0, -1, 1, -1, 1, -1, 1]


# TODO: function to check - tie if no space left, white win if black cannot place
def is_board_full(self, y, x):
    return False


# y,x 를 좌표를 받았을 때 이게 out of bound 인지 확인하는 함수
# is_valid 하는게 딱히 없으면 is_valid 로 바꿀 예정
def is_invalid(y, x):
    return x < 0 or x > 14 or y < 0 or y > 14

#####################################
# 이거 이름을 check_list_for_win() 같은걸로 바꾸는게?
#####################################
# only checks for winning status
def check_list(color, line):
    maxN = 0
    count = 0
    for i in range(len(line)):
        # check white is 5 (white can have 5+ in a row)
        if color == -1 and count == 5:
            return True

        if line[i] == color:
            count += 1
            maxN = max(maxN, count)
        else:
            count = 0

    if maxN == 5:
        return True

    return False


def check_if_win(gui_board, y, x, color):
    '''
    checks for winning state
    returns True if win, if not False
    '''

    # 가로줄 ; This removes edge cases
    if (gui_board[y][max(0,x-1)] == color) or (gui_board[y][min(14,x+1)] == color):  # TODO: is this needed??
        # normal case or edge cases
        if ((x != 14 and x != 0) or (x == 14 and gui_board[y][x-1] == color) or (x == 0 and gui_board[y][x+1] == color)):
            # checking line reset
            line = []
            for i in range(max(0, x-5), min(15, x+6), 1):
                # double checks index out of bound error
                if 0 <= y < 15 and 0 <= i < 15:
                    line.append(gui_board[y][i])
            # check for 5-win
            if check_list(color, line): return True

    # 세로줄 (14,12)
    if (gui_board[max(0, y - 1)][x] == color) or (gui_board[min(14, y + 1)][x] == color):
        if ((y != 14 and y != 0) or (y == 14 and gui_board[y-1][x] == color) or (y == 0 and gui_board[y+1][x] == color)):
            # checking line reset
            line = []
            for i in range(max(0, y - 5), min(15, y + 6), 1):
                if 0 <= i < 15 and 0 <= x < 15:
                    line.append(gui_board[i][x])
            # check for 5-win
            check = check_list(color, line)
            if check:
                return True

    # 좌측 상단 + 우측 하단 대각선
    if (gui_board[max(0, y - 1)][max(0, x - 1)] == color) or (gui_board[min(14, y + 1)][min(14, x + 1)] == color):
        # if ((y != 14 and y != 0 and x != 14 and x != 0) or ((y == 0 or x == 0) and gui_board[y+1][x+1] == color) or ((y == 14 or x == 14) and gui_board[y-1][x-1] == color)):
        # checking line reset
        line = []
        for i in range(-5 , 6 , 1):
            # double checks index out of bound error
            if 0 <= y + i < 15 and 0 <= x + i < 15:
                line.append(gui_board[y+i][x+i])
        # check for 5-win
        if check_list(color, line): return True

    # 좌측 하단 + 우측 상단
    if (gui_board[min(14, y + 1)][max(0, x - 1)] == color) or (gui_board[max(0, y - 1)][min(14, x + 1)] == color):
        # if ((y != 14 and y != 0 and x != 14 and x != 0) or ((y == 14 or x == 0) and gui_board[y-1][x+1]) or ((y == 0 or x == 14) and gui_board[y+1][x-1])):
        # checking line reset
        line = []
        for i in range(-5 , 6 , 1):
            # double checks index out of bound error
            if 0 <= y + i < 15 and 0 <= x - i < 15:
                line.append(gui_board[y+i][x-i])
        # check for 5-win
        if check_list(color, line): return True


    # if not 5 stone
    return False

# 흑돌이 둔 후, 금수가 생기는지 확인
# 금수가 생기면 ban 리스트에 저장
def find_prohibit_point(gui_board, y, x):
    cnt = 0
    empty = 0

    # list_dx = [-1, 1, -1, 1, 0, 0, 1, -1]
    # list_dy = [0, 0, -1, 1, -1, 1, -1, 1]

    for j in range(4):
        for i in range(2):
            cur_y, cur_x = y, x
            while True:

                if empty == 2:
                    break

                # Empty
                if gui_board[cur_y][cur_x] == 0:
                    empty += 1
                    if is_double_three(gui_board, cur_y, cur_x) or is_double_four(gui_board, cur_y, cur_x) or is_overline(gui_board, cur_y,cur_x):
                        ban.append({cur_y, cur_x})
                # Black
                elif gui_board[cur_y][cur_x] == 1:
                    cnt += 1
                # White
                elif gui_board[cur_y][cur_x] == -1:
                    break

                cur_y, cur_x = y + list_dy[j * 2 + i], x + list_dx[j * 2 + i]


# 백돌을 둠으로써 금수가 해제되는지 확인
def check_prohibit_point(gui_board, ban):
    if len(ban) == 0: return

    for i in ban:
        y, x = i
        if is_double_three(gui_board, y, x) or is_double_four(gui_board, y, x) or is_overline(gui_board, y, x):
            ban.remove(i)


# 8방향에서 흑돌, 첫번째 공백만 담는 리스트 생성
# 그 리스트에서 오목을 만들 수 있는지 확인
# 오목을 2개 이상 만들 수 있으면 4-4로 간주하고 리턴
def is_double_four(gui_board, y, x, stone):
    if stone == -1:
        return False
    four_cnt = 0
    # 양쪽 공백 나올때까지 저장(공백도 같이 저장)
    for j in range(4):
        # if open_four(y,x) == 2: return True
        line = [1]
        center = 0
        for i in range(2):
            empty_cnt = 0
            cur_y, cur_x = y, x
            while True:
                cur_y += list_dy[j * 2 + i]
                cur_x += list_dx[j * 2 + i]

                if is_invalid(cur_y, cur_x): break
                if gui_board[cur_y][cur_x] == 0:
                    empty_cnt += 1
                    if empty_cnt == 2:
                        break
                elif gui_board[cur_y][cur_x] == -1:
                    break

                if i == 0:
                    line.append(gui_board[cur_y][cur_x])
                else:
                    line.insert(0, gui_board[cur_y][cur_x])
                    center += 1
    
        make_five_check = make_five_row(line)
        if make_five_check >= 2:
            return True
        elif make_five_check == 1:
            four_cnt += 1
        
        if four_cnt >= 2: return True

    make_five_check = make_five_row(line)
    if make_five_check >= 2:
        return True
    elif make_five_check == 1:
        four_cnt += 1
    
    if four_cnt >= 2: return True

    return False

# 받은 리스트에서 오목을 만들 수 있는지 확인
def make_five_row(line):
    if len(line) < 6:
        return 0

    total_count = 0
    before_find = False
    for i in range(0, len(line) - 4):
        if before_find:
            before_find = False
            continue
        cnt = 0
        zero_count = 0
        # starting 부터 5개 체크
        for j in range(i, i + 6, 1):
            if j >= len(line): break
            if line[j] == 0:
                zero_count += 1
                cnt += 1
            elif line[j] == 1:
                cnt += 1
            if zero_count >= 2:
                break
        if cnt == 5:
            total_count += 1
            before_find = True

    return total_count
            


# 장목 확인
# 4방향 확인해서 돌이 6개 이상 인지 확인
def is_overline(gui_board, y,x):
    for j in range(4):
        cnt = 1
        # 양쪽의 돌을 같이 세서 한번에 확인
        for i in range(2):
            cur_y, cur_x = y, x
            while True:
                cur_y, cur_x = cur_y + list_dy[j * 2 + i], cur_x + list_dx[j * 2 + i]
                # out of bound check
                if is_invalid(cur_y, cur_x): break

                # board 에 다른 돌이면 탈출
                if gui_board[cur_y][cur_x] == 0 or gui_board[cur_y][cur_x] == -1:
                    break
                # 흑돌 일때만 카운트
                else:
                    cnt += 1
        # 6개 이상이면 장목
        if cnt > 5:
            return True

    return False


def is_double_three(board, y, x, color):
    if color != 1:
        return False

    cnt = 0
    gui_board = copy.deepcopy(board)
    gui_board[y][x] = 1

    for direction in range(4):
        open_three = False
        # for i in range(2):
        idx = direction * 2  # + i
        yy, xx = list_dy[idx], list_dx[idx]
        ny, nx = y, x
            
        # 한쪽 방향 탐색
        line = [1]  # 현재 위치의 돌
        empty_cnt = 0
        black_stone_cnt = 0
        white_stone_cnt = 0

        while True:
            ny += yy
            nx += xx

            if is_invalid(ny, nx):
                break

            if gui_board[ny][nx] == 0:
                line.append(0)
                empty_cnt += 1
            elif gui_board[ny][nx] == 1:
                line.append(1)
                black_stone_cnt += 1
            elif gui_board[ny][nx] == -1:
                line.append(-1)
                white_stone_cnt += 1
            else:  # 백돌이면 중단
                break
                
            if empty_cnt >= 2 or black_stone_cnt > 2 or white_stone_cnt >= 1:  # 빈칸이나 돌이 너무 많으면 중단
                break
            
        # 반대 방향 탐색
        ny, nx = y, x
        empty_cnt = 0
        black_stone_cnt = 0
        white_stone_cnt = 0

            
        while True:
            ny -= yy
            nx -= xx
            
            if is_invalid(ny, nx):
                break

            if white_stone_cnt > 2:
                break
                
            if gui_board[ny][nx] == 0:
                line.insert(0, 0)
                empty_cnt += 1
            elif gui_board[ny][nx] == 1:
                line.insert(0, 1)
                black_stone_cnt += 1
            elif empty_cnt <= 3 and gui_board[ny][nx] == -1:
                line.insert(0,-1)
                white_stone_cnt += 1
            else:
                break
                
            if empty_cnt >= 2 or black_stone_cnt > 2 or white_stone_cnt >= 1:
                break
            
        # check if open-3
        if is_open_three(line):
            open_three = True
        
        
        if open_three:
            cnt += 1
            if cnt >= 2:
                # print("is SAMSAM")
                return True
            

    gui_board[y][x] = 0
    return False

# returns True if open 3, else False
def is_open_three(line):
    # return False if closed 3
    if '-101110-1' in ''.join(map(str,line)):
        return False
    if len(line) >= 4:  # 최소 4칸 이상
        # 33 패턴들
        if (line[0] == 0 and line[-1] == 0):  # 양쪽이 열려있어야 함
            stone_count = sum(1 for s in line if s == 1)
            if stone_count == 3:  # 돌이 정확히 3개
                # 연속 패턴: ○●●●○
                consecutive = 0
                max_consecutive = 0
                for stone in line:
                    if stone == 1:
                        consecutive += 1
                        max_consecutive = max(max_consecutive, consecutive)
                    else:
                        consecutive = 0
                        
                # 불연속 패턴: ○●●○●○, ○●○●●○
                if max_consecutive >= 2 or '0110' in ''.join(map(str, line)) or '1010' in ''.join(map(str, line)):
                    return True
    
    return False
                        

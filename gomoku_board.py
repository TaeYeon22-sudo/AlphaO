import sys, os
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from ai_training.mcts_agent import GomokuState, MCTSAgent
from renju_rule import *
import time
from ai_training.minimax import Minimax
from renju_rule import check_if_win
import torch
from ai_training.nn_deeplearning import GomokuNet
from ai_training.nn_mcts import board_to_tensor

# default size for board and stone
BOARD_SIZE = 15
CELL_SIZE = 40
STONE_SIZE = 15


class GomokuBoard(QWidget):
    game_over_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = 1
        self.last_move = None
        self.selected_move = None
        self.parent_widget = parent

############  MINIMAX ##########################################################################################
        # self.ai_model = "minimax_model"                                                                      #
        self.ai = Minimax(depth=3)                                                                             #
        self.is_ai_turn = False                                                                                #
                                                                                                               #
        # AI related attributes                                                                                #
        self.ai = Minimax(depth=3)                                                                             #
############  MINIMAX ##########################################################################################

        self.is_ai_enabled = False
        self.is_ai_turn = False

        # trained_nn model loading
        model_path = os.path.join(os.path.dirname(__file__), "ai_training", "trained_data", "model_checkpoint_iter_20.pth")
        # use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = GomokuNet().to(device)
        # checkpoint = torch.load(model_path, map_location=device)
        # self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.load_state_dict(torch.load(model_path, map_location=device))

        self.net.eval()
        self.device = device
        
        # TODO : make the size to be flexible : if window becomes smaller, the board scales down too
        self.setFixedSize(BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE)
        
        #################################################################
        # TODO ??? self.setFixedSize(self.sizeHint())
        # TODO : using image background, need to work on making exact placements
        # self.board_image = QPixmap(os.path.join(os.path.dirname(__file__), "board_img.png"))
        self.background_img = QPixmap(os.path.join(os.path.dirname(__file__), "addets", "white_background.png"))
        #################################################################


    def paintEvent(self, event):
        """Draws the board grid and stones."""
        painter = QPainter(self)
        # antialiasing to make lines smoother
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # painter.setBackground(Qt.GlobalColor.white)

        #################################################################
        # TODO : using image backgounrd
        # scaled_board = self.board_image.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        scaled_board = self.background_img.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        painter.drawPixmap(0, 0, scaled_board)
        #################################################################

        # Draw grid
        pen = QPen(Qt.GlobalColor.black, 2)
        painter.setPen(pen)
        for i in range(BOARD_SIZE):
            painter.drawLine(i * CELL_SIZE + CELL_SIZE // 2, CELL_SIZE // 2,
                           i * CELL_SIZE + CELL_SIZE // 2, (BOARD_SIZE - 1) * CELL_SIZE + CELL_SIZE // 2)
            painter.drawLine(CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2,
                           (BOARD_SIZE - 1) * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2)

        # draw stones
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] != 0:  # 빈 공간이 아닌 경우
                    # 돌 색상 설정
                    if self.board[row][col] == 1:
                        painter.setBrush(QBrush(Qt.GlobalColor.black))
                    else:
                        painter.setBrush(QBrush(Qt.GlobalColor.white))
                    
                    # 돌 중심 좌표 계산
                    x = col * CELL_SIZE + CELL_SIZE // 2
                    y = row * CELL_SIZE + CELL_SIZE // 2
                    # 돌 그리기
                    painter.drawEllipse(QPoint(x, y), STONE_SIZE, STONE_SIZE)
                    
                    # highlight
                    if self.last_move == (row, col):
                        highlight_pen = QPen(Qt.GlobalColor.red, 3)
                        painter.setPen(highlight_pen)
                        painter.drawEllipse(QPoint(x, y), STONE_SIZE + 3, STONE_SIZE + 3)
                        painter.setPen(pen)

        # shadowing before pressing "place" button
        if self.selected_move and self.board[self.selected_move[0]][self.selected_move[1]] == 0:
            row, col = self.selected_move
            painter.setBrush(QBrush(Qt.GlobalColor.gray))
            x = col * CELL_SIZE + CELL_SIZE // 2
            y = row * CELL_SIZE + CELL_SIZE // 2
            painter.drawEllipse(QPoint(x, y), STONE_SIZE, STONE_SIZE)

        # shows forbidden moves
        if self.current_player == 1:
            forbidden = self.get_forbidden_moves()
            self.forbidden_pixmap = QPixmap(os.path.join(os.path.dirname(__file__), "assets", "forbidden.png"))
            img_scaled = self.forbidden_pixmap.scaled(STONE_SIZE * 2, STONE_SIZE * 2, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            for r, c in forbidden:
                x = c * CELL_SIZE + CELL_SIZE // 2 - img_scaled.width() // 2
                y = r * CELL_SIZE + CELL_SIZE // 2 - img_scaled.height() // 2
                painter.drawPixmap(x, y, img_scaled)


    # placing stones
    def mousePressEvent(self, event):
        if self.is_ai_turn:
            return

        x, y = event.position().x(), event.position().y()
        col = int(x // CELL_SIZE)
        row = int(y // CELL_SIZE)

        if self.is_valid_move(row, col):
            self.selected_move = (row, col)
            self.update()


    # creates new board when a game ends
    def clearBoard(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]  
        self.current_player = 1
        self.last_move = None  
        self.selected_move = None
        self.is_ai_turn = False
        self.update()


    def confirm_move(self):
        if not self.selected_move:
            return

        row, col = self.selected_move
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.last_move = (row, col)
            self.selected_move = None
            self.update()

            if check_if_win(self.board, row, col, self.current_player):
                winner_color = "Black" if self.current_player == 1 else "White"
                self.game_over_signal.emit(winner_color)
                self.is_ai_turn = False
                return

            self.current_player = -1
            self.is_ai_turn = True
            QTimer.singleShot(100, self.run_ai_move)

    
    def get_forbidden_moves(self):
        """Returns a list of forbidden (row, col) positions for Black."""
        if self.current_player != 1:
            return []
        forbidden = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == 0 and (
                    is_double_three(self.board, r, c, 1) or
                    is_double_four(self.board, r, c, 1) or
                    is_overline(self.board, r, c)
                ):
                    forbidden.append((r, c))
        return forbidden


    # pop up screen when someone wins / redirects to main page when clicking "ok"
    def show_win_popup(self, winner_color):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Game Over")
        msg_box.setText(f"{winner_color} color wins!")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        button = msg_box.exec()

        if button == QMessageBox.StandardButton.Ok:
            if self.parent_widget:
                self.parent_widget.stacked_widget.setCurrentWidget(self.parent_widget.main_page)
                self.clearBoard()

    def mcts_ai_move(self):
        # 현재 보드 상태와 현재 플레이어 정보를 사용하여 AI가 돌 두기
        current_state = GomokuState(copy.deepcopy(self.board), self.current_player)
        agent = MCTSAgent(iterations=5)    #TODO: iteration 수정
        move = agent.select_move(current_state)
        if move is not None:
            row, col = move
            self.board[row][col] = self.current_player
            self.last_move = (row, col)

            if check_if_win(self.board, row, col, self.current_player) == True:
                winner_color = "Black" if self.current_player == 1 else "White"
                self.game_over(winner_color)
                return

            self.current_player = -self.current_player
            self.update()
            # 만약 AI 턴 후에도 AI가 계속 두어야 한다면 (예: 두 명의 AI 대결), 여기서 다시 호출
            # if self.current_player == -1:
            #     QTimer.singleShot(500, self.ai_move)

    def minimax_ai_move(self):
        """Execute AI move"""
        # if self.is_ai_turn and self.is_ai_enabled:
        # QApplication.processEvents()  # Allow GUI to update
        move = self.ai.get_best_move(self.board, self.current_player)
        print(move)
        if move:
            row, col = move
            if self.is_valid_move(row, col):
                self.board[row][col] = self.current_player
                self.last_move = (row, col)
                
                if check_if_win(self.board, row, col, self.current_player):
                    winner_color = "Black" if self.current_player == 1 else "White"
                    self.game_over_signal.emit(winner_color)
                    self.is_ai_turn = False
                    return
                else:
                    self.current_player = -1 # if self.current_player == 1 else 1
                    self.is_ai_turn = False

    def nn_ai_move(self):
        """Execute one network‐guided move."""
        # assemble input tensor: shape (1, C, 15, 15)
        board_tensor = board_to_tensor(self.board, self.current_player)
        board_tensor = board_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.net(board_tensor)
            # assume policy_logits shape is [1, 225]
            policy = torch.softmax(policy_logits, dim=1).view(-1)

        # mask out illegal positions
        legal = []
        for idx in range(BOARD_SIZE * BOARD_SIZE):
            r, c = idx // BOARD_SIZE, idx % BOARD_SIZE
            if self.is_valid_move(r, c):
                legal.append(idx)

        # pick the legal idx with highest probability
        best_idx = max(legal, key=lambda i: policy[i].item())
        row, col = best_idx // BOARD_SIZE, best_idx % BOARD_SIZE

        # play it
        self.board[row][col] = self.current_player
        self.update()

        if check_if_win(self.board, row, col, self.current_player):
            winner_color = "Black" if self.current_player == 1 else "White"
            self.game_over_signal.emit(winner_color)
            self.is_ai_turn = False
            return

        # hand control back to the human
        self.current_player = 1
        self.is_ai_turn = False

            
    def run_ai_move(self):
        if not self.is_ai_turn:
            return

        start_time = time.time()
        if self.selected_ai_model == "minimax":
            # print("minimax")
            self.minimax_ai_move()
        elif self.selected_ai_model == "mcts":
            # print("mcts")
            self.mcts_ai_move()
        elif self.selected_ai_model == "dl":
            # print("dl")
            self.nn_ai_move()
        else:
            print("⚠️ Error occured in selecting model!")
        end_time = time.time()
        print(f"AI move execution time: {end_time - start_time:.6f} seconds")

        self.current_player = 1
        self.is_ai_turn = False
        self.update()


    def is_valid_move(self, row, col):
        """Check if move is valid according to game rules"""
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        if self.board[row][col] != 0:
            return False
            
        # Check renju rules for black
        if self.current_player == 1:
            if (is_double_three(self.board, row, col, self.current_player) or 
                is_double_four(self.board, row, col, self.current_player) or 
                is_overline(self.board, row, col)):
                return False
        return True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GomokuBoard()
    window.setWindowTitle("AlphO")
    window.show()
    sys.exit(app.exec())
import sys, os
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

from AlphaO.mcts_agent import GomokuState, MCTSAgent
from renju_rule import *
import time
# from ai_training.minimax import Minimax
from renju_rule import check_if_win

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
        self.parent_widget = parent

        # self.ai_model = "minimax_model"
        # self.ai = Minimax(depth=2)
        self.is_ai_turn = False

        # TODO : make the size to be flexible : if window becomes smaller, the board scales down too
        self.setFixedSize(BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE)

        #################################################################
        # TODO ??? self.setFixedSize(self.sizeHint())
        # TODO : using image background, need to work on making exact placements
        # self.board_image = QPixmap(os.path.join(os.path.dirname(__file__), "board_img.png"))
        self.background_img = QPixmap(os.path.join(os.path.dirname(__file__), "white_background.png"))
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
                if self.board[row][col] != 0:  # ensures blank space
                    if self.board[row][col] == 1:
                        painter.setBrush(QBrush(Qt.GlobalColor.black))
                    else:
                        painter.setBrush(QBrush(Qt.GlobalColor.white))
                    x = col * CELL_SIZE + CELL_SIZE // 2
                    y = row * CELL_SIZE + CELL_SIZE // 2
                    painter.drawEllipse(QPoint(x, y), STONE_SIZE, STONE_SIZE)

    # placing stones
    def mousePressEvent(self, event):
        x, y = event.position().x(), event.position().y()
        col = int(x // CELL_SIZE)  # x-value
        row = int(y // CELL_SIZE)  # y-value

        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.update()

            # check for win
            if check_if_win(self.board, row, col, self.current_player):
                winner_color = "Black" if self.current_player == 1 else "White"
                self.game_over_signal.emit(winner_color)
                self.is_ai_turn = False
                return
            # else:
            # self.current_player = -self.current_player
            self.current_player = -1
            self.is_ai_turn = True

            QTimer.singleShot(100, self.run_ai_move)

        # ai_turn to place stone (currently white only)
        # if self.is_ai_turn:
        #     start_time = time.time()
        #     self.ai_move()
        #     end_time = time.time()
        #     execution_time = end_time - start_time
        #     print(f"AI move execution time: {execution_time:.6f} seconds")
        #
        #     self.current_player = 1
        #     self.is_ai_turn = False

    # creates new board when a game ends
    def clearBoard(self):
        self.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = 1
        self.update()

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

    def ai_move(self):
        # 현재 보드 상태와 현재 플레이어 정보를 사용하여 AI가 돌 두기
        current_state = GomokuState(copy.deepcopy(self.board), self.current_player)
        agent = MCTSAgent(iterations=50)    #TODO: iteration 수정
        # agent = MCTSAgent()  # TODO: time 수정
        move = agent.select_move(current_state)
        if move is not None:
            r, c = move
            self.board[r][c] = self.current_player

            if check_if_win(self.board, r, c, self.current_player) == True:
                winner_color = "Black" if self.current_player == 1 else "White"
                self.game_over(winner_color)
                return

            self.current_player = -self.current_player
            self.update()
            # 만약 AI 턴 후에도 AI가 계속 두어야 한다면 (예: 두 명의 AI 대결), 여기서 다시 호출
            # if self.current_player == -1:
            #     QTimer.singleShot(500, self.ai_move)

    def run_ai_move(self):
        """Execute AI move separately after UI updates."""
        if not self.is_ai_turn:
            return

        start_time = time.time()
        self.ai_move()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"AI move execution time: {execution_time:.6f} seconds")

        self.is_ai_turn = False

    def is_valid_move(self, row, col):
        """Check if move is valid according to game rules"""
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        if self.board[row][col] != 0:
            return False

        # Check renju rules for black
        if self.current_player == 1:
            if (is_double_three(self.board, row, col, self.current_player) or
                    is_double_four(self.board, row, col) or
                    is_overline(self.board, row, col)):
                return False
        return True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GomokuBoard()
    window.setWindowTitle("AlphO")
    window.show()
    sys.exit(app.exec())
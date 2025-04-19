# gomoku_gui.py
import sys, os
from PyQt6.QtWidgets import QApplication, QDialog, QStackedWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QRadioButton, QMessageBox, QWidget
from PyQt6.QtCore import Qt
from gomoku_board import GomokuBoard
from PyQt6.QtGui import QPixmap


class Main(QDialog):
    def __init__(self):
        super().__init__()
        self.stacked_widget = QStackedWidget(self)
        self.main_page = self.init_main_page()
        self.game_page = self.init_game_page()
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.game_page)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)
        self.setWindowTitle("AlphaO")
        self.resize(800, 600)

    def init_main_page(self):
        main_page_widget = QWidget()
        main_layout = QFormLayout()
        
        # page layouts
        layout1 = QVBoxLayout()
        color_layout = QHBoxLayout()
        start_button_layout = QVBoxLayout()

        # widgets
        label_widget = QLabel("Main menu")
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_widget.setStyleSheet("font-size: 24px; font-weight: bold;")
    
        start_button_widget = QPushButton("click to start")
        start_button_layout.addWidget(start_button_widget)
        start_button_widget.clicked.connect(self.start_game_with_selected_model)

        # choose level difficulty
        self.level_widget = QComboBox()
        self.level_widget.addItems(["Choose model", "Minimax", "MCTS", "DL"])
        # choose white/black stone
        # TODO : 컴퓨터랑 사람이랑 턴제로 하는데, 사람이 색 선택시... function 만들기
        radio_white, radio_black = QRadioButton("White"), QRadioButton("Black")
        color_layout.addWidget(radio_white)
        color_layout.addWidget(radio_black)    
    
        # ordering of layouts/widgets (top -> down)
        layout1.addWidget(label_widget)
        layout1.addWidget(self.level_widget)
        layout1.addWidget(start_button_widget)
        main_layout.addRow(layout1)  
        main_layout.addRow(color_layout)
        main_layout.addRow(start_button_layout)

        # default settings
        # self.main_page.setLayout(main_layout)
        main_page_widget.setLayout(main_layout)
        return main_page_widget


    def init_game_page(self):
        game_page_widget = QWidget()
        main_layout = QFormLayout()
        
        # page layouts
        layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()
        board_layout = QHBoxLayout()
        board_layout.addStretch()

        # widgets
        label_widget = QLabel("ALPHAO!")
        surrender_button_widget = QPushButton("surrender")
        place_button_widget = QPushButton("place")
        self.gomoku_board = GomokuBoard(parent=self)
        place_button_widget.clicked.connect(self.gomoku_board.confirm_move)

        # widget customize & functions
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_widget.setStyleSheet("font-size: 24px; font-weight: bold;")
        surrender_button_widget.clicked.connect(lambda: self.handle_game_over("Black" if self.gomoku_board.current_player == -1 else "White"))
        self.gomoku_board.game_over_signal.connect(self.handle_game_over)
    
        # sub-layout ordering
        layout.addWidget(label_widget)
        buttons_layout.addWidget(surrender_button_widget)
        buttons_layout.addWidget(place_button_widget)
        board_layout.addWidget(self.gomoku_board)
        board_layout.addStretch()

        # main-layout ordering
        main_layout.addRow(layout)  
        main_layout.addRow(board_layout)
        main_layout.addRow(buttons_layout)

        # default settings
        # self.game_page.setLayout(main_layout)
        game_page_widget.setLayout(main_layout)
        return game_page_widget
    

    def start_game_with_selected_model(self):
        selected_text = self.level_widget.currentText().lower()
        if selected_text in ["minimax", "mcts", "dl"]:
            self.gomoku_board.selected_ai_model = selected_text
            self.stacked_widget.setCurrentWidget(self.game_page)
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Please select a valid AI model to start the game.")
            msg_box.setIconPixmap(QPixmap(os.path.join(os.path.dirname(__file__), "assets", "error.png")).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()


    # function for handling game-over
    # create pop-up screen showing who won, then redirects to main page
    def handle_game_over(self, winner_color):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Game Over")
        msg_box.setText(f"{winner_color} wins!")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
        self.stacked_widget.setCurrentWidget(self.main_page)
        self.gomoku_board.clearBoard()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec())
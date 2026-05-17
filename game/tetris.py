"""
Движок игры Тетрис (без графики).
Используется обеими ветками: генетическим алгоритмом и DQN.
"""

import numpy as np
import random

# Все 7 тетромино с вариантами поворота
PIECES = {
    'I': [[[1, 1, 1, 1]],
          [[1], [1], [1], [1]]],
    'O': [[[1, 1], [1, 1]]],
    'T': [[[0, 1, 0], [1, 1, 1]],
          [[1, 0], [1, 1], [1, 0]],
          [[1, 1, 1], [0, 1, 0]],
          [[0, 1], [1, 1], [0, 1]]],
    'S': [[[0, 1, 1], [1, 1, 0]],
          [[1, 0], [1, 1], [0, 1]]],
    'Z': [[[1, 1, 0], [0, 1, 1]],
          [[0, 1], [1, 1], [1, 0]]],
    'J': [[[1, 0, 0], [1, 1, 1]],
          [[1, 1], [1, 0], [1, 0]],
          [[1, 1, 1], [0, 0, 1]],
          [[0, 1], [0, 1], [1, 1]]],
    'L': [[[0, 0, 1], [1, 1, 1]],
          [[1, 0], [1, 0], [1, 1]],
          [[1, 1, 1], [1, 0, 0]],
          [[1, 1], [0, 1], [0, 1]]],
}

PIECE_NAMES = list(PIECES.keys())


class TetrisGame:
    """Логика игры Тетрис. Графика подключается отдельно через pygame."""

    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.score = 0
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        self.game_over = False
        self.current_piece = self._new_piece()
        return self.get_state_vector()

    # ------------------------------------------------------------------ #
    #  Внутренние методы                                                   #
    # ------------------------------------------------------------------ #

    def _new_piece(self):
        name = random.choice(PIECE_NAMES)
        return {'name': name, 'rotations': PIECES[name]}

    def _shape(self, piece, rotation):
        rotations = piece['rotations']
        return np.array(rotations[rotation % len(rotations)])

    def _can_place(self, shape, x, y):
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r][c]:
                    ny, nx = y + r, x + c
                    if ny >= self.height or nx < 0 or nx >= self.width:
                        return False
                    if ny >= 0 and self.board[ny][nx]:
                        return False
        return True

    def _drop_y(self, shape, x):
        """Опускаем фигуру до упора, возвращаем конечную y."""
        y = 0
        while self._can_place(shape, x, y + 1):
            y += 1
        return y

    def _place_on_board(self, board, shape, x, y):
        b = board.copy()
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r][c]:
                    nr, nc = y + r, x + c
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        b[nr][nc] = 1
        return b

    def _count_and_clear(self, board):
        """Считаем заполненные линии и убираем их."""
        keep = [row for row in board if not all(row)]
        cleared = self.height - len(keep)
        empty_rows = [np.zeros(self.width, dtype=int) for _ in range(cleared)]
        return np.array(empty_rows + keep), cleared

    # ------------------------------------------------------------------ #
    #  Признаки для ML-агентов                                            #
    # ------------------------------------------------------------------ #

    def get_features(self, board=None):
        """
        Возвращает 4 ключевых признака состояния поля.
        Именно эти цифры ест генетический алгоритм и DQN.
        """
        b = board if board is not None else self.board

        # Высота каждого столбца
        heights = []
        for col in range(self.width):
            filled = np.where(b[:, col])[0]
            heights.append(self.height - filled[0] if len(filled) > 0 else 0)

        total_height = sum(heights)

        # Дыры: пустая клетка под хотя бы одной заполненной
        holes = 0
        for col in range(self.width):
            block_found = False
            for row in range(self.height):
                if b[row][col]:
                    block_found = True
                elif block_found:
                    holes += 1

        # Неровность: сумма разниц высот соседних столбцов
        bumpiness = sum(abs(heights[i] - heights[i + 1])
                        for i in range(len(heights) - 1))

        return {
            'total_height': total_height,
            'holes': holes,
            'bumpiness': bumpiness,
            'lines_cleared': 0,   # заполняется в get_possible_moves
            'heights': heights,
        }

    def get_state_vector(self):
        """Числовой вектор состояния для нейросети DQN."""
        f = self.get_features()
        return np.array([f['total_height'], f['holes'],
                         f['bumpiness'], self.lines_cleared_total], dtype=float)

    # ------------------------------------------------------------------ #
    #  Возможные ходы                                                      #
    # ------------------------------------------------------------------ #

    def get_possible_moves(self):
        """
        Перебираем все (x, rotation) комбинации для текущей фигуры.
        Возвращаем список ходов с признаками результирующего поля.
        Это общий интерфейс для обоих агентов.
        """
        moves = []
        piece = self.current_piece
        num_rotations = len(piece['rotations'])

        for rotation in range(num_rotations):
            shape = self._shape(piece, rotation)
            for x in range(-shape.shape[1] + 1, self.width):
                if not self._can_place(shape, x, 0):
                    continue
                y = self._drop_y(shape, x)
                temp_board = self._place_on_board(self.board, shape, x, y)
                temp_board, lines = self._count_and_clear(temp_board)
                features = self.get_features(temp_board)
                features['lines_cleared'] = lines
                moves.append({
                    'x': x,
                    'rotation': rotation,
                    'features': features,
                })
        return moves

    # ------------------------------------------------------------------ #
    #  Игровой шаг                                                         #
    # ------------------------------------------------------------------ #

    def step(self, x, rotation):
        """
        Ставим фигуру на позицию x с поворотом rotation.
        Возвращает (reward, done).
        """
        if self.game_over:
            return 0, True

        piece = self.current_piece
        shape = self._shape(piece, rotation)

        if not self._can_place(shape, x, 0):
            self.game_over = True
            return -1, True

        y = self._drop_y(shape, x)
        self.board = self._place_on_board(self.board, shape, x, y)
        self.board, lines = self._count_and_clear(self.board)

        self.lines_cleared_total += lines
        self.pieces_placed += 1

        # Награда: за выживание + квадратичный бонус за линии
        reward = 1 + (lines ** 2) * self.width
        self.score += reward

        self.current_piece = self._new_piece()
        next_shape = self._shape(self.current_piece, 0)

        # Проверка проигрыша
        if not self._can_place(next_shape, self.width // 2 - 1, 0):
            self.game_over = True
            return reward, True

        return reward, False

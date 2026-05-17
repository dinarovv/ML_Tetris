"""
Визуализация обученного агента через pygame.

Использование:
    python play.py --mode genetic    # Генетический алгоритм
    python play.py --mode dqn        # DQN
    python play.py --mode human      # Ручная игра

Управление в ручном режиме:
    ← / →     — двигать фигуру влево/вправо
    ↑         — повернуть фигуру
    ↓ / Space — поставить фигуру на выбранную позицию
    R         — начать заново
    Esc       — выход
"""

import sys
import os
import json
import argparse
import pygame
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game.tetris import TetrisGame

# ------------------------------------------------------------------ #
#  Настройки отображения                                               #
# ------------------------------------------------------------------ #

CELL = 30
COLS, ROWS = 10, 20
PANEL_W = 220

WIDTH = COLS * CELL + PANEL_W
HEIGHT = ROWS * CELL

FPS_AI = 5          # шагов в секунду для агента (медленнее чтобы было видно)
FPS_HUMAN = 60      # для отзывчивого управления

BG = (18, 18, 28)
GRID = (35, 35, 50)
PANEL = (28, 28, 42)
WHITE = (220, 220, 230)
GRAY = (100, 100, 120)
PREVIEW = (255, 220, 100)   # жёлтый для превью текущей фигуры
GHOST = (80, 80, 100)       # серый для "тени" — куда фигура упадёт


# ------------------------------------------------------------------ #
#  Отрисовка                                                           #
# ------------------------------------------------------------------ #

def draw_cell(screen, color, col, row, border_color=None):
    rect = pygame.Rect(col * CELL, row * CELL, CELL, CELL)
    pygame.draw.rect(screen, color, rect)
    if border_color:
        pygame.draw.rect(screen, border_color, rect, 2)


def draw_board(screen, game):
    # Сетка
    for r in range(ROWS):
        for c in range(COLS):
            if game.board[r][c]:
                draw_cell(screen, WHITE, c, r, BG)
            else:
                rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
                pygame.draw.rect(screen, GRID, rect, 1)


def draw_preview(screen, game, x, rotation, color):
    """
    Рисует превью текущей фигуры на позиции (x, rotation) — туда, куда она упадёт.
    """
    piece = game.current_piece
    shape = game._shape(piece, rotation)

    if not game._can_place(shape, x, 0):
        return

    y = game._drop_y(shape, x)

    for r in range(shape.shape[0]):
        for c in range(shape.shape[1]):
            if shape[r][c]:
                draw_cell(screen, color, x + c, y + r, BG)


def draw_panel(screen, game, font_small, font_big, mode, score_history,
               help_lines=None):
    px = COLS * CELL
    pygame.draw.rect(screen, PANEL, (px, 0, PANEL_W, HEIGHT))

    y = 20
    screen.blit(font_small.render(f"Режим: {mode}", True, GRAY), (px + 12, y))

    y += 30
    screen.blit(font_small.render("Счёт", True, GRAY), (px + 12, y))
    y += 18
    screen.blit(font_big.render(f"{int(game.score)}", True, WHITE), (px + 12, y))

    y += 50
    screen.blit(font_small.render(f"Линии: {game.lines_cleared_total}", True, GRAY), (px + 12, y))
    y += 22
    screen.blit(font_small.render(f"Фигуры: {game.pieces_placed}", True, GRAY), (px + 12, y))

    # Текущая фигура
    if game.current_piece:
        y += 30
        screen.blit(font_small.render(f"Фигура: {game.current_piece['name']}", True, GRAY), (px + 12, y))

    # Подсказки управления
    if help_lines:
        y += 35
        for line in help_lines:
            screen.blit(font_small.render(line, True, GRAY), (px + 12, y))
            y += 18

    # График счёта
    if len(score_history) > 1:
        y = HEIGHT - 110
        screen.blit(font_small.render("История счёта:", True, GRAY), (px + 12, y))
        y += 18
        graph_h = 70
        graph_w = PANEL_W - 24
        max_s = max(score_history) or 1
        pts = []
        for i, s in enumerate(score_history[-50:]):
            gx = px + 12 + int(i / max(1, min(50, len(score_history)) - 1) * graph_w)
            gy = y + graph_h - int(s / max_s * graph_h)
            pts.append((gx, gy))
        if len(pts) > 1:
            pygame.draw.lines(screen, (91, 79, 214), False, pts, 2)


def draw_game_over(screen, font_big, font_small):
    overlay = pygame.Surface((COLS * CELL, HEIGHT))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    text = font_big.render("ИГРА ОКОНЧЕНА", True, WHITE)
    rect = text.get_rect(center=(COLS * CELL // 2, HEIGHT // 2 - 20))
    screen.blit(text, rect)

    hint = font_small.render("R — заново, Esc — выход", True, GRAY)
    rect = hint.get_rect(center=(COLS * CELL // 2, HEIGHT // 2 + 20))
    screen.blit(hint, rect)


# ------------------------------------------------------------------ #
#  Загрузка агентов                                                    #
# ------------------------------------------------------------------ #

def load_genetic_agent():
    from genetic.agent import GeneticAgent
    path = os.path.join(os.path.dirname(__file__), 'genetic', 'best_weights.json')
    if not os.path.exists(path):
        print(f"[!] Файл не найден: {path}")
        print("    Сначала запусти: python genetic/train.py")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    agent = GeneticAgent(data['weights'])
    print(f"Загружен генетический агент: {agent}")
    return agent


def load_dqn_agent():
    from dqn.agent import DQNAgent
    path = os.path.join(os.path.dirname(__file__), 'dqn', 'dqn_model.pt')
    if not os.path.exists(path):
        print(f"[!] Файл не найден: {path}")
        print("    Сначала запусти: python dqn/train.py")
        sys.exit(1)
    agent = DQNAgent(epsilon=0.0)
    agent.load(path)
    print("Загружен DQN-агент")
    return agent


# ------------------------------------------------------------------ #
#  Вспомогательное для ручного режима                                  #
# ------------------------------------------------------------------ #

def find_valid_x(game, rotation, preferred_x):
    """
    Находим валидную позицию x для текущей фигуры с поворотом rotation.
    Сначала пытаемся preferred_x, если нельзя — ищем ближайшую валидную.
    """
    shape = game._shape(game.current_piece, rotation)
    # Пытаемся preferred_x
    if game._can_place(shape, preferred_x, 0):
        return preferred_x
    # Ищем ближайший валидный x
    for delta in range(1, COLS + 1):
        for x in [preferred_x - delta, preferred_x + delta]:
            if game._can_place(shape, x, 0):
                return x
    return preferred_x  # fallback


def clamp_x(game, x, rotation):
    """Прижимаем x к допустимому диапазону для данного поворота."""
    shape = game._shape(game.current_piece, rotation)
    # Двигаем влево пока не валидно
    while x > -shape.shape[1] + 1 and not game._can_place(shape, x, 0):
        x -= 1
    while x < COLS and not game._can_place(shape, x, 0):
        x += 1
    return x


# ------------------------------------------------------------------ #
#  Главный цикл                                                        #
# ------------------------------------------------------------------ #

def run(mode='genetic'):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Тетрис — {mode}')
    clock = pygame.time.Clock()

    # Шрифты с поддержкой кириллицы
    font_small = pygame.font.SysFont('arial,segoeui,dejavusans', 14)
    font_big = pygame.font.SysFont('arial,segoeui,dejavusans', 28, bold=True)

    if mode == 'genetic':
        agent = load_genetic_agent()
        fps = FPS_AI
        help_lines = ["", "Управление:", "R — рестарт", "Esc — выход"]
    elif mode == 'dqn':
        agent = load_dqn_agent()
        fps = FPS_AI
        help_lines = ["", "Управление:", "R — рестарт", "Esc — выход"]
    else:
        agent = None
        fps = FPS_HUMAN
        help_lines = ["", "Управление:",
                      "← → — двигать",
                      "↑ — поворот",
                      "↓/Space — поставить",
                      "R — рестарт",
                      "Esc — выход"]

    score_history = []
    game = TetrisGame()

    # Состояние ручного режима
    cursor_x = COLS // 2 - 1
    cursor_rotation = 0
    if mode == 'human':
        cursor_x = clamp_x(game, cursor_x, cursor_rotation)

    running = True

    while running:
        # ---------- Обработка событий ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_r:
                    score_history.append(game.score)
                    game = TetrisGame()
                    cursor_x = COLS // 2 - 1
                    cursor_rotation = 0
                    if mode == 'human':
                        cursor_x = clamp_x(game, cursor_x, cursor_rotation)

                # Обработка управления — только в ручном режиме и если игра не окончена
                if mode == 'human' and not game.game_over:
                    if event.key == pygame.K_LEFT:
                        new_x = cursor_x - 1
                        shape = game._shape(game.current_piece, cursor_rotation)
                        if game._can_place(shape, new_x, 0):
                            cursor_x = new_x

                    elif event.key == pygame.K_RIGHT:
                        new_x = cursor_x + 1
                        shape = game._shape(game.current_piece, cursor_rotation)
                        if game._can_place(shape, new_x, 0):
                            cursor_x = new_x

                    elif event.key == pygame.K_UP:
                        num_rotations = len(game.current_piece['rotations'])
                        new_rotation = (cursor_rotation + 1) % num_rotations
                        cursor_x = clamp_x(game, cursor_x, new_rotation)
                        cursor_rotation = new_rotation

                    elif event.key in (pygame.K_DOWN, pygame.K_SPACE):
                        # Фиксируем ход
                        game.step(cursor_x, cursor_rotation)
                        # Сбрасываем курсор для новой фигуры
                        if not game.game_over:
                            cursor_rotation = 0
                            cursor_x = clamp_x(game, COLS // 2 - 1, 0)

        # ---------- Логика агента ----------
        if mode in ('genetic', 'dqn') and not game.game_over:
            moves = game.get_possible_moves()
            move = agent.choose_move(moves)
            if move:
                game.step(move['x'], move['rotation'])

        # ---------- Автоматический рестарт после game over для агентов ----------
        if mode in ('genetic', 'dqn') and game.game_over:
            score_history.append(game.score)
            pygame.time.wait(1500)
            game = TetrisGame()

        # ---------- Отрисовка ----------
        screen.fill(BG)
        draw_board(screen, game)

        # В ручном режиме рисуем превью фигуры
        if mode == 'human' and not game.game_over:
            draw_preview(screen, game, cursor_x, cursor_rotation, PREVIEW)

        draw_panel(screen, game, font_small, font_big, mode, score_history,
                   help_lines)

        if game.game_over:
            draw_game_over(screen, font_big, font_small)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['genetic', 'dqn', 'human'],
                        default='genetic')
    args = parser.parse_args()
    run(args.mode)
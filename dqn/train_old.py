"""
Запуск обучения — ветка DQN.

Использование:
    python dqn/train.py

Требует PyTorch: pip install torch
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.tetris import TetrisGame
from dqn.agent import DQNAgent

# ------------------------------------------------------------------ #
#  Параметры обучения                                                  #
# ------------------------------------------------------------------ #

EPISODES = 2000          # было 500 → главное изменение, даём DQN доучиться
MAX_STEPS = 1500         # было 500 → снимаем потолок (как для ГА в Эксп. A3)
PRINT_EVERY = 50         # было 25 → реже выводим, чтобы не засорять консоль

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dqn_model.pt')
PLOT_PATH = os.path.join(os.path.dirname(__file__), 'training_plot.png')


def plot_training(scores, losses, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # График счёта
    ax1.plot(scores, color='#5B4FD6', alpha=0.4, linewidth=0.8)
    # Скользящее среднее
    window = min(20, len(scores))
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(scores)), moving_avg,
                 color='#5B4FD6', linewidth=2, label=f'Среднее ({window} эп.)')
    ax1.set_xlabel('Эпизод')
    ax1.set_ylabel('Счёт')
    ax1.set_title('DQN — рост счёта по эпизодам')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График функции потерь
    losses_clean = [l for l in losses if l is not None]
    if losses_clean:
        ax2.plot(losses_clean, color='#D85A30', alpha=0.5, linewidth=0.8)
        ax2.set_xlabel('Шаг обучения')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('DQN — функция потерь')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"  График сохранён: {path}")
    plt.show()


def main():
    agent = DQNAgent(
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10_000,
        batch_size=64,
    )

    scores = []
    all_losses = []
    best_score = 0

    print("=" * 60)
    print(f"DQN: {EPISODES} эпизодов, max {MAX_STEPS} шагов")
    print("=" * 60)

    for episode in range(1, EPISODES + 1):
        game = TetrisGame()
        episode_loss = []

        for step in range(MAX_STEPS):
            moves = game.get_possible_moves()
            if not moves:
                break

            # Сохраняем признаки текущего лучшего хода ДО шага
            current_move = agent.choose_move(moves)
            if current_move is None:
                break

            state_feat = current_move['features']
            reward, done = game.step(current_move['x'], current_move['rotation'])

            # Признаки следующего состояния
            next_moves = game.get_possible_moves()
            if next_moves:
                next_feat = agent.choose_move(next_moves)['features']
            else:
                next_feat = {'holes': 0, 'total_height': 0,
                             'bumpiness': 0, 'lines_cleared': 0}

            agent.remember(state_feat, reward, next_feat, float(done))

            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
                all_losses.append(loss)

            if done:
                break

        agent.decay_epsilon()
        scores.append(game.score)

        if game.score > best_score:
            best_score = game.score
            agent.save(MODEL_PATH)

        if episode % PRINT_EVERY == 0:
            avg = np.mean(scores[-PRINT_EVERY:])
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            print(f"  Эпизод {episode:4d}/{EPISODES} | "
                  f"счёт={game.score:7.0f} | "
                  f"лучший={best_score:7.0f} | "
                  f"avg={avg:7.0f} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"loss={avg_loss:.4f}")

    print(f"\n✓ Обучение завершено! Лучший счёт: {best_score:.0f}")
    plot_training(scores, all_losses, PLOT_PATH)


if __name__ == '__main__':
    main()

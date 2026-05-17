"""
Запуск обучения — ветка DQN.

ВЕРСИЯ ПОСЛЕ ПАКЕТОВ 1, 2, 3:
  - Корректный target и разделённый exploration (Пакет 1)
  - Multi-seed запуски + CSV логирование (Пакет 2)
  - Huber loss и gradient clipping (Пакет 3) — в agent.py

Использование:
    python dqn/train.py                          # один запуск с seed=42
    python dqn/train.py --seeds 42,123,777       # три запуска подряд
    python dqn/train.py --seeds 42 --episodes 1000  # короткий тест

Артефакты:
    dqn/logs/dqn_seed{N}.csv      — лог по эпизодам
    dqn/dqn_model_seed{N}.pt      — модель для каждого seed
    dqn/training_plot_seed{N}.png — график для каждого seed
"""

import sys
import os
import csv
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.tetris import TetrisGame
from dqn.agent import DQNAgent

# ------------------------------------------------------------------ #
#  Параметры обучения по умолчанию (соответствуют конфигурации B3)    #
# ------------------------------------------------------------------ #

DEFAULT_SEED = 42

EPISODES = 2000
MAX_STEPS = 1500
PRINT_EVERY = 50

LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998
MEMORY_SIZE = 30_000
BATCH_SIZE = 128
TARGET_UPDATE = 500

DQN_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(DQN_DIR, 'logs')


def set_seed(seed):
    """Фиксируем все источники случайности."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def plot_training(scores, losses, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(scores, color='#5B4FD6', alpha=0.4, linewidth=0.8)
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

    losses_clean = [l for l in losses if l is not None]
    if losses_clean:
        ax2.plot(losses_clean, color='#D85A30', alpha=0.5, linewidth=0.8)
        ax2.set_xlabel('Шаг обучения')
        ax2.set_ylabel('Loss (Huber)')
        ax2.set_title('DQN — функция потерь')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"  График сохранён: {path}")
    plt.close()


def run_single(seed, episodes, max_steps):
    """Один полный прогон обучения с заданным seed."""
    set_seed(seed)

    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(LOGS_DIR, f'dqn_seed{seed}.csv')
    model_path = os.path.join(DQN_DIR, f'dqn_model_seed{seed}.pt')
    plot_path = os.path.join(DQN_DIR, f'training_plot_seed{seed}.png')

    agent = DQNAgent(
        lr=LR, gamma=GAMMA,
        epsilon=EPSILON_START, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY,
        memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE, target_update=TARGET_UPDATE,
        use_huber=True, grad_clip=1.0,
    )

    scores = []
    all_losses = []
    best_score = 0

    print("=" * 60)
    print(f"DQN | seed={seed} | episodes={episodes} | max_steps={max_steps}")
    print("=" * 60)

    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(['episode', 'score', 'best_score',
                     'avg_last_50', 'epsilon', 'avg_loss'])

    for episode in range(1, episodes + 1):
        game = TetrisGame()
        episode_loss = []

        for step in range(max_steps):
            moves = game.get_possible_moves()
            if not moves:
                break

            current_move = agent.choose_move(moves)
            if current_move is None:
                break

            state_feat = current_move['features']
            reward, done = game.step(current_move['x'], current_move['rotation'])

            if done:
                next_feats_list = []
            else:
                next_moves = game.get_possible_moves()
                next_feats_list = [m['features'] for m in next_moves]

            agent.remember(state_feat, reward, next_feats_list, float(done))

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
            agent.save(model_path)

        avg_50 = np.mean(scores[-50:])
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        writer.writerow([episode, int(game.score), int(best_score),
                         f'{avg_50:.1f}', f'{agent.epsilon:.4f}',
                         f'{avg_loss:.4f}'])

        if episode % PRINT_EVERY == 0:
            print(f"  Эпизод {episode:4d}/{episodes} | "
                  f"счёт={game.score:7.0f} | лучший={best_score:7.0f} | "
                  f"avg={avg_50:7.0f} | ε={agent.epsilon:.3f} | "
                  f"loss={avg_loss:.4f}")

    csv_file.close()
    print(f"\n✓ seed={seed} завершён. Лучший счёт: {best_score:.0f}")
    print(f"  Лог CSV: {csv_path}")
    plot_training(scores, all_losses, plot_path)

    return {
        'seed': seed,
        'best': best_score,
        'final_avg_50': float(np.mean(scores[-50:])),
        'final_avg_200': float(np.mean(scores[-200:])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default=str(DEFAULT_SEED),
                        help='Один seed (42) или несколько через запятую (42,123,777)')
    parser.add_argument('--episodes', type=int, default=EPISODES)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    results = []

    for seed in seeds:
        result = run_single(seed, args.episodes, args.max_steps)
        results.append(result)
        print()

    # Сводка по всем прогонам
    if len(results) > 1:
        print("=" * 60)
        print("СВОДКА ПО ВСЕМ ПРОГОНАМ")
        print("=" * 60)
        bests = [r['best'] for r in results]
        avgs = [r['final_avg_50'] for r in results]
        print(f"  Лучшие счёта:   {bests}")
        print(f"  Mean ± Std:     {np.mean(bests):.0f} ± {np.std(bests):.0f}")
        print(f"  Финальное avg(50): {np.mean(avgs):.0f} ± {np.std(avgs):.0f}")
        print()
        print("  Для детальной агрегации запусти: python analyze.py")


if __name__ == '__main__':
    main()

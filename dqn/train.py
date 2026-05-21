"""
Запуск обучения — ветка DQN.

ВЕРСИЯ ПОСЛЕ ПАКЕТА 4 (полная):
  4.1: правильная формула target (max_a (r_a + γ·Q_target(s_a)))
  4.2: eval-фаза с epsilon=0
  4.3: расширенное логирование (steps, lines_cleared, eval_score)
  4.4: единый конфиг через --config <path>
  4.5: --output-dir для интеграции в run_experiments.py

Использование:
    # Один прогон с дефолтами:
    python dqn/train.py

    # Несколько seed:
    python dqn/train.py --seeds 42,123,777

    # С конфигом:
    python dqn/train.py --config configs/dqn.json --seeds 42,123,777

    # С указанием выходной папки (для run_experiments.py):
    python dqn/train.py --config configs/dqn.json --seeds 42 \\
        --output-dir runs/exp1/dqn_seed42
"""

import sys
import os
import csv
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.tetris import TetrisGame
from dqn.agent import DQNAgent, _compute_reward

# ------------------------------------------------------------------ #
#  Дефолты (используются, если не передан --config)                    #
# ------------------------------------------------------------------ #

DEFAULTS = {
    'seed': 42,
    'episodes': 2000,
    'max_steps': 1500,
    'eval_every': 25,
    'eval_max_steps': 1500,
    'print_every': 50,
    'lr': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.998,
    'memory_size': 30_000,
    'batch_size': 128,
    'target_update': 500,
    'use_huber': True,
    'grad_clip': 1.0,
}

DQN_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ #
#  Конфиг                                                              #
# ------------------------------------------------------------------ #

def load_config(path):
    """Загружает JSON-конфиг. None или пустая строка → пустой словарь."""
    if not path:
        return {}
    with open(path, encoding='utf-8') as f:
        config = json.load(f)
    # Убираем служебные ключи (начинающиеся с _)
    return {k: v for k, v in config.items() if not k.startswith('_')}


def merge_params(defaults, config):
    """Конфиг переопределяет дефолты."""
    result = dict(defaults)
    result.update(config)
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------ #
#  Эпизоды                                                             #
# ------------------------------------------------------------------ #

def play_training_episode(agent, max_steps):
    """Тренировочный эпизод с обучением (afterstate-формулировка)."""
    game = TetrisGame()
    prev_state_feat = None
    losses = []
    steps = 0

    while steps < max_steps:
        moves = game.get_possible_moves()
        if not moves:
            break

        next_options = [
            (_compute_reward(m['features']), m['features']) for m in moves
        ]

        if prev_state_feat is not None:
            agent.remember(prev_state_feat, next_options)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        chosen = agent.choose_move(moves)
        if chosen is None:
            break

        _, done = game.step(chosen['x'], chosen['rotation'])
        prev_state_feat = chosen['features']
        steps += 1

        if done:
            break

    if prev_state_feat is not None:
        agent.remember(prev_state_feat, [])  # terminal
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)

    return {
        'score': game.score,
        'steps': steps,
        'lines_cleared': game.lines_cleared_total,
        'losses': losses,
    }


def play_eval_episode(agent, max_steps):
    """Eval-эпизод без exploration и без обучения (ε=0)."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    game = TetrisGame()
    steps = 0
    while steps < max_steps:
        moves = game.get_possible_moves()
        if not moves:
            break
        chosen = agent.choose_move(moves)
        if chosen is None:
            break
        _, done = game.step(chosen['x'], chosen['rotation'])
        steps += 1
        if done:
            break

    agent.epsilon = saved_eps
    return {
        'score': game.score,
        'steps': steps,
        'lines_cleared': game.lines_cleared_total,
    }


# ------------------------------------------------------------------ #
#  Графики                                                             #
# ------------------------------------------------------------------ #

def plot_training(train_scores, eval_scores, eval_episodes, losses, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(train_scores, color='#5B4FD6', alpha=0.3, linewidth=0.7,
            label='Train (per episode)')
    window = min(20, len(train_scores))
    if len(train_scores) >= window:
        ma = np.convolve(train_scores, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(train_scores)), ma,
                color='#5B4FD6', linewidth=2, label=f'Train avg-{window}')
    if eval_scores:
        ax.plot(eval_episodes, eval_scores, 'o-', color='#1D9E75',
                linewidth=1.5, markersize=4, label='Eval (ε=0)')
    ax.set_xlabel('Эпизод')
    ax.set_ylabel('Счёт')
    ax.set_title('DQN — обучение и eval')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    clean = [l for l in losses if l is not None]
    if clean:
        ax.plot(clean, color='#D85A30', alpha=0.5, linewidth=0.7)
        ax.set_xlabel('Шаг обучения')
        ax.set_ylabel('Loss (Huber)')
        ax.set_title('DQN — функция потерь')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------------------------------------------------ #
#  Один прогон                                                         #
# ------------------------------------------------------------------ #

def run_single(seed, params, output_dir=None):
    """
    Один полный прогон обучения с заданным seed.

    Параметры:
      seed       — числовое значение seed
      params     — словарь параметров (после merge_params)
      output_dir — куда сохранять артефакты. По умолчанию dqn/

    Возвращает словарь с метриками + путями к артефактам.
    """
    set_seed(seed)
    output_dir = output_dir or DQN_DIR

    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    csv_path   = os.path.join(logs_dir,    f'dqn_seed{seed}.csv')
    model_path = os.path.join(output_dir,  f'dqn_model_seed{seed}.pt')
    plot_path  = os.path.join(output_dir,  f'training_plot_seed{seed}.png')

    agent = DQNAgent(
        lr=params['lr'], gamma=params['gamma'],
        epsilon=params['epsilon_start'],
        epsilon_min=params['epsilon_min'],
        epsilon_decay=params['epsilon_decay'],
        memory_size=params['memory_size'],
        batch_size=params['batch_size'],
        target_update=params['target_update'],
        use_huber=params['use_huber'],
        grad_clip=params['grad_clip'],
    )

    train_scores  = []
    eval_scores   = []
    eval_episodes = []
    all_losses    = []
    best_eval     = 0
    best_train    = 0

    print("=" * 64)
    print(f"DQN | seed={seed} | episodes={params['episodes']} | "
          f"max_steps={params['max_steps']}")
    print(f"     eval каждые {params['eval_every']} эпизодов (ε=0)")
    print(f"     output: {output_dir}")
    print("=" * 64)

    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow([
        'episode', 'train_score', 'eval_score', 'best_eval',
        'steps', 'lines_cleared', 'epsilon', 'avg_loss',
    ])

    for episode in range(1, params['episodes'] + 1):
        train_res = play_training_episode(agent, params['max_steps'])
        train_scores.append(train_res['score'])
        all_losses.extend(train_res['losses'])
        agent.decay_epsilon()

        best_train = max(best_train, train_res['score'])

        eval_score = None
        if episode % params['eval_every'] == 0:
            eval_res = play_eval_episode(agent, params['eval_max_steps'])
            eval_score = eval_res['score']
            eval_scores.append(eval_score)
            eval_episodes.append(episode)
            if eval_score > best_eval:
                best_eval = eval_score
                agent.save(model_path)

        avg_loss = (np.mean(train_res['losses'])
                    if train_res['losses'] else 0.0)
        writer.writerow([
            episode, int(train_res['score']),
            int(eval_score) if eval_score is not None else '',
            int(best_eval),
            train_res['steps'], train_res['lines_cleared'],
            f"{agent.epsilon:.4f}", f"{avg_loss:.4f}",
        ])

        if episode % params['print_every'] == 0:
            recent_train = np.mean(train_scores[-params['print_every']:])
            recent_eval = (np.mean(eval_scores[-3:]) if eval_scores else 0)
            print(f"  Эп {episode:4d}/{params['episodes']} | "
                  f"train={train_res['score']:6.0f} "
                  f"(avg={recent_train:6.0f}) | "
                  f"eval≈{recent_eval:6.0f} | "
                  f"best_eval={best_eval:6.0f} | "
                  f"ε={agent.epsilon:.3f} | loss={avg_loss:.3f}")

    csv_file.close()
    plot_training(train_scores, eval_scores, eval_episodes,
                  all_losses, plot_path)

    print(f"\n✓ seed={seed} завершён.")
    print(f"  Best train: {best_train:.0f}")
    print(f"  Best eval:  {best_eval:.0f}")

    return {
        'method': 'dqn',
        'seed': seed,
        'best_train': float(best_train),
        'best_eval':  float(best_eval),
        'final_train_avg_50': float(np.mean(train_scores[-50:])),
        'final_eval': float(eval_scores[-1]) if eval_scores else 0.0,
        'final_epsilon': float(agent.epsilon),
        'artifacts': {
            'csv':   csv_path,
            'model': model_path,
            'plot':  plot_path,
        },
    }


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description='Обучение DQN')
    parser.add_argument('--config', type=str, default=None,
                        help='Путь к JSON-конфигу (необязательно)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Один seed или несколько через запятую '
                             '(переопределяет конфиг)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Переопределить количество эпизодов')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Переопределить максимум шагов')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Папка для артефактов (по умолчанию dqn/)')
    args = parser.parse_args()

    # Сбор параметров: дефолты → конфиг → CLI-аргументы
    config = load_config(args.config)
    params = merge_params(DEFAULTS, config)

    if args.episodes is not None:
        params['episodes'] = args.episodes
    if args.max_steps is not None:
        params['max_steps'] = args.max_steps

    # Seeds: CLI → конфиг → дефолт
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seed_from_config = params.get('seed', DEFAULTS['seed'])
        seeds = [seed_from_config]

    if args.config:
        print(f"  Конфиг: {args.config}")

    results = []
    for seed in seeds:
        result = run_single(seed, params, output_dir=args.output_dir)
        results.append(result)
        print()

    if len(results) > 1:
        print("=" * 64)
        print("СВОДКА ПО ВСЕМ ПРОГОНАМ")
        print("=" * 64)
        best_evals = [r['best_eval'] for r in results]
        final_evals = [r['final_eval'] for r in results]
        print(f"  Seeds:        {[r['seed'] for r in results]}")
        print(f"  Best eval:    {[int(v) for v in best_evals]}")
        print(f"  Best eval ± std:   "
              f"{np.mean(best_evals):.0f} ± {np.std(best_evals):.0f}")
        print(f"  Final eval ± std:  "
              f"{np.mean(final_evals):.0f} ± {np.std(final_evals):.0f}")

    return results


if __name__ == '__main__':
    main()

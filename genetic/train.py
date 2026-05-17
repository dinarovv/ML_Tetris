"""
Запуск обучения — ветка Генетический алгоритм.

ВЕРСИЯ ПОСЛЕ ПАКЕТОВ 1, 2, 3:
  - Фиксация seed (Пакет 1)
  - Multi-seed запуски + CSV логирование (Пакет 2)

Использование:
    python genetic/train.py                          # один запуск seed=42
    python genetic/train.py --seeds 42,123,777       # три запуска подряд
    python genetic/train.py --seeds 42 --generations 5  # короткий тест

Артефакты:
    genetic/logs/genetic_seed{N}.csv      — лог по поколениям
    genetic/best_weights_seed{N}.json     — веса лучшего агента для каждого seed
    genetic/training_plot_seed{N}.png     — график для каждого seed
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

from genetic.agent import GeneticAlgorithm

# ------------------------------------------------------------------ #
#  Параметры обучения по умолчанию (соответствуют конфигурации A3)    #
# ------------------------------------------------------------------ #

DEFAULT_SEED = 42

POPULATION_SIZE = 50
GENERATIONS = 10
GAMES_PER_AGENT = 5
MAX_STEPS = 1500

GA_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(GA_DIR, 'logs')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_weights(agent, path):
    data = {'weights': agent.weights.tolist()}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Веса сохранены: {path}")


def plot_training(history, path):
    gens = [h['generation'] for h in history]
    best = [h['best'] for h in history]
    avg = [h['avg'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best, label='Лучший счёт', color='#5B4FD6', linewidth=2)
    ax.plot(gens, avg, label='Средний счёт', color='#1D9E75',
            linewidth=1.5, linestyle='--')
    ax.fill_between(gens, avg, best, alpha=0.1, color='#5B4FD6')
    ax.set_xlabel('Поколение')
    ax.set_ylabel('Счёт')
    ax.set_title('Обучение генетического алгоритма — Тетрис')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"  График сохранён: {path}")
    plt.close()


def write_history_csv(history, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'best', 'avg',
                         'w_holes', 'w_height', 'w_bumpiness', 'w_lines'])
        for h in history:
            w = h['best_weights']
            writer.writerow([
                h['generation'],
                f"{h['best']:.1f}",
                f"{h['avg']:.1f}",
                f"{w[0]:.4f}", f"{w[1]:.4f}",
                f"{w[2]:.4f}", f"{w[3]:.4f}",
            ])


def run_single(seed, population_size, generations, games_per_agent, max_steps):
    """Один полный прогон с заданным seed."""
    set_seed(seed)

    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(LOGS_DIR, f'genetic_seed{seed}.csv')
    weights_path = os.path.join(GA_DIR, f'best_weights_seed{seed}.json')
    plot_path = os.path.join(GA_DIR, f'training_plot_seed{seed}.png')

    print("=" * 60)
    print(f"GA | seed={seed} | pop={population_size} | gens={generations} | "
          f"games={games_per_agent} | max_steps={max_steps}")
    print("=" * 60)

    ga = GeneticAlgorithm(
        population_size=population_size,
        games_per_agent=games_per_agent,
        max_steps=max_steps,
    )
    best_agent = ga.run(generations=generations)

    save_weights(best_agent, weights_path)
    write_history_csv(ga.history, csv_path)
    print(f"  Лог CSV: {csv_path}")
    plot_training(ga.history, plot_path)

    final = ga.history[-1]
    return {
        'seed': seed,
        'best': float(ga.best_score),
        'final_best': final['best'],
        'final_avg': final['avg'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default=str(DEFAULT_SEED),
                        help='Один seed (42) или несколько через запятую (42,123,777)')
    parser.add_argument('--population', type=int, default=POPULATION_SIZE)
    parser.add_argument('--generations', type=int, default=GENERATIONS)
    parser.add_argument('--games', type=int, default=GAMES_PER_AGENT)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    results = []

    for seed in seeds:
        result = run_single(
            seed,
            args.population, args.generations, args.games, args.max_steps,
        )
        results.append(result)
        print()

    # Сводка
    if len(results) > 1:
        print("=" * 60)
        print("СВОДКА ПО ВСЕМ ПРОГОНАМ")
        print("=" * 60)
        bests = [r['best'] for r in results]
        print(f"  Лучшие счёта:   {bests}")
        print(f"  Mean ± Std:     {np.mean(bests):.0f} ± {np.std(bests):.0f}")
        print(f"  Медиана:        {np.median(bests):.0f}")
        print()
        print("  Для детальной агрегации запусти: python analyze.py")


if __name__ == '__main__':
    main()

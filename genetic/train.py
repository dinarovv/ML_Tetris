"""
Запуск обучения — ветка Генетический алгоритм.

ВЕРСИЯ ПОСЛЕ ПАКЕТА 4 (полная):
  4.4: единый конфиг через --config <path>
  4.5: --output-dir для интеграции в run_experiments.py

Использование:
    python genetic/train.py
    python genetic/train.py --seeds 42,123,777
    python genetic/train.py --config configs/genetic.json
    python genetic/train.py --config configs/genetic.json --seeds 42 \\
        --output-dir runs/exp1/genetic_seed42
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
#  Дефолты                                                             #
# ------------------------------------------------------------------ #

DEFAULTS = {
    'seed': 42,
    'population_size': 50,
    'generations': 10,
    'games_per_agent': 5,
    'max_steps': 1500,
    'elite_frac': 0.2,
}

GA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    if not path:
        return {}
    with open(path, encoding='utf-8') as f:
        config = json.load(f)
    return {k: v for k, v in config.items() if not k.startswith('_')}


def merge_params(defaults, config):
    result = dict(defaults)
    result.update(config)
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_weights(agent, path):
    with open(path, 'w') as f:
        json.dump({'weights': agent.weights.tolist()}, f, indent=2)


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


def plot_training(history, path):
    gens = [h['generation'] for h in history]
    best = [h['best'] for h in history]
    avg  = [h['avg']  for h in history]

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
    plt.close()


# ------------------------------------------------------------------ #
#  Один прогон                                                         #
# ------------------------------------------------------------------ #

def run_single(seed, params, output_dir=None):
    """Один прогон GA с заданным seed."""
    set_seed(seed)
    output_dir = output_dir or GA_DIR

    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    csv_path     = os.path.join(logs_dir,   f'genetic_seed{seed}.csv')
    weights_path = os.path.join(output_dir, f'best_weights_seed{seed}.json')
    plot_path    = os.path.join(output_dir, f'training_plot_seed{seed}.png')

    print("=" * 64)
    print(f"GA | seed={seed} | pop={params['population_size']} | "
          f"gens={params['generations']} | games={params['games_per_agent']} | "
          f"max_steps={params['max_steps']}")
    print(f"     output: {output_dir}")
    print("=" * 64)

    ga = GeneticAlgorithm(
        population_size=params['population_size'],
        games_per_agent=params['games_per_agent'],
        max_steps=params['max_steps'],
    )
    best_agent = ga.run(generations=params['generations'])

    save_weights(best_agent, weights_path)
    print(f"  Веса сохранены: {weights_path}")

    write_history_csv(ga.history, csv_path)
    print(f"  CSV: {csv_path}")

    plot_training(ga.history, plot_path)
    print(f"  График: {plot_path}")

    final = ga.history[-1]
    global_best = max(h['best'] for h in ga.history)

    return {
        'method': 'genetic',
        'seed': seed,
        'best_score': float(global_best),
        'final_best': float(final['best']),
        'final_avg':  float(final['avg']),
        'best_weights': best_agent.weights.tolist(),
        'artifacts': {
            'csv':     csv_path,
            'weights': weights_path,
            'plot':    plot_path,
        },
    }


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description='Обучение GA')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--generations', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    params = merge_params(DEFAULTS, config)

    if args.generations is not None:
        params['generations'] = args.generations

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = [params.get('seed', DEFAULTS['seed'])]

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
        bests = [r['best_score'] for r in results]
        print(f"  Seeds:           {[r['seed'] for r in results]}")
        print(f"  Лучшие за прогон:{[int(v) for v in bests]}")
        print(f"  Mean ± std:      {np.mean(bests):.0f} ± {np.std(bests):.0f}")
        print(f"  Медиана:         {np.median(bests):.0f}")

    return results


if __name__ == '__main__':
    main()

"""
Агрегация и анализ результатов обучения по нескольким seed.

Читает CSV-логи из genetic/logs/ и dqn/logs/, строит сводные графики
(mean ± std) и выводит таблицу метрик.

Использование:
    python analyze.py                    # обе ветки
    python analyze.py --method genetic   # только GA
    python analyze.py --method dqn       # только DQN

Результаты сохраняются в analysis/.
"""

import os
import sys
import csv
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(ROOT, 'analysis')


def load_genetic_csv(path):
    """Возвращает dict с массивами numpy."""
    gens, bests, avgs = [], [], []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gens.append(int(row['generation']))
            bests.append(float(row['best']))
            avgs.append(float(row['avg']))
    return {
        'generation': np.array(gens),
        'best': np.array(bests),
        'avg': np.array(avgs),
    }


def load_dqn_csv(path):
    eps, scores, bests, avgs = [], [], [], []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(int(row['episode']))
            scores.append(float(row['score']))
            bests.append(float(row['best_score']))
            avgs.append(float(row['avg_last_50']))
    return {
        'episode': np.array(eps),
        'score': np.array(scores),
        'best': np.array(bests),
        'avg_50': np.array(avgs),
    }


def aggregate(runs, key):
    """Стэкаем массивы по seed и считаем mean/std/min/max."""
    # Подгоняем все массивы под минимальную длину (на случай разной длины прогонов)
    min_len = min(len(r[key]) for r in runs)
    stacked = np.array([r[key][:min_len] for r in runs])
    return {
        'mean': stacked.mean(axis=0),
        'std': stacked.std(axis=0),
        'min': stacked.min(axis=0),
        'max': stacked.max(axis=0),
        'all': stacked,
    }


# ------------------------------------------------------------------ #
#  Анализ GA                                                           #
# ------------------------------------------------------------------ #

def analyze_genetic():
    logs = sorted(glob.glob(os.path.join(ROOT, 'genetic', 'logs', 'genetic_seed*.csv')))
    if not logs:
        print("[GA] CSV-логов не найдено в genetic/logs/. Пропускаю.")
        return None

    print(f"[GA] Найдено {len(logs)} прогонов: {[os.path.basename(p) for p in logs]}")

    runs = [load_genetic_csv(p) for p in logs]
    seeds = [int(os.path.basename(p).replace('genetic_seed', '').replace('.csv', ''))
             for p in logs]

    gens = runs[0]['generation']
    agg_best = aggregate(runs, 'best')
    agg_avg = aggregate(runs, 'avg')

    # График
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(gens, agg_best['mean'], color='#5B4FD6', linewidth=2, label='Лучший — mean')
    ax.fill_between(gens, agg_best['mean'] - agg_best['std'],
                    agg_best['mean'] + agg_best['std'],
                    color='#5B4FD6', alpha=0.2, label='Лучший — ±std')
    ax.plot(gens, agg_avg['mean'], color='#1D9E75', linewidth=2, linestyle='--',
            label='Средний — mean')
    ax.fill_between(gens, agg_avg['mean'] - agg_avg['std'],
                    agg_avg['mean'] + agg_avg['std'],
                    color='#1D9E75', alpha=0.15)

    ax.set_xlabel('Поколение')
    ax.set_ylabel('Счёт')
    ax.set_title(f'GA — агрегация по {len(seeds)} seed (seeds: {seeds})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    plot_path = os.path.join(ANALYSIS_DIR, 'genetic_aggregated.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[GA] График: {plot_path}")

    # Сводная таблица
    finals_best = np.array([r['best'][-1] for r in runs])
    finals_avg = np.array([r['avg'][-1] for r in runs])
    global_best = np.array([r['best'].max() for r in runs])

    print("\n--- GA: финальные метрики ---")
    print(f"  Seeds:                {seeds}")
    print(f"  Лучшие за прогон:     {[int(v) for v in global_best]}")
    print(f"  Лучшие mean ± std:    {global_best.mean():.0f} ± {global_best.std():.0f}")
    print(f"  Финальный best:       {finals_best.mean():.0f} ± {finals_best.std():.0f}")
    print(f"  Финальный avg:        {finals_avg.mean():.0f} ± {finals_avg.std():.0f}")

    return {
        'seeds': seeds,
        'global_best_mean': global_best.mean(),
        'global_best_std': global_best.std(),
        'final_avg_mean': finals_avg.mean(),
        'final_avg_std': finals_avg.std(),
    }


# ------------------------------------------------------------------ #
#  Анализ DQN                                                          #
# ------------------------------------------------------------------ #

def analyze_dqn():
    logs = sorted(glob.glob(os.path.join(ROOT, 'dqn', 'logs', 'dqn_seed*.csv')))
    if not logs:
        print("[DQN] CSV-логов не найдено в dqn/logs/. Пропускаю.")
        return None

    print(f"[DQN] Найдено {len(logs)} прогонов: {[os.path.basename(p) for p in logs]}")

    runs = [load_dqn_csv(p) for p in logs]
    seeds = [int(os.path.basename(p).replace('dqn_seed', '').replace('.csv', ''))
             for p in logs]

    episodes = runs[0]['episode']
    agg_score = aggregate(runs, 'score')
    agg_avg = aggregate(runs, 'avg_50')
    agg_best = aggregate(runs, 'best')

    # График
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(episodes, agg_avg['mean'], color='#5B4FD6', linewidth=2,
            label='Среднее (50 эп.) — mean')
    ax.fill_between(episodes, agg_avg['mean'] - agg_avg['std'],
                    agg_avg['mean'] + agg_avg['std'],
                    color='#5B4FD6', alpha=0.2, label='±std')
    ax.set_xlabel('Эпизод')
    ax.set_ylabel('Счёт')
    ax.set_title(f'DQN — скользящее среднее, {len(seeds)} seed')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(episodes, agg_best['mean'], color='#D85A30', linewidth=2,
            label='Лучший — mean')
    ax.fill_between(episodes, agg_best['mean'] - agg_best['std'],
                    agg_best['mean'] + agg_best['std'],
                    color='#D85A30', alpha=0.2, label='±std')
    ax.set_xlabel('Эпизод')
    ax.set_ylabel('Лучший счёт')
    ax.set_title(f'DQN — рекорд по эпизодам, {len(seeds)} seed (seeds: {seeds})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    plot_path = os.path.join(ANALYSIS_DIR, 'dqn_aggregated.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[DQN] График: {plot_path}")

    # Метрики
    finals_avg = np.array([r['avg_50'][-1] for r in runs])
    global_best = np.array([r['best'].max() for r in runs])

    print("\n--- DQN: финальные метрики ---")
    print(f"  Seeds:                  {seeds}")
    print(f"  Лучшие за прогон:       {[int(v) for v in global_best]}")
    print(f"  Лучшие mean ± std:      {global_best.mean():.0f} ± {global_best.std():.0f}")
    print(f"  Финальный avg(50):      {finals_avg.mean():.0f} ± {finals_avg.std():.0f}")
    print(f"  Медиана лучших:         {np.median(global_best):.0f}")

    return {
        'seeds': seeds,
        'global_best_mean': global_best.mean(),
        'global_best_std': global_best.std(),
        'final_avg_50_mean': finals_avg.mean(),
        'final_avg_50_std': finals_avg.std(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['genetic', 'dqn', 'both'], default='both')
    args = parser.parse_args()

    print("=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
    print("=" * 60)

    if args.method in ('genetic', 'both'):
        print()
        analyze_genetic()

    if args.method in ('dqn', 'both'):
        print()
        analyze_dqn()

    print()
    print(f"Графики сохранены в: {ANALYSIS_DIR}/")


if __name__ == '__main__':
    main()

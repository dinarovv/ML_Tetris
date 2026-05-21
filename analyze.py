"""
Агрегация и анализ результатов обучения по нескольким seed.

ВЕРСИЯ ПОСЛЕ ПАКЕТА 4.2:
  - Понимает обе версии CSV (старую и новую с eval_score).
  - Для DQN строит отдельный график eval-результатов
    (это «честная» метрика без exploration-шума).

Использование:
    python analyze.py
    python analyze.py --method genetic
    python analyze.py --method dqn
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


# ------------------------------------------------------------------ #
#  Загрузка CSV (поддерживает обе версии формата)                     #
# ------------------------------------------------------------------ #

def load_genetic_csv(path):
    gens, bests, avgs = [], [], []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            gens.append(int(row['generation']))
            bests.append(float(row['best']))
            avgs.append(float(row['avg']))
    return {
        'generation': np.array(gens),
        'best': np.array(bests),
        'avg': np.array(avgs),
    }


def load_dqn_csv(path):
    """
    Поддерживает обе версии CSV:
      Старая: episode, score, best_score, avg_last_50, epsilon, avg_loss
      Новая:  episode, train_score, eval_score, best_eval, steps,
              lines_cleared, epsilon, avg_loss
    """
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    if not rows:
        return None

    has_eval = 'eval_score' in rows[0]
    eps   = [int(r['episode']) for r in rows]

    if has_eval:
        train = [float(r['train_score']) for r in rows]
        # eval_score есть не на каждом эпизоде — собираем отдельно
        eval_eps, eval_scores = [], []
        best_eval = []
        for r in rows:
            best_eval.append(float(r['best_eval']))
            if r['eval_score'] != '':
                eval_eps.append(int(r['episode']))
                eval_scores.append(float(r['eval_score']))
        return {
            'episode':    np.array(eps),
            'train':      np.array(train),
            'eval_eps':   np.array(eval_eps),
            'eval':       np.array(eval_scores),
            'best_eval':  np.array(best_eval),
            'format':     'new',
        }
    else:
        # Старый формат
        scores = [float(r['score']) for r in rows]
        bests  = [float(r['best_score']) for r in rows]
        avgs   = [float(r['avg_last_50']) for r in rows]
        return {
            'episode': np.array(eps),
            'train':   np.array(scores),
            'best':    np.array(bests),
            'avg_50':  np.array(avgs),
            'format':  'old',
        }


def aggregate(runs, key):
    min_len = min(len(r[key]) for r in runs)
    stacked = np.array([r[key][:min_len] for r in runs])
    return {
        'mean': stacked.mean(axis=0),
        'std':  stacked.std(axis=0),
        'min':  stacked.min(axis=0),
        'max':  stacked.max(axis=0),
        'all':  stacked,
    }


# ------------------------------------------------------------------ #
#  GA                                                                  #
# ------------------------------------------------------------------ #

def analyze_genetic():
    logs = sorted(glob.glob(
        os.path.join(ROOT, 'genetic', 'logs', 'genetic_seed*.csv')))
    if not logs:
        print("[GA] CSV-логов не найдено. Пропускаю.")
        return None

    seeds = [int(os.path.basename(p)
                 .replace('genetic_seed', '').replace('.csv', ''))
             for p in logs]
    print(f"[GA] Найдено {len(logs)} прогонов: {seeds}")

    runs = [load_genetic_csv(p) for p in logs]
    gens = runs[0]['generation']
    agg_best = aggregate(runs, 'best')
    agg_avg  = aggregate(runs, 'avg')

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(gens, agg_best['mean'], color='#5B4FD6', linewidth=2,
            label='Лучший — mean')
    ax.fill_between(gens,
                    agg_best['mean'] - agg_best['std'],
                    agg_best['mean'] + agg_best['std'],
                    color='#5B4FD6', alpha=0.2, label='Лучший — ±std')
    ax.plot(gens, agg_avg['mean'], color='#1D9E75', linewidth=2,
            linestyle='--', label='Средний — mean')
    ax.fill_between(gens,
                    agg_avg['mean'] - agg_avg['std'],
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

    finals_best = np.array([r['best'][-1] for r in runs])
    finals_avg  = np.array([r['avg'][-1]  for r in runs])
    global_best = np.array([r['best'].max() for r in runs])

    print("\n--- GA: финальные метрики ---")
    print(f"  Seeds:               {seeds}")
    print(f"  Лучшие за прогон:    {[int(v) for v in global_best]}")
    print(f"  Лучший mean ± std:   "
          f"{global_best.mean():.0f} ± {global_best.std():.0f}")
    print(f"  Финальный best:      "
          f"{finals_best.mean():.0f} ± {finals_best.std():.0f}")
    print(f"  Финальный avg:       "
          f"{finals_avg.mean():.0f} ± {finals_avg.std():.0f}")


# ------------------------------------------------------------------ #
#  DQN                                                                 #
# ------------------------------------------------------------------ #

def analyze_dqn():
    logs = sorted(glob.glob(
        os.path.join(ROOT, 'dqn', 'logs', 'dqn_seed*.csv')))
    if not logs:
        print("[DQN] CSV-логов не найдено. Пропускаю.")
        return None

    seeds = [int(os.path.basename(p)
                 .replace('dqn_seed', '').replace('.csv', ''))
             for p in logs]
    print(f"[DQN] Найдено {len(logs)} прогонов: {seeds}")

    runs = [load_dqn_csv(p) for p in logs]
    has_eval = all(r['format'] == 'new' for r in runs)

    if has_eval:
        return _analyze_dqn_new(runs, seeds)
    else:
        return _analyze_dqn_old(runs, seeds)


def _analyze_dqn_new(runs, seeds):
    """Новый формат с eval-метрикой."""
    episodes = runs[0]['episode']
    agg_train = aggregate(runs, 'train')

    # Eval по эпизодам (одинаковые позиции, агрегируем)
    eval_eps_all = runs[0]['eval_eps']
    eval_stacked = []
    min_eval_len = min(len(r['eval']) for r in runs)
    for r in runs:
        eval_stacked.append(r['eval'][:min_eval_len])
    eval_stacked = np.array(eval_stacked)
    eval_mean = eval_stacked.mean(axis=0)
    eval_std  = eval_stacked.std(axis=0)
    eval_eps  = eval_eps_all[:min_eval_len]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # train скользящее среднее
    ax = axes[0]
    window = 20
    for r, color in zip(runs, ['#5B4FD6', '#1D9E75', '#D85A30']):
        train = r['train']
        if len(train) >= window:
            ma = np.convolve(train, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(train)), ma,
                    color=color, alpha=0.4, linewidth=1)

    # eval с фоном
    ax.plot(eval_eps, eval_mean, 'o-', color='#1a1a2e',
            linewidth=2, markersize=4, label='Eval mean (ε=0)')
    ax.fill_between(eval_eps, eval_mean - eval_std, eval_mean + eval_std,
                    color='#1a1a2e', alpha=0.15, label='Eval ±std')
    ax.set_xlabel('Эпизод')
    ax.set_ylabel('Счёт')
    ax.set_title(f'DQN — eval-метрика по {len(seeds)} seed (seeds: {seeds})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # best_eval
    ax = axes[1]
    agg_best = aggregate(runs, 'best_eval')
    ax.plot(episodes, agg_best['mean'], color='#D85A30', linewidth=2,
            label='Best eval — mean')
    ax.fill_between(episodes,
                    agg_best['mean'] - agg_best['std'],
                    agg_best['mean'] + agg_best['std'],
                    color='#D85A30', alpha=0.2, label='±std')
    ax.set_xlabel('Эпизод')
    ax.set_ylabel('Лучший eval-счёт')
    ax.set_title('DQN — рекорд по eval-эпизодам')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    plot_path = os.path.join(ANALYSIS_DIR, 'dqn_aggregated.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[DQN] График: {plot_path}")

    final_evals = np.array([r['eval'][-1] for r in runs])
    best_evals  = np.array([r['eval'].max() for r in runs])
    final_trains = np.array([r['train'][-50:].mean() for r in runs])

    print("\n--- DQN: финальные метрики (новый формат, с eval) ---")
    print(f"  Seeds:               {seeds}")
    print(f"  Лучшие eval:         {[int(v) for v in best_evals]}")
    print(f"  Лучший eval ± std:   "
          f"{best_evals.mean():.0f} ± {best_evals.std():.0f}")
    print(f"  Финальный eval:      "
          f"{final_evals.mean():.0f} ± {final_evals.std():.0f}")
    print(f"  Финальный train avg-50: "
          f"{final_trains.mean():.0f} ± {final_trains.std():.0f}")


def _analyze_dqn_old(runs, seeds):
    """Старый формат — для совместимости."""
    episodes = runs[0]['episode']
    agg_avg  = aggregate(runs, 'avg_50')
    agg_best = aggregate(runs, 'best')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(episodes, agg_avg['mean'], color='#5B4FD6', linewidth=2,
            label='Среднее (50 эп.) — mean')
    ax.fill_between(episodes,
                    agg_avg['mean'] - agg_avg['std'],
                    agg_avg['mean'] + agg_avg['std'],
                    color='#5B4FD6', alpha=0.2)
    ax.set_xlabel('Эпизод')
    ax.set_ylabel('Скользящее среднее (50 эп.)')
    ax.set_title(f'DQN — avg(50) (старый формат, seeds: {seeds})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(episodes, agg_best['mean'], color='#D85A30', linewidth=2,
            label='Лучший — mean')
    ax.fill_between(episodes,
                    agg_best['mean'] - agg_best['std'],
                    agg_best['mean'] + agg_best['std'],
                    color='#D85A30', alpha=0.2)
    ax.set_xlabel('Эпизод')
    ax.set_ylabel('Лучший счёт')
    ax.set_title('DQN — рекорд по эпизодам')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    plot_path = os.path.join(ANALYSIS_DIR, 'dqn_aggregated.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[DQN] График: {plot_path}")

    finals = np.array([r['avg_50'][-1] for r in runs])
    bests  = np.array([r['best'].max() for r in runs])
    print("\n--- DQN: финальные метрики (старый формат) ---")
    print(f"  Seeds:           {seeds}")
    print(f"  Лучшие:          {[int(v) for v in bests]}")
    print(f"  Лучший ± std:    {bests.mean():.0f} ± {bests.std():.0f}")
    print(f"  Финальный avg50: {finals.mean():.0f} ± {finals.std():.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',
                        choices=['genetic', 'dqn', 'both'], default='both')
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

    print(f"\nГрафики: {ANALYSIS_DIR}/")


if __name__ == '__main__':
    main()

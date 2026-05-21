"""
Единый пайплайн обучения и анализа — для защиты.

Запускает обучение указанного метода (или обоих) на нескольких seed,
собирает все артефакты в структурированной папке `runs/<timestamp>/`,
строит агрегированные графики и итоговую таблицу.

Использование:
    # Стандартный сценарий для защиты:
    python run_experiments.py --method both --seeds 42,123,777

    # Только DQN с кастомным конфигом:
    python run_experiments.py --method dqn \\
        --config-dqn configs/dqn.json --seeds 42,123,777

    # Быстрый тест:
    python run_experiments.py --method genetic --seeds 42 --generations 2

Структура артефактов:
    runs/experiment_<timestamp>/
        config_dqn.json                # snapshot конфига DQN
        config_genetic.json            # snapshot конфига GA
        genetic_seed42/
            logs/genetic_seed42.csv
            best_weights_seed42.json
            training_plot_seed42.png
        genetic_seed123/ ...
        dqn_seed42/
            logs/dqn_seed42.csv
            dqn_model_seed42.pt
            training_plot_seed42.png
        dqn_seed123/ ...
        analysis/                      # агрегированные графики
            genetic_aggregated.png
            dqn_aggregated.png
        summary.json                   # все метрики в JSON
        summary.csv                    # сводная таблица для отчёта
"""

import os
import sys
import json
import csv
import shutil
import argparse
from datetime import datetime
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DEFAULT_CONFIG_DQN     = os.path.join(ROOT, 'configs', 'dqn.json')
DEFAULT_CONFIG_GENETIC = os.path.join(ROOT, 'configs', 'genetic.json')


# ------------------------------------------------------------------ #
#  Запуск одного метода                                                #
# ------------------------------------------------------------------ #

def run_method(method, config_path, seeds, run_dir, overrides=None):
    """Запускает обучение указанного метода через все seeds."""
    overrides = overrides or {}

    # Импорт нужного train-модуля
    if method == 'dqn':
        from dqn.train import (load_config, merge_params, run_single,
                               DEFAULTS)
    elif method == 'genetic':
        from genetic.train import (load_config, merge_params, run_single,
                                   DEFAULTS)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Параметры
    config = load_config(config_path) if config_path else {}
    params = merge_params(DEFAULTS, config)
    params.update(overrides)

    # Сохраняем snapshot конфига
    if config_path and os.path.exists(config_path):
        snapshot = os.path.join(run_dir, f'config_{method}.json')
        shutil.copy(config_path, snapshot)
    else:
        # Сохраняем дефолтные параметры
        snapshot = os.path.join(run_dir, f'config_{method}.json')
        with open(snapshot, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)

    # Запуск для каждого seed
    results = []
    for seed in seeds:
        seed_dir = os.path.join(run_dir, f'{method}_seed{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        print(f"\n{'#' * 64}")
        print(f"# {method.upper()} — seed={seed}")
        print(f"# Папка: {seed_dir}")
        print(f"{'#' * 64}\n")

        result = run_single(seed, params, output_dir=seed_dir)
        results.append(result)

    return results, params


# ------------------------------------------------------------------ #
#  Агрегация результатов                                               #
# ------------------------------------------------------------------ #

def aggregate_dqn(results):
    """Сводные метрики для DQN."""
    best_evals  = np.array([r['best_eval']  for r in results])
    best_trains = np.array([r['best_train'] for r in results])
    final_evals = np.array([r['final_eval'] for r in results])
    return {
        'best_eval_mean':   float(best_evals.mean()),
        'best_eval_std':    float(best_evals.std()),
        'best_eval_median': float(np.median(best_evals)),
        'best_train_mean':  float(best_trains.mean()),
        'best_train_std':   float(best_trains.std()),
        'final_eval_mean':  float(final_evals.mean()),
        'final_eval_std':   float(final_evals.std()),
        'individual':       [{'seed':       r['seed'],
                              'best_eval':  r['best_eval'],
                              'best_train': r['best_train'],
                              'final_eval': r['final_eval']}
                             for r in results],
    }


def aggregate_genetic(results):
    """Сводные метрики для GA."""
    bests       = np.array([r['best_score'] for r in results])
    finals_best = np.array([r['final_best'] for r in results])
    finals_avg  = np.array([r['final_avg']  for r in results])
    return {
        'best_mean':         float(bests.mean()),
        'best_std':          float(bests.std()),
        'best_median':       float(np.median(bests)),
        'final_best_mean':   float(finals_best.mean()),
        'final_best_std':    float(finals_best.std()),
        'final_avg_mean':    float(finals_avg.mean()),
        'final_avg_std':     float(finals_avg.std()),
        'individual':        [{'seed':       r['seed'],
                               'best':       r['best_score'],
                               'final_best': r['final_best'],
                               'final_avg':  r['final_avg']}
                              for r in results],
    }


# ------------------------------------------------------------------ #
#  Запуск анализа (агрегированные графики)                            #
# ------------------------------------------------------------------ #

def build_aggregated_plots(run_dir, methods):
    """
    Собирает CSV-логи из run_dir/<method>_seed*/logs/ во временную папку,
    совместимую со структурой analyze.py, и запускает анализ.
    """
    # Перемещаем CSV в плоскую структуру для analyze.py
    temp_logs = {}
    for method in methods:
        method_logs_dir = os.path.join(
            run_dir, '_analyze_temp', method, 'logs'
        )
        os.makedirs(method_logs_dir, exist_ok=True)
        temp_logs[method] = method_logs_dir

        for entry in os.listdir(run_dir):
            seed_dir = os.path.join(run_dir, entry)
            if (os.path.isdir(seed_dir)
                    and entry.startswith(f'{method}_seed')):
                src_logs = os.path.join(seed_dir, 'logs')
                if os.path.isdir(src_logs):
                    for f in os.listdir(src_logs):
                        if f.endswith('.csv'):
                            shutil.copy(os.path.join(src_logs, f),
                                        method_logs_dir)

    # Запускаем analyze.py с указанием базовой папки
    analyze_root = os.path.join(run_dir, '_analyze_temp')
    sys.path.insert(0, ROOT)
    from analyze import analyze_genetic, analyze_dqn

    # Подменяем ROOT в analyze, чтобы он искал в нашей временной папке
    import analyze
    original_root = analyze.ROOT
    analyze.ROOT = analyze_root
    analyze.ANALYSIS_DIR = os.path.join(run_dir, 'analysis')

    try:
        if 'genetic' in methods:
            print(f"\n--- Анализ GA ---")
            analyze_genetic()
        if 'dqn' in methods:
            print(f"\n--- Анализ DQN ---")
            analyze_dqn()
    finally:
        analyze.ROOT = original_root

    # Удаляем временные файлы
    temp_dir = os.path.join(run_dir, '_analyze_temp')
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)


# ------------------------------------------------------------------ #
#  Summary                                                             #
# ------------------------------------------------------------------ #

def write_summary(run_dir, summary):
    """Сохраняем summary.json и summary.csv."""
    json_path = os.path.join(run_dir, 'summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {json_path}")

    csv_path = os.path.join(run_dir, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'metric', 'value'])

        if 'genetic' in summary['aggregated']:
            agg = summary['aggregated']['genetic']
            w.writerow(['genetic', 'best_mean',       f"{agg['best_mean']:.1f}"])
            w.writerow(['genetic', 'best_std',        f"{agg['best_std']:.1f}"])
            w.writerow(['genetic', 'final_avg_mean',  f"{agg['final_avg_mean']:.1f}"])
            w.writerow(['genetic', 'final_avg_std',   f"{agg['final_avg_std']:.1f}"])
            for ind in agg['individual']:
                w.writerow(['genetic',
                            f"seed{ind['seed']}_best",       f"{ind['best']:.1f}"])
                w.writerow(['genetic',
                            f"seed{ind['seed']}_final_avg",  f"{ind['final_avg']:.1f}"])

        if 'dqn' in summary['aggregated']:
            agg = summary['aggregated']['dqn']
            w.writerow(['dqn', 'best_eval_mean',   f"{agg['best_eval_mean']:.1f}"])
            w.writerow(['dqn', 'best_eval_std',    f"{agg['best_eval_std']:.1f}"])
            w.writerow(['dqn', 'final_eval_mean',  f"{agg['final_eval_mean']:.1f}"])
            w.writerow(['dqn', 'final_eval_std',   f"{agg['final_eval_std']:.1f}"])
            for ind in agg['individual']:
                w.writerow(['dqn',
                            f"seed{ind['seed']}_best_eval",  f"{ind['best_eval']:.1f}"])
                w.writerow(['dqn',
                            f"seed{ind['seed']}_final_eval", f"{ind['final_eval']:.1f}"])

    print(f"  CSV:  {csv_path}")


def print_summary_table(summary):
    """Печатает сводку в консоль."""
    print()
    print("=" * 70)
    print(" ИТОГОВАЯ СВОДКА")
    print("=" * 70)
    print(f"  Эксперимент: {summary['experiment_name']}")
    print(f"  Запущен:     {summary['timestamp']}")
    print(f"  Seeds:       {summary['seeds']}")
    print()

    agg = summary['aggregated']

    if 'genetic' in agg:
        a = agg['genetic']
        print(" GENETIC ALGORITHM")
        print(" " + "-" * 68)
        print(f"  Лучший за прогон:   "
              f"{[int(i['best']) for i in a['individual']]}")
        print(f"  Mean ± std:          "
              f"{a['best_mean']:.0f} ± {a['best_std']:.0f}")
        print(f"  Медиана:             {a['best_median']:.0f}")
        print(f"  Финальный avg:       "
              f"{a['final_avg_mean']:.0f} ± {a['final_avg_std']:.0f}")
        print()

    if 'dqn' in agg:
        a = agg['dqn']
        print(" DQN")
        print(" " + "-" * 68)
        print(f"  Best eval за прогон: "
              f"{[int(i['best_eval']) for i in a['individual']]}")
        print(f"  Best eval mean ± std: "
              f"{a['best_eval_mean']:.0f} ± {a['best_eval_std']:.0f}")
        print(f"  Медиана:             {a['best_eval_median']:.0f}")
        print(f"  Финальный eval:      "
              f"{a['final_eval_mean']:.0f} ± {a['final_eval_std']:.0f}")
        print()

    print(" Артефакты:")
    print(f"  {summary['run_dir']}")
    print("=" * 70)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description='Единый пайплайн обучения')
    parser.add_argument('--method',
                        choices=['genetic', 'dqn', 'both'], default='both',
                        help='Какой метод обучать')
    parser.add_argument('--seeds', type=str, default='42,123,777',
                        help='Seeds через запятую (по умолчанию 42,123,777)')
    parser.add_argument('--config-dqn', type=str,
                        default=DEFAULT_CONFIG_DQN,
                        help='Путь к конфигу DQN')
    parser.add_argument('--config-genetic', type=str,
                        default=DEFAULT_CONFIG_GENETIC,
                        help='Путь к конфигу GA')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Переопределить EPISODES в DQN-конфиге')
    parser.add_argument('--generations', type=int, default=None,
                        help='Переопределить GENERATIONS в GA-конфиге')
    parser.add_argument('--name', type=str, default=None,
                        help='Имя эксперимента (по умолчанию timestamp)')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Создаём папку эксперимента
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    name = args.name or f'experiment_{timestamp}'
    run_dir = os.path.join(ROOT, 'runs', name)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 70)
    print(f" ЗАПУСК ЭКСПЕРИМЕНТА: {name}")
    print(f" Папка: {run_dir}")
    print(f" Метод(ы): {args.method}, seeds: {seeds}")
    print("=" * 70)

    methods = (['genetic', 'dqn'] if args.method == 'both'
               else [args.method])

    summary = {
        'experiment_name': name,
        'timestamp':       datetime.now().isoformat(),
        'run_dir':         run_dir,
        'methods':         methods,
        'seeds':           seeds,
        'params':          {},
        'individual':      {},
        'aggregated':      {},
    }

    # ---- GA ----
    if 'genetic' in methods:
        ga_overrides = {}
        if args.generations is not None:
            ga_overrides['generations'] = args.generations
        ga_results, ga_params = run_method(
            'genetic', args.config_genetic, seeds, run_dir, ga_overrides
        )
        summary['params']['genetic']     = ga_params
        summary['individual']['genetic'] = ga_results
        summary['aggregated']['genetic'] = aggregate_genetic(ga_results)

    # ---- DQN ----
    if 'dqn' in methods:
        dqn_overrides = {}
        if args.episodes is not None:
            dqn_overrides['episodes'] = args.episodes
        dqn_results, dqn_params = run_method(
            'dqn', args.config_dqn, seeds, run_dir, dqn_overrides
        )
        summary['params']['dqn']     = dqn_params
        summary['individual']['dqn'] = dqn_results
        summary['aggregated']['dqn'] = aggregate_dqn(dqn_results)

    # ---- Агрегированные графики ----
    print(f"\n{'=' * 70}")
    print(" ПОСТРОЕНИЕ АГРЕГИРОВАННЫХ ГРАФИКОВ")
    print("=" * 70)
    try:
        build_aggregated_plots(run_dir, methods)
    except Exception as e:
        print(f"  [!] Не удалось построить графики: {e}")

    # ---- Сохранение summary ----
    print(f"\n{'=' * 70}")
    print(" СОХРАНЕНИЕ СВОДКИ")
    print("=" * 70)
    write_summary(run_dir, summary)

    # ---- Печать таблицы ----
    print_summary_table(summary)


if __name__ == '__main__':
    main()

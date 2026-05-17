"""
Запуск обучения — ветка Генетический алгоритм.

Использование:
    python genetic/train.py

После обучения сохраняет лучшие веса и строит графики.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from genetic.agent import GeneticAlgorithm

# ------------------------------------------------------------------ #
#  Параметры обучения                                                  #
# ------------------------------------------------------------------ #

POPULATION_SIZE = 30      # размер популяции
GENERATIONS = 80          # сколько поколений эволюции
GAMES_PER_AGENT = 5      # игр на оценку агента
MAX_STEPS = 400           # максимум ходов за игру

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'best_weights.json')
PLOT_PATH = os.path.join(os.path.dirname(__file__), 'training_plot.png')


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
    plt.show()


def main():
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        games_per_agent=GAMES_PER_AGENT,
        max_steps=MAX_STEPS,
    )

    best_agent = ga.run(generations=GENERATIONS)

    save_weights(best_agent, SAVE_PATH)
    plot_training(ga.history, PLOT_PATH)


if __name__ == '__main__':
    main()

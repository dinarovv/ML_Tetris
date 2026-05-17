"""
Ветка 1: Генетический алгоритм.

Идея:
  Каждый агент — это набор из 4 весов, которые оценивают каждый ход
  по признакам поля (дыры, высота, неровность, очищенные линии).
  Лучшие агенты "скрещиваются" и порождают следующее поколение.

Не требует никаких ML-библиотек — только numpy.
"""

import numpy as np
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game.tetris import TetrisGame


# ------------------------------------------------------------------ #
#  Агент                                                               #
# ------------------------------------------------------------------ #

class GeneticAgent:
    """
    Агент с 4 весами: [holes, total_height, bumpiness, lines_cleared].
    Оценивает каждый ход как взвешенную сумму признаков
    и выбирает ход с максимальным счётом.
    """

    NUM_WEIGHTS = 4

    def __init__(self, weights=None):
        if weights is not None:
            self.weights = np.array(weights, dtype=float)
        else:
            # Инициализация с правильными знаками:
            # holes, height, bumpiness — отрицательные (штрафы)
            # lines_cleared — положительный (награда)
            self.weights = np.array([
                np.random.uniform(-1.0, -0.1),  # holes
                np.random.uniform(-1.0, -0.1),  # height
                np.random.uniform(-1.0, -0.1),  # bumpiness
                np.random.uniform(0.1, 1.0),  # lines_cleared
            ])

    def score_move(self, features):
        """Линейная оценка хода по признакам."""
        return (self.weights[0] * features['holes'] +
                self.weights[1] * features['total_height'] +
                self.weights[2] * features['bumpiness'] +
                self.weights[3] * features['lines_cleared'])

    def choose_move(self, moves):
        """Выбираем ход с наибольшим score_move."""
        if not moves:
            return None
        return max(moves, key=lambda m: self.score_move(m['features']))

    # -------- Генетические операции -------- #

    def mutate(self, rate=0.15, strength=0.3):
        """Мутация: каждый вес с вероятностью rate сдвигается случайно."""
        new_weights = self.weights.copy()
        for i in range(self.NUM_WEIGHTS):
            if random.random() < rate:
                new_weights[i] += np.random.randn() * strength
        return GeneticAgent(new_weights)

    @staticmethod
    def crossover(parent1, parent2):
        """
        Одноточечное скрещивание.
        Первая часть весов — от одного родителя, вторая — от другого.
        """
        point = random.randint(1, GeneticAgent.NUM_WEIGHTS - 1)
        child_w = np.concatenate([
            parent1.weights[:point],
            parent2.weights[point:]
        ])
        return GeneticAgent(child_w)

    def __repr__(self):
        w = self.weights
        return (f"GeneticAgent(holes={w[0]:.3f}, height={w[1]:.3f}, "
                f"bump={w[2]:.3f}, lines={w[3]:.3f})")


# ------------------------------------------------------------------ #
#  Алгоритм эволюции                                                   #
# ------------------------------------------------------------------ #

class GeneticAlgorithm:
    """
    Управляет популяцией агентов и запускает эволюцию по поколениям.

    Параметры:
      population_size  — размер популяции (рекомендуется 50–100)
      games_per_agent  — сколько игр проигрывает каждый агент (усредняем)
      max_steps        — лимит ходов за одну игру (защита от бесконечных игр)
      elite_frac       — доля лучших, которых оставляем без изменений
    """

    def __init__(self,
                 population_size=50,
                 games_per_agent=3,
                 max_steps=500,
                 elite_frac=0.2):
        self.population_size = population_size
        self.games_per_agent = games_per_agent
        self.max_steps = max_steps
        self.elite_count = max(2, int(population_size * elite_frac))

        self.population = [GeneticAgent() for _ in range(population_size)]
        self.generation = 0
        self.history = []   # список словарей для matplotlib
        self.best_agent = None
        self.best_score = 0

    def _evaluate_agent(self, agent):
        """Средний счёт агента за несколько игр."""
        total = 0
        for _ in range(self.games_per_agent):
            game = TetrisGame()
            steps = 0
            while not game.game_over and steps < self.max_steps:
                moves = game.get_possible_moves()
                move = agent.choose_move(moves)
                if move is None:
                    break
                game.step(move['x'], move['rotation'])
                steps += 1
            total += game.score
        return total / self.games_per_agent

    def evolve_one_generation(self):
        """
        Один цикл эволюции:
          1. Оцениваем всех агентов
          2. Отбираем элиту
          3. Скрещиваем и мутируем
          4. Формируем новое поколение
        """
        # Шаг 1: фитнес
        scored = [(agent, self._evaluate_agent(agent))
                  for agent in self.population]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_agent, best_score = scored[0]
        avg_score = np.mean([s for _, s in scored])

        # Сохраняем статистику для графиков
        record = {
            'generation': self.generation,
            'best': best_score,
            'avg': avg_score,
            'best_weights': best_agent.weights.copy(),
        }
        self.history.append(record)

        if best_score > self.best_score:
            self.best_score = best_score
            self.best_agent = GeneticAgent(best_agent.weights.copy())

        print(f"  Поколение {self.generation:3d} | "
              f"лучший={best_score:8.1f} | средний={avg_score:8.1f} | "
              f"веса={best_agent.weights.round(3)}")

        # Шаг 2: элита (копируем без изменений)
        elite = [agent for agent, _ in scored[:self.elite_count]]

        # Шаг 3–4: заполняем остаток скрещиванием + мутацией
        new_population = list(elite)
        while len(new_population) < self.population_size:
            p1, p2 = random.sample(elite, 2)
            child = GeneticAgent.crossover(p1, p2).mutate()
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return best_agent, best_score

    def run(self, generations=30):
        """Запустить полное обучение на заданное число поколений."""
        print("=" * 60)
        print(f"Генетический алгоритм: {self.population_size} особей, "
              f"{generations} поколений")
        print("=" * 60)

        for _ in range(generations):
            self.evolve_one_generation()

        print("\n✓ Обучение завершено!")
        print(f"  Лучший агент: {self.best_agent}")
        print(f"  Лучший счёт:  {self.best_score:.1f}")
        return self.best_agent

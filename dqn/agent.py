"""
Ветка 2: Deep Q-Network (DQN).

ВЕРСИЯ ПОСЛЕ ПАКЕТА 4.1 — исправлена формула target.

ИСТОРИЯ ИЗМЕНЕНИЙ:
  Пакет 1: max по всем next-ходам, разделён exploration и target
  Пакет 3: Huber loss + gradient clipping
  Пакет 4.1: ПРАВИЛЬНАЯ формула target для afterstate-формулировки

  Раньше target = r_текущего_хода + γ · max Q(next_afterstates)
  Стало  target = max_a (r_a + γ · Q_target(next_afterstate_a))

  Где r_a — награда за следующий ход (а не за тот, который привёл в state_feat).
  Это устраняет смещение Q-значений и согласует Bellman-уравнение
  с afterstate-семантикой Q-функции.

ИДЕЯ:
  Q-сеть оценивает значение afterstate — состояния поля после хода агента.
  Q(s_after) = ожидаемая будущая дисконтированная награда, начиная с s_after.
  При выборе хода агент сравнивает Q-значения всех возможных afterstate.
"""

import numpy as np
import random
from collections import deque
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[!] PyTorch не найден. Установи: pip install torch")


# ------------------------------------------------------------------ #
#  Награда за ход — общая функция (не требует PyTorch)                #
# ------------------------------------------------------------------ #

BOARD_WIDTH = 10  # ширина поля Тетриса

def _compute_reward(features):
    """
    Награда за ход: 1 (за выживание) + бонус за очищенные линии.
    Соответствует формуле reward в TetrisGame.step().

    Используется при:
      1) построении next_options для записи в memory
      2) выборе хода через _greedy_move (учёт непосредственной награды)
    """
    lc = features['lines_cleared']
    return 1.0 + (lc ** 2) * BOARD_WIDTH


if TORCH_AVAILABLE:

    class QNetwork(nn.Module):
        """4 → 64 → 64 → 1. Выход — Q-значение afterstate."""

        def __init__(self, input_dim=4, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x)

    class ReplayMemory:
        """
        Каждый элемент: (state_feat, next_options)
          state_feat   — признаки afterstate (после хода агента)
          next_options — список (reward_a, next_feat_a) для всех возможных
                         следующих ходов из этого afterstate. Пустой список
                         означает terminal состояние (Q должно быть 0).

        Важно: reward_a — это награда за СЛЕДУЮЩИЙ ход, а не за тот,
        который привёл в state_feat. Это ключевое отличие от прошлых версий.
        """

        def __init__(self, capacity=30_000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state_feat, next_options):
            self.buffer.append((state_feat, next_options))

        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)

        def __len__(self):
            return len(self.buffer)

    class DQNAgent:
        """DQN-агент с корректной Bellman-формулой для afterstate Q."""

        def __init__(self,
                     lr=0.001,
                     gamma=0.99,
                     epsilon=1.0,
                     epsilon_min=0.01,
                     epsilon_decay=0.998,
                     memory_size=30_000,
                     batch_size=128,
                     target_update=500,
                     use_huber=True,
                     grad_clip=1.0):

            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update = target_update
            self.grad_clip = grad_clip

            self.memory = ReplayMemory(memory_size)
            self.steps_done = 0

            self.policy_net = QNetwork()
            self.target_net = QNetwork()
            self._sync_target()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.loss_fn = nn.SmoothL1Loss() if use_huber else nn.MSELoss()

        def _sync_target(self):
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        @staticmethod
        def _feats_to_batch(features_list):
            return torch.stack([
                torch.FloatTensor([
                    f['holes'], f['total_height'],
                    f['bumpiness'], f['lines_cleared'],
                ])
                for f in features_list
            ])

        # -------- Выбор хода -------- #

        def _greedy_move(self, moves):
            """
            Чисто-greedy выбор: ход с максимальным (reward + γ * Q_policy(afterstate)).

            Раньше выбор был просто argmax Q(afterstate), что фактически
            эквивалентно argmax при одинаковом γ для всех. Теперь явно учитываем
            непосредственную награду каждого возможного хода.
            """
            if not moves:
                return None

            self.policy_net.eval()
            with torch.no_grad():
                feats_t = self._feats_to_batch([m['features'] for m in moves])
                qs = self.policy_net(feats_t).squeeze(-1)
                # Учитываем непосредственный reward каждого хода
                rewards = torch.FloatTensor([
                    _compute_reward(m['features']) for m in moves
                ])
                scores = rewards + self.gamma * qs
                best_idx = int(scores.argmax().item())
            return moves[best_idx]

        def choose_move(self, moves):
            """Epsilon-greedy для среды."""
            if not moves:
                return None
            if random.random() < self.epsilon:
                return random.choice(moves)
            return self._greedy_move(moves)

        # -------- Память и обучение -------- #

        def remember(self, state_feat, next_options):
            self.memory.push(state_feat, next_options)

        def train_step(self):
            """
            Один шаг обучения. Целевая функция:
                target = 0,                                 если state_feat терминальный
                target = max_a (r_a + γ · Q_target(s_a)),   иначе

            где (r_a, s_a) — пары (награда, afterstate-feat) для всех
            возможных next-ходов из state_feat.
            """
            if len(self.memory) < self.batch_size:
                return None

            batch = self.memory.sample(self.batch_size)

            states_t = self._feats_to_batch([s for s, _ in batch])

            self.policy_net.train()
            current_q = self.policy_net(states_t).squeeze(-1)

            target_values = torch.zeros(len(batch))
            with torch.no_grad():
                for i, (_, next_options) in enumerate(batch):
                    if not next_options:
                        # Terminal afterstate: Q = 0
                        target_values[i] = 0.0
                    else:
                        next_rewards = torch.FloatTensor(
                            [r for r, _ in next_options]
                        )
                        next_feats_t = self._feats_to_batch(
                            [f for _, f in next_options]
                        )
                        next_qs = self.target_net(next_feats_t).squeeze(-1)
                        # max_a (r_a + γ · Q_target(s_a))
                        target_values[i] = (
                            next_rewards + self.gamma * next_qs
                        ).max()

            loss = self.loss_fn(current_q, target_values)
            self.optimizer.zero_grad()
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), max_norm=self.grad_clip
                )

            self.optimizer.step()

            self.steps_done += 1
            if self.steps_done % self.target_update == 0:
                self._sync_target()

            return loss.item()

        def decay_epsilon(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def save(self, path):
            torch.save({
                'policy': self.policy_net.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps_done,
            }, path)
            print(f"  Модель сохранена: {path}")

        def load(self, path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy'])
            self._sync_target()
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.steps_done = checkpoint.get('steps', 0)
            print(f"  Модель загружена: {path}")

else:
    class DQNAgent:
        def __init__(self, **kwargs):
            raise RuntimeError("PyTorch не установлен. Запусти: pip install torch")

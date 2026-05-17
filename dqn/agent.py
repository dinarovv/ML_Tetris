"""
Ветка 2: Deep Q-Network (DQN).

ВЕРСИЯ ПОСЛЕ ПАКЕТОВ 1, 2 И 3:
  Пакет 1 (научная корректность):
    - Target = r + γ·max_a' Q_target(s', a') по всем next-ходам
    - Выбор для среды и расчёт target полностью разделены

  Пакет 3 (стабильность обучения):
    - Huber loss (SmoothL1Loss) вместо MSE — устойчивее к выбросам
    - Gradient clipping (max_norm=1.0) — защита от взрыва градиентов

Идея:
  Нейросеть принимает 4 признака состояния поля (после хода) и предсказывает
  Q-значение — ожидаемую суммарную награду из этого состояния.
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


if TORCH_AVAILABLE:

    class QNetwork(nn.Module):
        """4 → 64 → 64 → 1. Выход — Q-значение состояния."""

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
        Каждый элемент: (state_feat, reward, next_feats_list, done)
        next_feats_list — список признаков всех возможных next-ходов
        (нужно для корректного max в target).
        """

        def __init__(self, capacity=30_000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state_feat, reward, next_feats_list, done):
            self.buffer.append((state_feat, reward, next_feats_list, done))

        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)

        def __len__(self):
            return len(self.buffer)

    class DQNAgent:
        """
        Агент DQN.

        Параметры (по умолчанию — конфигурация для B3):
          lr             — скорость обучения Adam
          gamma          — дисконт-фактор
          epsilon        — начальная вероятность исследования
          epsilon_min    — минимальное значение epsilon
          epsilon_decay  — множитель убывания за эпизод
          memory_size    — объём Replay Memory
          batch_size     — размер мини-батча обучения
          target_update  — частота синхронизации target network (в шагах)
          use_huber      — Huber loss вместо MSE (по умолчанию True)
          grad_clip      — max-norm для gradient clipping (None = выключено)
        """

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
            # Huber устойчивее к выбросам в target, что важно для DQN
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
            """Greedy выбор без epsilon (через policy_net)."""
            if not moves:
                return None
            self.policy_net.eval()
            with torch.no_grad():
                feats_t = self._feats_to_batch([m['features'] for m in moves])
                qs = self.policy_net(feats_t).squeeze(-1)
                best_idx = int(qs.argmax().item())
            return moves[best_idx]

        def choose_move(self, moves):
            """Epsilon-greedy для взаимодействия со средой."""
            if not moves:
                return None
            if random.random() < self.epsilon:
                return random.choice(moves)
            return self._greedy_move(moves)

        # -------- Память и обучение -------- #

        def remember(self, state_feat, reward, next_feats_list, done):
            self.memory.push(state_feat, reward, next_feats_list, done)

        def train_step(self):
            """
            Один шаг обучения. Target = r + γ·max_a' Q_target(s', a').
            """
            if len(self.memory) < self.batch_size:
                return None

            batch = self.memory.sample(self.batch_size)

            states_t = self._feats_to_batch([s for s, _, _, _ in batch])
            rewards_t = torch.FloatTensor([r for _, r, _, _ in batch])
            dones_t = torch.FloatTensor([d for _, _, _, d in batch])

            self.policy_net.train()
            current_q = self.policy_net(states_t).squeeze(-1)

            target_q_max = torch.zeros(len(batch))
            with torch.no_grad():
                for i, (_, _, next_feats_list, done) in enumerate(batch):
                    if done or not next_feats_list:
                        target_q_max[i] = 0.0
                    else:
                        next_t = self._feats_to_batch(next_feats_list)
                        target_q_max[i] = self.target_net(next_t).squeeze(-1).max()

            target_q = rewards_t + self.gamma * target_q_max * (1 - dones_t)

            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()

            # Защита от взрыва градиентов (рекомендация научрука)
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

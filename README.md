# Тетрис с машинным обучением

Курсовая работа: «Реализация одной игры в жанре „головоломка“ с применением машинного обучения».

Реализованы две независимые ML-ветки для сравнения подходов:

| Ветка | Алгоритм | Библиотеки |
|---|---|---|
| `genetic/` | Генетический алгоритм (4 веса) | numpy |
| `dqn/` | Deep Q-Network (нейросеть) | numpy + PyTorch |

---

## Установка

```bash
pip install -r requirements.txt
pip install torch        # только для ветки DQN
```

---

## Структура проекта

```
tetris_project/
├── game/
│   └── tetris.py                # Общий движок
├── genetic/
│   ├── agent.py                 # GeneticAgent + GeneticAlgorithm
│   ├── train.py                 # Обучение (поддержка multi-seed)
│   └── logs/                    # CSV-логи прогонов
├── dqn/
│   ├── agent.py                 # DQNAgent + QNetwork + ReplayMemory
│   ├── train.py                 # Обучение (поддержка multi-seed)
│   └── logs/                    # CSV-логи прогонов
├── analyze.py                   # Агрегация результатов по нескольким seed
├── play.py                      # Визуализация через pygame
├── start.bat                    # Меню запуска (Windows)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Быстрый запуск

```bash
# Через меню (Windows)
start.bat

# Или напрямую:
python genetic/train.py            # Обучение GA (1 seed)
python dqn/train.py                # Обучение DQN (1 seed)
python play.py --mode genetic      # Демо
python play.py --mode dqn          # Демо
python play.py --mode human        # Ручная игра
```

---

## Конфигурация экспериментов

Все параметры зафиксированы в коде как константы в верхней части `train.py`. Это значения, под которые писался отчёт.

### Генетический алгоритм

| Параметр | Значение |
|---|---|
| `SEED` (по умолчанию) | 42 |
| `POPULATION_SIZE` | 50 |
| `GENERATIONS` | 10 |
| `GAMES_PER_AGENT` | 5 |
| `MAX_STEPS` | 1500 |

### DQN

| Параметр | Значение |
|---|---|
| `SEED` (по умолчанию) | 42 |
| `EPISODES` | 2000 |
| `MAX_STEPS` | 1500 |
| `LR` | 0.001 |
| `GAMMA` | 0.99 |
| `EPSILON_START → MIN` | 1.0 → 0.01 |
| `EPSILON_DECAY` | 0.998 |
| `MEMORY_SIZE` | 30000 |
| `BATCH_SIZE` | 128 |
| `TARGET_UPDATE` | 500 шагов |
| Loss | Huber (SmoothL1) |
| Gradient clipping | max_norm = 1.0 |

---

## Воспроизводимость и multi-seed

Каждый запуск фиксирует `random`, `numpy.random` и `torch.manual_seed`. При одном и том же seed результаты воспроизводимы между запусками на одной машине.

Для научной достоверности рекомендуется запускать **несколько прогонов с разными seed** и агрегировать результаты.

### Один прогон (как раньше)

```bash
python genetic/train.py
python dqn/train.py
```

### Серия прогонов с разными seed

```bash
# GA: три прогона подряд
python genetic/train.py --seeds 42,123,777

# DQN: три прогона подряд
python dqn/train.py --seeds 42,123,777
```

Каждый прогон создаёт отдельные артефакты:
- `genetic/logs/genetic_seed{N}.csv` — лог по поколениям
- `genetic/best_weights_seed{N}.json` — веса лучшего агента
- `genetic/training_plot_seed{N}.png` — график
- `dqn/logs/dqn_seed{N}.csv` — лог по эпизодам
- `dqn/dqn_model_seed{N}.pt` — модель
- `dqn/training_plot_seed{N}.png` — график

### Агрегация результатов

После серии прогонов запусти анализ:

```bash
python analyze.py
```

Скрипт:
- Читает все CSV-логи из `genetic/logs/` и `dqn/logs/`
- Строит сводные графики `mean ± std` по нескольким seed
- Выводит таблицу метрик: лучший счёт за прогон, mean/std/median, финальные значения
- Сохраняет графики в `analysis/`

---

## Как воспроизвести результаты отчёта

```bash
# 1. Установка
pip install -r requirements.txt
pip install torch

# 2. Серия прогонов (GA — ~1.5 часа, DQN — ~6-9 часов)
python genetic/train.py --seeds 42,123,777
python dqn/train.py --seeds 42,123,777

# 3. Агрегация и графики
python analyze.py

# 4. Демо лучшего агента
python play.py --mode genetic
python play.py --mode dqn
```

При фиксированных seed=42,123,777 результаты должны воспроизводиться (±машинная точность операций с плавающей точкой) между запусками на одном компьютере.

---

## Управление в ручном режиме

| Клавиша | Действие |
|---|---|
| `←` `→` | Двигать фигуру по горизонтали |
| `↑` | Поворот фигуры |
| `↓` или `Space` | Зафиксировать ход |
| `R` | Начать заново |
| `Esc` | Выход |

Жёлтое превью на поле показывает, куда упадёт фигура — позицию можно менять стрелками перед фиксацией.

---

## Как работает генетический алгоритм

Агент описывается 4 весами:
- `w_holes` — штраф за дыры
- `w_height` — штраф за высоту стопки
- `w_bumpiness` — штраф за неровность
- `w_lines` — награда за очищенные линии

Оценка хода: `score = w0·holes + w1·height + w2·bumpiness + w3·lines`

Цикл: популяция из 50 агентов играет → лучшие 20% (элита) переходят без изменений → остальные создаются скрещиванием + мутацией → новое поколение.

---

## Как работает DQN

Нейросеть `4 → 64 → 64 → 1` принимает признаки поля и предсказывает Q-значение (ожидаемая суммарная награда из этого состояния).

Ключевые механизмы:
- **Replay Memory** — кольцевой буфер опыта
- **Target Network** — отдельная копия сети для стабильных целевых Q
- **Epsilon-greedy** — баланс исследования и эксплуатации
- **Huber loss** — устойчивость к выбросам в target
- **Gradient clipping** (max_norm=1.0) — защита от взрыва градиентов

Целевая функция:
```
target = reward + γ · max_a' Q_target(s', a')
```
где max берётся по всем возможным следующим ходам через target network (исправление формулы относительно ранних версий проекта).

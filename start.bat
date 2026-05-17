@echo off
chcp 65001 > nul
cd /d "%~dp0"
setlocal enabledelayedexpansion

:menu
cls
echo ============================================================
echo       ТЕТРИС с машинным обучением - главное меню
echo ============================================================
echo.
echo   [1] Установить зависимости
echo.
echo   --- Обучение моделей ---
echo   [2] Обучить генетический алгоритм
echo   [3] Обучить DQN
echo.
echo   --- Демонстрация (нужна обученная модель) ---
echo   [4] Запустить генетический алгоритм (агент играет)
echo   [5] Запустить DQN (агент играет)
echo   [6] Ручная игра (управление человеком)
echo.
echo   [0] Выход
echo.
echo ============================================================
set /p ch="Выбери пункт меню: "

if "%ch%"=="1" goto install
if "%ch%"=="2" goto train_genetic
if "%ch%"=="3" goto train_dqn
if "%ch%"=="4" goto play_genetic
if "%ch%"=="5" goto play_dqn
if "%ch%"=="6" goto play_human
if "%ch%"=="0" goto fin
echo.
echo [!] Неверный выбор. Попробуй ещё раз.
pause
goto menu

:install
echo.
echo === Установка зависимостей ===
pip install -r requirements.txt
echo.
echo Готово! Не забудь установить PyTorch отдельно для DQN:
echo     pip install torch
echo.
pause
goto menu

:train_genetic
echo.
echo === Обучение генетического алгоритма ===
echo Параметры заданы в genetic/train.py
echo Время: ~25-50 минут
echo Прервать: Ctrl+C
echo.
python genetic/train.py
echo.
pause
goto menu

:train_dqn
echo.
echo === Обучение DQN ===
echo Параметры заданы в dqn/train.py
echo Время: 1-3 часа
echo Прервать: Ctrl+C
echo.
python dqn/train.py
echo.
pause
goto menu

:play_genetic
echo.
echo === Демонстрация: генетический алгоритм ===
if not exist "genetic\best_weights.json" (
    echo [!] Файл genetic\best_weights.json не найден!
    echo     Сначала обучи модель - пункт [2] в меню.
    echo.
    pause
    goto menu
)
python play.py --mode genetic
goto menu

:play_dqn
echo.
echo === Демонстрация: DQN ===
if not exist "dqn\dqn_model.pt" (
    echo [!] Файл dqn\dqn_model.pt не найден!
    echo     Сначала обучи модель - пункт [3] в меню.
    echo.
    pause
    goto menu
)
python play.py --mode dqn
goto menu

:play_human
echo.
echo === Ручная игра ===
python play.py --mode human
goto menu

:fin
echo.
echo До свидания!
exit /b 0
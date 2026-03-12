# 📋 ML TRADING BACKTESTER - ПОЛНЫЙ СПРАВОЧНИК КОМАНД

## 🔧 БЫСТРЫЙ СТАРТ

### Установка проекта
```bash
# 1. Клонирование репозитория
git clone https://github.com/StepanMelikyan/ml-trading-backtester.git
cd ml-trading-backtester

# 2. Создание виртуального окружения (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Настройка .env файла
copy .env.example .env
# Отредактируйте .env, добавьте свои данные MT5 и Telegram


# Быстрый тест (1 год данных)
python run_pipeline.py --symbol XAUUSD --years 1 --quick

# Полный анализ (5 лет)
python run_pipeline.py --symbol XAUUSD --years 5

# Анализ нефти
python run_pipeline.py --symbol Brent --years 5

# Для разных таймфреймов
python run_pipeline.py --symbol XAUUSD --timeframe H4 --years 3
python run_pipeline.py --symbol XAUUSD --timeframe M15 --years 1

# С отключенным кэшем
python run_pipeline.py --symbol XAUUSD --years 5 --no-cache

# Без LSTM моделей
python run_pipeline.py --symbol XAUUSD --years 5 --no-lstm








# Полный пайплайн (по умолчанию)
python main.py --mode full --symbol XAUUSD --years 5

# Только загрузка данных
python main.py --mode data --symbol XAUUSD

# Только расчет индикаторов
python main.py --mode features --symbol XAUUSD

# Только обучение модели
python main.py --mode train --symbol XAUUSD --years 3

# Только бэктестинг (с простой RSI стратегией)
python main.py --mode backtest --symbol XAUUSD

# Только генерация отчета из已有的 файлов
python main.py --mode report --symbol XAUUSD


🤖 ТОРГОВЫЕ БОТЫ
1. Простой сигнальный бот (trading_signal_bot.py)

# Однократная проверка
python trading_signal_bot.py --symbol XAUUSD --once

# Непрерывный мониторинг (каждый час)
python trading_signal_bot.py --symbol XAUUSD --interval 60

# Для нефти
python trading_signal_bot.py --symbol Brent --interval 60

# С реальными данными MT5
python trading_signal_bot.py --symbol XAUUSD --interval 60 --real-mt5


2. Внутридневной бот (intraday_signal_bot.py)
# Агрессивная торговля (низкий порог, частые сигналы)
python intraday_signal_bot.py --symbol XAUUSD --confidence 0.65 --interval 10

# Консервативная торговля (высокий порог, редкие сигналы)
python intraday_signal_bot.py --symbol XAUUSD --confidence 0.85 --interval 30

# Стандартный режим
python intraday_signal_bot.py --symbol XAUUSD --confidence 0.75 --interval 15

# Тестовый режим (без MT5, из файлов)
python intraday_signal_bot.py --symbol XAUUSD --test



3. ⭐ Многотаймфреймовый бот (dynamic_level_bot.py) - РЕКОМЕНДУЕТСЯ
# Стандартный запуск (H4 → H1 → M15)
python dynamic_level_bot.py --symbol XAUUSD --senior H4 --medium H1 --junior M15 --zone-width 0.3

# Для нефти
python dynamic_level_bot.py --symbol Brent --senior H4 --medium H1 --junior M15

# Узкая зона входа (более точные сигналы)
python dynamic_level_bot.py --symbol XAUUSD --zone-width 0.2 --confidence 0.8

# Широкая зона (больше сигналов)
python dynamic_level_bot.py --symbol XAUUSD --zone-width 0.5 --confidence 0.7

# С другими таймфреймами
python dynamic_level_bot.py --symbol XAUUSD --senior D1 --medium H4 --junior M30

# Тестовый режим
python dynamic_level_bot.py --symbol XAUUSD --test

📊 ОБУЧЕНИЕ МОДЕЛЕЙ ДЛЯ РАЗНЫХ ТАЙМФРЕЙМОВ
Создание данных и обучение
# Шаг 1: Создайте данные для нужного таймфрейма
python create_test_data.py        # для H1 (базовые)
python create_h4_data.py          # для H4
python create_m15_data.py         # для M15
python create_brent_data.py       # для нефти

# Шаг 2: Обучите модель
python run_pipeline.py --symbol XAUUSD_H4 --years 3 --timeframe H4
python run_pipeline.py --symbol XAUUSD_M15 --years 1 --timeframe M15
python run_pipeline.py --symbol Brent --years 5

# Шаг 3: Проверьте созданные модели
dir models\saved

🧪 ТЕСТИРОВАНИЕ
Запуск тестов
# Все тесты с подробным выводом
pytest tests/ -v

# Конкретные тесты
pytest tests/test_backtest.py -v
pytest tests/test_models.py -v
pytest tests/test_features.py -v
pytest tests/test_data.py -v
pytest tests/test_integration.py -v

# С покрытием кода
pytest tests/ --cov=. --cov-report=html

# Быстрый тест конкретного файла
pytest tests/test_backtest.py::TestBacktestEngine -v


📈 ПРИМЕРЫ ДЛЯ РАЗНЫХ СЦЕНАРИЕВ
Сценарий 1: Быстрый анализ золота
# 1. Быстрый бэктестинг
python run_pipeline.py --symbol XAUUSD --years 1 --quick

# 2. Проверка сигнала
python trading_signal_bot.py --symbol XAUUSD --once


Сценарий 2: Полноценная торговля нефтью
# 1. Обучить модель на 5 лет
python run_pipeline.py --symbol Brent --years 5

# 2. Запустить многотаймфреймового бота
python dynamic_level_bot.py --symbol Brent --senior H4 --medium H1 --junior M15


Сценарий 3: Тестирование новой стратегии
# 1. Быстрый тест
python run_pipeline.py --symbol XAUUSD --years 2 --quick --no-lstm

# 2. Посмотреть отчет
start reports\XAUUSD_report_*.html

# 3. Посмотреть графики
start reports\plots\XAUUSD_equity_curve.png


Сценарий 4: Минутная торговля
# 1. Обучить M15 модель
python create_m15_data.py
python run_pipeline.py --symbol XAUUSD_M15 --years 1 --timeframe M15

# 2. Запустить минутный анализ
python intraday_signal_bot.py --symbol XAUUSD --confidence 0.7 --interval 5



🛠 ВСПОМОГАТЕЛЬНЫЕ КОМАНДЫ
Работа с git
# Проверить статус
git status

# Добавить изменения
git add .
git commit -m "описание изменений"
git push

# Обновить репозиторий
git pull

# Посмотреть историю
git log --oneline



Работа с виртуальным окружением
# Активация (Windows)
venv\Scripts\activate

# Деактивация
deactivate

# Пересоздать окружение (если проблемы)
deactivate
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


Работа с зависимостями
# Обновить все пакеты
pip list --outdated
pip install --upgrade -r requirements.txt

# Заморозить текущие версии
pip freeze > requirements_fixed.txt


Работа с данными
# Очистить кэш
rmdir /s data\cache

# Посмотреть загруженные данные
dir data\*.csv

# Посмотреть последний отчет
dir reports\*.html /od

# Удалить все отчеты
del /q reports\*.html
del /q reports\plots\*.png


⚠️ ВАЖНЫЕ ЗАМЕЧАНИЯ
Перед первым запуском
Создайте файл .env из .env.example

Укажите реальные данные MT5 (логин, пароль, сервер)

Для Telegram укажите реальный Bot Token и Chat ID


Для работы MT5
Терминал MetaTrader 5 должен быть запущен

Должен быть открыт демо-счет

Проверьте подключение: python test_mt5_connection.py

Риск-менеджмент
Не рискуйте более 1-2% капитала на одну сделку

Всегда используйте стоп-лоссы

Тестируйте на демо-счете перед реальной торговлей


🆘 ПОЛУЧЕНИЕ ПОМОЩИ
# Справка по любому скрипту
python main.py --help
python run_pipeline.py --help
python dynamic_level_bot.py --help
python intraday_signal_bot.py --help


🚀 БЫСТРЫЕ КОМАНДЫ (CHEAT SHEET)
# Самые частые команды:

# 1. Быстрый тест золота
python run_pipeline.py --symbol XAUUSD --years 1 --quick

# 2. Запуск бота
python dynamic_level_bot.py --symbol XAUUSD

# 3. Проверка подключения MT5
python test_mt5_connection.py

# 4. Посмотреть отчет
start reports\XAUUSD_report_*.html

# 5. Очистить кэш
rmdir /s data\cache

Happy Trading! 🎯🚀	

### Как использовать


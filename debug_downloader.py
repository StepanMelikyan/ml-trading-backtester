# debug_downloader.py
from data.downloader import DataDownloader
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

print("=" * 60)
print("ДИАГНОСТИКА ЗАГРУЗЧИКА ДАННЫХ")
print("=" * 60)

# Создаем загрузчик
downloader = DataDownloader("XAUUSD", timeframe="H1", years=1, use_cache=False)

# Тест 1: Проверка подключения
print("\n1. ТЕСТ ПОДКЛЮЧЕНИЯ К MT5")
print("-" * 40)

if not mt5.initialize():
    print(f"❌ Ошибка инициализации: {mt5.last_error()}")
    mt5.shutdown()
    exit()

print("✅ MT5 инициализирован")

# Тест 2: Проверка символа
print("\n2. ПРОВЕРКА СИМВОЛА XAUUSD")
print("-" * 40)

symbol_info = mt5.symbol_info("XAUUSD")
if symbol_info:
    print(f"✅ Символ найден")
    print(f"   Имя: {symbol_info.name}")
    print(f"   Спред: {symbol_info.spread}")
    print(f"   Видим: {symbol_info.visible}")
    print(f"   Торговый режим: {symbol_info.trade_mode}")

    # Если символ не видим, делаем его видимым
    if not symbol_info.visible:
        mt5.symbol_select("XAUUSD", True)
        print("   Символ активирован")
else:
    print(f"❌ Символ не найден")

# Тест 3: Пробуем получить данные разными методами
print("\n3. ТЕСТ ЗАГРУЗКИ ДАННЫХ")
print("-" * 40)

# Метод 1: copy_rates_from_pos
print("\nМетод 1: copy_rates_from_pos (последние 100 свечей)")
rates1 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 100)
if rates1 is not None:
    print(f"✅ Успешно: {len(rates1)} записей")
    df1 = pd.DataFrame(rates1)
    df1['time'] = pd.to_datetime(df1['time'], unit='s')
    print(f"   Период: {df1['time'].min()} - {df1['time'].max()}")
else:
    print(f"❌ Ошибка: {mt5.last_error()}")

# Метод 2: copy_rates_range (последний год)
print("\nМетод 2: copy_rates_range (последний год)")
to_date = datetime.now()
from_date = to_date - timedelta(days=365)
rates2 = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1, from_date, to_date)
if rates2 is not None:
    print(f"✅ Успешно: {len(rates2)} записей")
    df2 = pd.DataFrame(rates2)
    df2['time'] = pd.to_datetime(df2['time'], unit='s')
    print(f"   Период: {df2['time'].min()} - {df2['time'].max()}")
else:
    print(f"❌ Ошибка: {mt5.last_error()}")

# Метод 3: copy_rates_from
print("\nМетод 3: copy_rates_from (от определенной даты)")
start_date = datetime.now() - timedelta(days=30)
rates3 = mt5.copy_rates_from("XAUUSD", mt5.TIMEFRAME_H1, start_date, 1000)
if rates3 is not None:
    print(f"✅ Успешно: {len(rates3)} записей")
else:
    print(f"❌ Ошибка: {mt5.last_error()}")

mt5.shutdown()

print("\n" + "=" * 60)
print("ДИАГНОСТИКА ЗАВЕРШЕНА")
print("=" * 60)
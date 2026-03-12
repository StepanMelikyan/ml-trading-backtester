# test_mt5_connection.py
import MetaTrader5 as mt5
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

print("=" * 50)
print("ПРОВЕРКА ПОДКЛЮЧЕНИЯ К MT5")
print("=" * 50)

# Данные из .env
login = int(os.getenv('MT5_LOGIN', 0))
password = os.getenv('MT5_PASSWORD', '')
server = os.getenv('MT5_SERVER', '')

print(f"Логин: {login}")
print(f"Сервер: {server}")
print(f"Пароль: {'*' * len(password) if password else 'не указан'}")

# Шаг 1: Инициализация
print("\n1. Инициализация MT5...")
if not mt5.initialize():
    print(f"   ❌ Ошибка: {mt5.last_error()}")
    mt5.shutdown()
    exit()

print("   ✅ MT5 инициализирован")

# Шаг 2: Проверка версии
version = mt5.version()
print(f"   Версия MT5: {version}")

# Шаг 3: Авторизация
print("\n2. Авторизация...")
if login and password and server:
    authorized = mt5.login(login, password, server)
    if authorized:
        print("   ✅ Авторизация успешна")

        # Информация о счете
        account_info = mt5.account_info()
        if account_info:
            print(f"   Счет: {account_info.login}")
            print(f"   Баланс: {account_info.balance} {account_info.currency}")
            print(f"   Сервер: {account_info.server}")
    else:
        error = mt5.last_error()
        print(f"   ❌ Ошибка авторизации: {error}")
else:
    print("   ⚠️ Данные авторизации неполные")

# Шаг 4: Проверка доступных символов
print("\n3. Проверка символов...")
symbols_to_check = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY"]

for symbol in symbols_to_check:
    info = mt5.symbol_info(symbol)
    if info is not None:
        print(f"   ✅ {symbol} - доступен (спред: {info.spread})")
    else:
        print(f"   ❌ {symbol} - недоступен")

# Шаг 5: Попытка загрузить данные
print("\n4. Загрузка тестовых данных XAUUSD...")
rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 10)
if rates is not None and len(rates) > 0:
    print(f"   ✅ Загружено {len(rates)} свечей")

    # Преобразуем в DataFrame для удобства
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    print("\n   Первые 3 свечи:")
    for i in range(min(3, len(df))):
        print(
            f"      Свеча {i + 1}: {df['time'].iloc[i]} - Open: {df['open'].iloc[i]:.2f}, Close: {df['close'].iloc[i]:.2f}, High: {df['high'].iloc[i]:.2f}, Low: {df['low'].iloc[i]:.2f}")
else:
    error = mt5.last_error()
    print(f"   ❌ Ошибка загрузки: {error}")

mt5.shutdown()
print("\n✅ Тест завершен")
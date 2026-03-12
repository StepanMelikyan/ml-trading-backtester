# create_test_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Создание тестовых данных для XAUUSD...")

# Создаем 8760 свечей (1 год часовых свечей)
dates = pd.date_range(end=datetime.now(), periods=8760, freq='H')

# Генерируем реалистичные данные для золота
np.random.seed(42)
base_price = 1900
trend = np.linspace(0, 200, 8760)  # восходящий тренд
cycle = 50 * np.sin(np.linspace(0, 20*np.pi, 8760))  # циклы
noise = np.random.randn(8760) * 10  # случайный шум

close = base_price + trend + cycle + noise

# Создаем OHLC данные
df = pd.DataFrame({
    'time': dates,
    'open': close - np.random.randn(8760) * 2,
    'high': close + np.abs(np.random.randn(8760) * 5) + 3,
    'low': close - np.abs(np.random.randn(8760) * 5) - 3,
    'close': close,
    'tick_volume': np.random.randint(100, 1000, 8760),
    'spread': np.random.randint(1, 10, 8760),
    'real_volume': np.random.randint(1000, 10000, 8760)
})

# Корректируем high/low чтобы они были корректными
df['high'] = df[['open', 'close', 'high']].max(axis=1)
df['low'] = df[['open', 'close', 'low']].min(axis=1)

# Сохраняем в файл
df.to_csv('data/XAUUSD_1years.csv', index=False)
print(f"✅ Тестовые данные созданы: {len(df)} записей")
print(f"   Период: {df['time'].min()} - {df['time'].max()}")
print(f"   Цена: {df['close'].min():.2f} - {df['close'].max():.2f}")
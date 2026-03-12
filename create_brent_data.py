# create_brent_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Создание тестовых данных для Brent Crude Oil...")

# Создаем 8760 свечей (1 год часовых свечей)
dates = pd.date_range(end=datetime.now(), periods=8760, freq='h')

# Генерируем цену нефти (обычно Brent торгуется в районе $70-90)
np.random.seed(43)  # другой seed для разнообразия
base_price = 75
trend = np.linspace(0, 15, 8760)  # восходящий тренд
cycle = 8 * np.sin(np.linspace(0, 20*np.pi, 8760))  # циклы
noise = np.random.randn(8760) * 2  # случайный шум

close = base_price + trend + cycle + noise

# Создаем OHLC данные
df = pd.DataFrame({
    'time': dates,
    'open': close - np.random.randn(8760) * 0.5,
    'high': close + np.abs(np.random.randn(8760) * 1.2) + 0.8,
    'low': close - np.abs(np.random.randn(8760) * 1.2) - 0.8,
    'close': close,
    'tick_volume': np.random.randint(1000, 10000, 8760),
    'spread': np.random.randint(1, 5, 8760),
    'real_volume': np.random.randint(10000, 100000, 8760)
})

# Корректируем high/low чтобы они были корректными
df['high'] = df[['open', 'close', 'high']].max(axis=1)
df['low'] = df[['open', 'close', 'low']].min(axis=1)

# Сохраняем в файл
df.to_csv('data/Brent_1years.csv', index=False)
print(f"✅ Тестовые данные для Brent созданы: {len(df)} записей")
print(f"   Период: {df['time'].min()} - {df['time'].max()}")
print(f"   Цена: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
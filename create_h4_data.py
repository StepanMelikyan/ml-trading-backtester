# create_h4_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Создание данных H4 для XAUUSD...")

# 3 года часовых данных = 26280 свечей
# H4 = 4 часа, значит делим на 4
periods = 8760 * 3 // 4  # примерно 6570 свечей H4 за 3 года

dates = pd.date_range(end=datetime.now(), periods=periods, freq='4H')

np.random.seed(45)
price = 1900 + np.cumsum(np.random.randn(periods) * 2)

df = pd.DataFrame({
    'time': dates,
    'open': price - np.random.randn(periods) * 1,
    'high': price + np.abs(np.random.randn(periods) * 3) + 2,
    'low': price - np.abs(np.random.randn(periods) * 3) - 2,
    'close': price,
    'tick_volume': np.random.randint(1000, 10000, periods)
})

df.to_csv('data/XAUUSD_H4_3years.csv', index=False)
print(f"✅ Данные H4 созданы: {len(df)} свечей")
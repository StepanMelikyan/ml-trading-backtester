# create_m15_data.py
import pandas as pd
import numpy as np
from datetime import datetime

# 1 год M15 данных
periods = 8760 * 4  # 4 свечи в час * 8760 часов

dates = pd.date_range(end=datetime.now(), periods=periods, freq='15min')

np.random.seed(46)
price = 1900 + np.cumsum(np.random.randn(periods) * 0.5)

df = pd.DataFrame({
    'time': dates,
    'open': price - np.random.randn(periods) * 0.2,
    'high': price + np.abs(np.random.randn(periods) * 0.8) + 0.5,
    'low': price - np.abs(np.random.randn(periods) * 0.8) - 0.5,
    'close': price,
    'tick_volume': np.random.randint(100, 1000, periods)
})

df.to_csv('data/XAUUSD_M15_1years.csv', index=False)
print(f"✅ Данные M15 созданы: {len(df)} свечей")
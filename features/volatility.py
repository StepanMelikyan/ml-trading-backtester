# features/volatility.py
"""
Индикаторы волатильности: ATR, Bollinger Bands, Keltner Channels.
"""

import pandas as pd
import numpy as np


class VolatilityIndicators:
    """
    Индикаторы для измерения и анализа волатильности.
    """

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет все индикаторы волатильности.
        """
        df = VolatilityIndicators.add_atr(df)
        df = VolatilityIndicators.add_bollinger_bands(df)
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average True Range - средний истинный диапазон.
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df['ATR'] = tr.rolling(period).mean()
        df['ATR_pct'] = (df['ATR'] / df['close']) * 100

        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands - полосы волатильности.
        """
        ma = df['close'].rolling(period).mean()
        bb_std = df['close'].rolling(period).std()

        df['BB_upper'] = ma + (bb_std * std_dev)
        df['BB_lower'] = ma - (bb_std * std_dev)
        df['BB_middle'] = ma
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / ma

        return df
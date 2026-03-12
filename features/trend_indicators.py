# features/trend_indicators.py
"""
Трендовые индикаторы: ADX, Parabolic SAR, Ichimoku Cloud.
"""

import pandas as pd
import numpy as np


class TrendIndicators:
    """
    Индикаторы для определения и оценки силы тренда.
    """

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет все трендовые индикаторы.
        """
        df = TrendIndicators.add_adx(df)
        df = TrendIndicators.add_parabolic_sar(df)
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index - измеряет силу тренда.
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['ADX'] = dx.rolling(period).mean()
        df['DI_plus'] = plus_di
        df['DI_minus'] = minus_di

        return df

    @staticmethod
    def add_parabolic_sar(df: pd.DataFrame, start: float = 0.02,
                          increment: float = 0.02, maximum: float = 0.2) -> pd.DataFrame:
        """
        Parabolic SAR - определяет потенциальные точки разворота.
        """
        high = df['high'].values
        low = df['low'].values

        length = len(df)
        psar = df['close'].copy()
        psar_bull = np.zeros(length)
        psar_bear = np.zeros(length)
        trend = np.ones(length)
        af = start * np.ones(length)

        if length == 0:
            return df

        # Bullish
        psar_bull[0] = low[0]
        ep_bull = high[0]

        # Bearish
        psar_bear[0] = high[0]
        ep_bear = low[0]

        for i in range(1, length):
            # Bullish
            if trend[i - 1] == 1:
                psar_bull[i] = psar_bull[i - 1] + af[i - 1] * (ep_bull - psar_bull[i - 1])

                if high[i] > ep_bull:
                    ep_bull = high[i]
                    af[i] = min(af[i - 1] + increment, maximum)
                else:
                    af[i] = af[i - 1]

                if low[i] < psar_bull[i]:
                    trend[i] = -1
                    psar_bear[i] = ep_bull
                    ep_bear = low[i]
                    af[i] = start
                else:
                    trend[i] = 1
                    psar_bear[i] = psar_bear[i - 1]

            # Bearish
            else:
                psar_bear[i] = psar_bear[i - 1] + af[i - 1] * (ep_bear - psar_bear[i - 1])

                if low[i] < ep_bear:
                    ep_bear = low[i]
                    af[i] = min(af[i - 1] + increment, maximum)
                else:
                    af[i] = af[i - 1]

                if high[i] > psar_bear[i]:
                    trend[i] = 1
                    psar_bull[i] = ep_bear
                    ep_bull = high[i]
                    af[i] = start
                else:
                    trend[i] = -1
                    psar_bull[i] = psar_bull[i - 1]

        df['PSAR'] = np.where(trend == 1, psar_bull, psar_bear)
        df['PSAR_trend'] = trend

        return df
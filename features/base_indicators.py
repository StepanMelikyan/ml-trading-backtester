# features/base_indicators.py
"""
Базовые технические индикаторы: скользящие средние, RSI, MACD.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.decorators import timer
from utils.logger import log


class BaseIndicators:
    """
    Класс для расчета базовых технических индикаторов.
    Все методы статические для легкого тестирования.
    """

    @staticmethod
    @timer
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет все базовые индикаторы.

        Args:
            df: DataFrame с колонками open, high, low, close, volume

        Returns:
            DataFrame с добавленными индикаторами
        """
        df = BaseIndicators.add_sma(df)
        df = BaseIndicators.add_ema(df)
        df = BaseIndicators.add_rsi(df)
        df = BaseIndicators.add_macd(df)

        log.info(f"📊 Добавлены базовые индикаторы")
        return df

    @staticmethod
    def add_sma(df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Simple Moving Average (простое скользящее среднее).

        Args:
            df: DataFrame с колонкой 'close'
            windows: список периодов SMA

        Returns:
            DataFrame с колонками SMA_{window}
        """
        if windows is None:
            windows = [10, 20, 50, 200]  # Значения по умолчанию

        for window in windows:
            df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()

            # Наклон SMA (важный признак тренда)
            df[f'SMA_{window}_slope'] = df[f'SMA_{window}'].diff(5)

            # Расстояние цены от SMA
            df[f'price_to_SMA_{window}'] = (df['close'] - df[f'SMA_{window}']) / df[f'SMA_{window}']

            # Сигнал пересечения
            df[f'SMA_{window}_cross'] = np.where(
                (df['close'] > df[f'SMA_{window}']) &
                (df['close'].shift(1) <= df[f'SMA_{window}'].shift(1)), 1,
                np.where((df['close'] < df[f'SMA_{window}']) &
                         (df['close'].shift(1) >= df[f'SMA_{window}'].shift(1)), -1, 0)
            )

        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Exponential Moving Average (экспоненциальное скользящее среднее).

        Args:
            df: DataFrame с колонкой 'close'
            windows: список периодов EMA

        Returns:
            DataFrame с колонками EMA_{window}
        """
        if windows is None:
            windows = [12, 26, 50]  # Значения по умолчанию

        for window in windows:
            df[f'EMA_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

            # Наклон EMA
            df[f'EMA_{window}_slope'] = df[f'EMA_{window}'].diff(3)

        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Relative Strength Index (индекс относительной силы).

        Args:
            df: DataFrame с колонкой 'close'
            periods: список периодов RSI

        Returns:
            DataFrame с колонками RSI_{period}
        """
        if periods is None:
            periods = [14]  # Значения по умолчанию

        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            # Избегаем деления на ноль
            rs = gain / (loss + 1e-10)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

            # Сигналы перекупленности/перепроданности
            df[f'RSI_{period}_overbought'] = (df[f'RSI_{period}'] > 70).astype(int)
            df[f'RSI_{period}_oversold'] = (df[f'RSI_{period}'] < 30).astype(int)

        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame с колонкой 'close'
            fast: быстрый период
            slow: медленный период
            signal: период сигнальной линии

        Returns:
            DataFrame с колонками MACD, MACD_signal, MACD_histogram
        """
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()

        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # Сигналы пересечения
        df['MACD_cross'] = np.where(
            (df['MACD'] > df['MACD_signal']) &
            (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 1,
            np.where((df['MACD'] < df['MACD_signal']) &
                     (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), -1, 0)
        )

        return df

    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Williams %R.

        Args:
            df: DataFrame с колонками high, low, close
            period: период расчета

        Returns:
            DataFrame с колонкой Williams_R
        """
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()

        df['Williams_R'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low + 1e-10))

        # Сигналы
        df['Williams_overbought'] = (df['Williams_R'] > -20).astype(int)
        df['Williams_oversold'] = (df['Williams_R'] < -80).astype(int)

        return df

    @staticmethod
    def add_ppo(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
        """
        Percentage Price Oscillator.

        Args:
            df: DataFrame с колонкой 'close'
            fast: быстрый период
            slow: медленный период

        Returns:
            DataFrame с колонкой PPO
        """
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()

        df['PPO'] = ((exp1 - exp2) / exp2) * 100

        return df
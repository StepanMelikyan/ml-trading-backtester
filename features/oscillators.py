# features/oscillators.py
"""
Осцилляторы: Stochastic, CCI, Williams %R, ROC, RVI.
"""

import pandas as pd
import numpy as np
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.indicators_config import INDICATORS_CONFIG
from utils.decorators import timer
from utils.logger import log


class OscillatorIndicators:
    """
    Класс для расчета осцилляторов - индикаторов перекупленности/перепроданности.
    """

    @staticmethod
    @timer
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет все осцилляторы.

        Args:
            df: DataFrame с колонками open, high, low, close

        Returns:
            DataFrame с добавленными осцилляторами
        """
        df = OscillatorIndicators.add_stochastic(df)
        df = OscillatorIndicators.add_cci(df)
        df = OscillatorIndicators.add_williams_r(df)
        df = OscillatorIndicators.add_roc(df)
        df = OscillatorIndicators.add_rvi(df)
        df = OscillatorIndicators.add_ultimate_oscillator(df)

        log.info(f"📊 Добавлены осцилляторы")
        return df

    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14,
                       d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator - определение перекупленности/перепроданности.

        Args:
            df: DataFrame с колонками high, low, close
            k_period: период %K
            d_period: период %D
            slowing: сглаживание

        Returns:
            DataFrame с колонками Stoch_K, Stoch_D
        """
        # %K
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()

        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))

        # Slow %K (сглаживание)
        stoch_k_slow = stoch_k.rolling(slowing).mean()

        # %D (сигнальная линия)
        stoch_d = stoch_k_slow.rolling(d_period).mean()

        df['Stoch_K'] = stoch_k
        df['Stoch_K_slow'] = stoch_k_slow
        df['Stoch_D'] = stoch_d

        # Сигналы перекупленности/перепроданности
        df['Stoch_overbought'] = (df['Stoch_K'] > 80).astype(int)
        df['Stoch_oversold'] = (df['Stoch_K'] < 20).astype(int)

        # Пересечения
        df['Stoch_cross_up'] = ((df['Stoch_K'] > df['Stoch_D']) &
                                (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1))).astype(int)
        df['Stoch_cross_down'] = ((df['Stoch_K'] < df['Stoch_D']) &
                                  (df['Stoch_K'].shift(1) >= df['Stoch_D'].shift(1))).astype(int)

        return df

    @staticmethod
    def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Commodity Channel Index - измеряет отклонение от среднего.

        Args:
            df: DataFrame с колонками high, low, close
            period: период расчета

        Returns:
            DataFrame с колонкой CCI
        """
        # Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3

        # SMA of TP
        sma_tp = tp.rolling(period).mean()

        # Mean Deviation
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

        # CCI
        df['CCI'] = (tp - sma_tp) / (0.015 * md + 1e-10)

        # Сигналы
        df['CCI_overbought'] = (df['CCI'] > 100).astype(int)
        df['CCI_oversold'] = (df['CCI'] < -100).astype(int)

        # Возврат из зон
        df['CCI_exit_overbought'] = ((df['CCI'] < 100) & (df['CCI'].shift(1) >= 100)).astype(int)
        df['CCI_exit_oversold'] = ((df['CCI'] > -100) & (df['CCI'].shift(1) <= -100)).astype(int)

        # Дивергенции (упрощенно)
        df['CCI_div_pos'] = ((df['close'] > df['close'].shift(5)) &
                             (df['CCI'] < df['CCI'].shift(5))).astype(int)
        df['CCI_div_neg'] = ((df['close'] < df['close'].shift(5)) &
                             (df['CCI'] > df['CCI'].shift(5))).astype(int)

        return df

    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Williams %R - аналогичен Stochastic, но инвертирован.

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
    def add_roc(df: pd.DataFrame, periods: list = [5, 10, 20]) -> pd.DataFrame:
        """
        Rate of Change - скорость изменения цены.

        Args:
            df: DataFrame с колонкой 'close'
            periods: список периодов

        Returns:
            DataFrame с колонками ROC_{period}
        """
        for period in periods:
            df[f'ROC_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

            # Сглаженный ROC
            df[f'ROC_ma_{period}'] = df[f'ROC_{period}'].rolling(5).mean()

        return df

    @staticmethod
    def add_rvi(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        Relative Volatility Index - волатильность относительно цены.

        Args:
            df: DataFrame с колонками high, low, close
            period: период расчета

        Returns:
            DataFrame с колонкой RVI
        """
        # Стандартное отклонение доходностей
        returns = df['close'].pct_change()
        std_dev = returns.rolling(period).std()

        # RVI
        df['RVI'] = 100 * (std_dev / std_dev.rolling(period * 2).max())

        return df

    @staticmethod
    def add_ultimate_oscillator(df: pd.DataFrame,
                                periods: list = [7, 14, 28]) -> pd.DataFrame:
        """
        Ultimate Oscillator - комбинация нескольких периодов.

        Args:
            df: DataFrame с колонками high, low, close
            periods: список периодов (обычно 7, 14, 28)

        Returns:
            DataFrame с колонкой Ultimate_Osc
        """
        # Buying Pressure
        bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))

        # True Range
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # Average for each period
        avg7 = bp.rolling(periods[0]).sum() / (tr.rolling(periods[0]).sum() + 1e-10)
        avg14 = bp.rolling(periods[1]).sum() / (tr.rolling(periods[1]).sum() + 1e-10)
        avg28 = bp.rolling(periods[2]).sum() / (tr.rolling(periods[2]).sum() + 1e-10)

        # Ultimate Oscillator
        df['Ultimate_Osc'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)

        # Сигналы
        df['Ultimate_overbought'] = (df['Ultimate_Osc'] > 70).astype(int)
        df['Ultimate_oversold'] = (df['Ultimate_Osc'] < 30).astype(int)

        return df

    @staticmethod
    def add_dpo(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Detrended Price Oscillator - удаляет тренд для выявления циклов.

        Args:
            df: DataFrame с колонкой 'close'
            period: период расчета

        Returns:
            DataFrame с колонкой DPO
        """
        # Смещенное скользящее среднее
        shift = period // 2 + 1
        sma = df['close'].rolling(period).mean()

        df['DPO'] = df['close'] - sma.shift(shift)

        return df
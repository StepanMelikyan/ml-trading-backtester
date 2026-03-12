# features/price_patterns.py
"""
Обнаружение ценовых паттернов: свечные паттерны, фракталы, уровни поддержки/сопротивления.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.indicators_config import INDICATORS_CONFIG
from utils.decorators import timer
from utils.logger import log


class PricePatterns:
    """
    Класс для обнаружения ценовых паттернов и графических формаций.
    """

    @staticmethod
    @timer
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет все паттерны.

        Args:
            df: DataFrame с колонками open, high, low, close

        Returns:
            DataFrame с колонками паттернов
        """
        df = PricePatterns.add_candle_patterns(df)
        df = PricePatterns.add_fractals(df)
        df = PricePatterns.add_support_resistance(df)
        df = PricePatterns.add_pivot_points(df)

        log.info(f"📊 Добавлены ценовые паттерны")
        return df

    @staticmethod
    def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение свечных паттернов.

        Args:
            df: DataFrame с колонками open, high, low, close

        Returns:
            DataFrame с колонками паттернов (0/1)
        """
        # Размеры свечи
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        total_range = df['high'] - df['low']

        # Doji (тело очень маленькое)
        df['Doji'] = (body / (total_range + 1e-10) < 0.1).astype(int)

        # Hammer (молот) - длинная нижняя тень, маленькое тело вверху
        df['Hammer'] = ((lower_shadow > body * 2) &
                        (upper_shadow < body) &
                        (body / (total_range + 1e-10) < 0.3) &
                        (df['close'] > df['open'])).astype(int)

        # Shooting Star (падающая звезда) - длинная верхняя тень
        df['ShootingStar'] = ((upper_shadow > body * 2) &
                              (lower_shadow < body) &
                              (body / (total_range + 1e-10) < 0.3) &
                              (df['close'] < df['open'])).astype(int)

        # Bullish Engulfing (бычье поглощение)
        df['BullishEngulfing'] = ((df['close'] > df['open']) &
                                  (df['open'] < df['close'].shift(1)) &
                                  (df['close'] > df['open'].shift(1)) &
                                  (df['open'].shift(1) > df['close'].shift(1))).astype(int)

        # Bearish Engulfing (медвежье поглощение)
        df['BearishEngulfing'] = ((df['close'] < df['open']) &
                                  (df['open'] > df['close'].shift(1)) &
                                  (df['close'] < df['open'].shift(1)) &
                                  (df['open'].shift(1) < df['close'].shift(1))).astype(int)

        # Morning Star (утренняя звезда) - упрощенно
        df['MorningStar'] = ((df['close'].shift(2) < df['open'].shift(2)) &  # Первая медвежья
                             (abs(df['close'].shift(1) - df['open'].shift(1)) < body.shift(1) * 0.3) &  # Вторая дожи
                             (df['close'] > df['open']) &  # Третья бычья
                             (df['close'] > (df['close'].shift(2) + df['open'].shift(2)) / 2)).astype(int)

        # Evening Star (вечерняя звезда)
        df['EveningStar'] = ((df['close'].shift(2) > df['open'].shift(2)) &
                             (abs(df['close'].shift(1) - df['open'].shift(1)) < body.shift(1) * 0.3) &
                             (df['close'] < df['open']) &
                             (df['close'] < (df['close'].shift(2) + df['open'].shift(2)) / 2)).astype(int)

        # Three White Soldiers (три белых солдата)
        df['ThreeWhiteSoldiers'] = ((df['close'] > df['open']) &
                                    (df['close'].shift(1) > df['open'].shift(1)) &
                                    (df['close'].shift(2) > df['open'].shift(2)) &
                                    (df['close'] > df['close'].shift(1)) &
                                    (df['close'].shift(1) > df['close'].shift(2))).astype(int)

        # Three Black Crows (три черных вороны)
        df['ThreeBlackCrows'] = ((df['close'] < df['open']) &
                                 (df['close'].shift(1) < df['open'].shift(1)) &
                                 (df['close'].shift(2) < df['open'].shift(2)) &
                                 (df['close'] < df['close'].shift(1)) &
                                 (df['close'].shift(1) < df['close'].shift(2))).astype(int)

        # Harami (беременность)
        df['BullishHarami'] = ((df['open'] < df['close']) &
                               (df['open'].shift(1) > df['close'].shift(1)) &
                               (df['open'] > df['close'].shift(1)) &
                               (df['close'] < df['open'].shift(1))).astype(int)

        df['BearishHarami'] = ((df['open'] > df['close']) &
                               (df['open'].shift(1) < df['close'].shift(1)) &
                               (df['open'] < df['close'].shift(1)) &
                               (df['close'] > df['open'].shift(1))).astype(int)

        return df

    @staticmethod
    def add_fractals(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """
        Фракталы Билла Вильямса - локальные максимумы/минимумы.

        Args:
            df: DataFrame с колонками high, low
            period: период для поиска фракталов (должен быть нечетным)

        Returns:
            DataFrame с колонками FractalUp, FractalDown
        """
        df['FractalUp'] = 0
        df['FractalDown'] = 0

        half = period // 2

        for i in range(half, len(df) - half):
            # Верхний фрактал (максимум)
            is_fractal_up = True
            for j in range(1, half + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or \
                        df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_fractal_up = False
                    break

            if is_fractal_up:
                df.loc[df.index[i], 'FractalUp'] = 1

            # Нижний фрактал (минимум)
            is_fractal_down = True
            for j in range(1, half + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or \
                        df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_fractal_down = False
                    break

            if is_fractal_down:
                df.loc[df.index[i], 'FractalDown'] = 1

        # Расстояние до последнего фрактала
        df['DistToFractalUp'] = 0
        df['DistToFractalDown'] = 0

        last_up = -1
        last_down = -1

        for i in range(len(df)):
            if df['FractalUp'].iloc[i] == 1:
                last_up = i
            if last_up != -1:
                df.loc[df.index[i], 'DistToFractalUp'] = i - last_up

            if df['FractalDown'].iloc[i] == 1:
                last_down = i
            if last_down != -1:
                df.loc[df.index[i], 'DistToFractalDown'] = i - last_down

        return df

    @staticmethod
    def add_support_resistance(df: pd.DataFrame, window: int = 20,
                               num_levels: int = 5) -> pd.DataFrame:
        """
        Определение уровней поддержки и сопротивления.

        Args:
            df: DataFrame с колонками high, low, close
            window: окно для поиска локальных экстремумов
            num_levels: количество уровней

        Returns:
            DataFrame с информацией об уровнях
        """
        # Локальные минимумы (поддержка)
        df['Support'] = df['low'].rolling(window, center=True).min()
        df['IsSupport'] = (df['low'] == df['Support']).astype(int)

        # Локальные максимумы (сопротивление)
        df['Resistance'] = df['high'].rolling(window, center=True).max()
        df['IsResistance'] = (df['high'] == df['Resistance']).astype(int)

        # Расстояние до ближайших уровней
        df['DistToSupport'] = df['close'] - df['Support']
        df['DistToResistance'] = df['Resistance'] - df['close']

        # Относительное расстояние
        df['SupportDistancePct'] = df['DistToSupport'] / df['close'] * 100
        df['ResistanceDistancePct'] = df['DistToResistance'] / df['close'] * 100

        # Пробой уровней
        df['SupportBreak'] = (df['close'] < df['Support'].shift(1)).astype(int)
        df['ResistanceBreak'] = (df['close'] > df['Resistance'].shift(1)).astype(int)

        # Тест уровней (подход цены к уровню)
        df['TestingSupport'] = (abs(df['DistToSupport']) / df['close'] * 100 < 0.5).astype(int)
        df['TestingResistance'] = (abs(df['DistToResistance']) / df['close'] * 100 < 0.5).astype(int)

        return df

    @staticmethod
    def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
        """
        Классические пивот уровни.

        Args:
            df: DataFrame с колонками high, low, close

        Returns:
            DataFrame с колонками Pivot, R1, R2, R3, S1, S2, S3
        """
        # Получаем данные предыдущего дня
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)

        # Pivot Point
        df['Pivot'] = (df['prev_high'] + df['prev_low'] + df['prev_close']) / 3

        # Сопротивления
        df['R1'] = 2 * df['Pivot'] - df['prev_low']
        df['R2'] = df['Pivot'] + (df['prev_high'] - df['prev_low'])
        df['R3'] = df['prev_high'] + 2 * (df['Pivot'] - df['prev_low'])

        # Поддержки
        df['S1'] = 2 * df['Pivot'] - df['prev_high']
        df['S2'] = df['Pivot'] - (df['prev_high'] - df['prev_low'])
        df['S3'] = df['prev_low'] - 2 * (df['prev_high'] - df['Pivot'])

        # Положение цены относительно пивотов
        df['PriceToPivot'] = (df['close'] - df['Pivot']) / df['Pivot'] * 100

        return df

    @staticmethod
    def add_pattern_strength(df: pd.DataFrame) -> pd.DataFrame:
        """
        Общая сила паттернов (сумма всех бычьих и медвежьих паттернов).

        Args:
            df: DataFrame с колонками паттернов

        Returns:
            DataFrame с колонками BullishPatterns, BearishPatterns
        """
        # Список бычьих паттернов
        bullish_patterns = ['Hammer', 'BullishEngulfing', 'MorningStar',
                            'ThreeWhiteSoldiers', 'BullishHarami']

        # Список медвежьих паттернов
        bearish_patterns = ['ShootingStar', 'BearishEngulfing', 'EveningStar',
                            'ThreeBlackCrows', 'BearishHarami']

        # Считаем только существующие колонки
        existing_bullish = [p for p in bullish_patterns if p in df.columns]
        existing_bearish = [p for p in bearish_patterns if p in df.columns]

        if existing_bullish:
            df['BullishPatterns'] = df[existing_bullish].sum(axis=1)

        if existing_bearish:
            df['BearishPatterns'] = df[existing_bearish].sum(axis=1)

        return df
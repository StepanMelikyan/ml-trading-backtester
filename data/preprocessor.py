# data/preprocessor.py
"""
Модуль для очистки, валидации и предобработки данных.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log
from utils.decorators import timer


class DataPreprocessor:
    """
    Класс для предобработки и очистки данных перед анализом.
    """

    @timer
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Полная очистка данных от аномалий и пропусков.

        Args:
            df: Исходный DataFrame

        Returns:
            Очищенный DataFrame
        """
        df = df.copy()
        original_len = len(df)

        # Удаляем дубликаты
        df = df.drop_duplicates(subset=['time'])

        # Сортируем по времени
        df = df.sort_values('time')

        # Проверяем на пропуски во временном ряду
        df = self._fill_missing_candles(df)

        # Удаляем строки с нулевой ценой
        df = df[(df['open'] > 0) & (df['high'] > 0) &
                (df['low'] > 0) & (df['close'] > 0)]

        # Проверяем корректность OHLC
        df = df[(df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])]

        # Обработка выбросов
        for col in ['open', 'high', 'low', 'close']:
            df[col] = self._remove_outliers(df[col])

        # Сброс индекса
        df = df.reset_index(drop=True)

        removed = original_len - len(df)
        if removed > 0:
            log.info(f"🧹 Удалено {removed} некорректных записей")

        return df

    def _fill_missing_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заполняет пропущенные свечи интерполяцией.
        """
        if len(df) < 2:
            return df

        # Определяем ожидаемый интервал (медианный)
        time_diff = df['time'].diff().median()

        # Находим пропуски
        df = df.set_index('time')
        expected_freq = self._infer_freq(time_diff)

        # Создаем полный индекс
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)

        # Реиндексируем и интерполируем
        df = df.reindex(full_index)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')

        df = df.reset_index().rename(columns={'index': 'time'})

        return df

    def _infer_freq(self, time_diff: pd.Timedelta) -> str:
        """Определяет частоту временного ряда"""
        if time_diff <= pd.Timedelta(minutes=1):
            return '1min'
        elif time_diff <= pd.Timedelta(minutes=5):
            return '5min'
        elif time_diff <= pd.Timedelta(minutes=15):
            return '15min'
        elif time_diff <= pd.Timedelta(minutes=30):
            return '30min'
        elif time_diff <= pd.Timedelta(hours=1):
            return '1H'
        elif time_diff <= pd.Timedelta(hours=4):
            return '4H'
        else:
            return '1D'

    def _remove_outliers(self, series: pd.Series, threshold: float = 3) -> pd.Series:
        """
        Удаляет выбросы по z-score (заменяет на скользящее среднее).
        """
        z_scores = np.abs((series - series.mean()) / series.std())
        mask = z_scores < threshold

        # Заменяем выбросы на скользящее среднее
        result = series.where(mask, series.rolling(20, min_periods=1).mean())

        return result

    @timer
    def add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет временные признаки (час, день недели, месяц и т.д.).

        Args:
            df: DataFrame с колонкой 'time'

        Returns:
            DataFrame с дополнительными колонками
        """
        df = df.copy()

        # Убеждаемся, что time в правильном формате
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        df = df.set_index('time')

        # Базовые временные признаки
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Циклические признаки для периодичности
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Признаки торговых сессий
        df['is_asia'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
        df['is_london'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
        df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 17)).astype(int)

        df = df.reset_index()

        log.info(
            f"📅 Добавлено {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} временных признаков")

        return df

    def normalize(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Нормализует указанные колонки (Z-score).

        Args:
            df: Исходный DataFrame
            columns: Список колонок для нормализации

        Returns:
            DataFrame с добавленными нормализованными колонками
        """
        df = df.copy()

        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f'{col}_norm'] = (df[col] - mean) / std
                    log.debug(f"📊 Нормализована колонка: {col}")

        return df

    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.6,
                   val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделяет данные на train/val/test с сохранением временного порядка.

        Args:
            df: DataFrame с индексом по времени
            train_ratio: доля обучающей выборки
            val_ratio: доля валидационной выборки

        Returns:
            (train_df, val_df, test_df)
        """
        df = df.copy()

        # Убеждаемся, что индекс - время
        if 'time' in df.columns:
            df = df.set_index('time')

        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))

        train = df.iloc[:train_idx]
        val = df.iloc[train_idx:val_idx]
        test = df.iloc[val_idx:]

        log.info(f"📊 Разделение данных:")
        log.info(f"  Train: {len(train)} записей ({train.index[0]} - {train.index[-1]})")
        log.info(f"  Val:   {len(val)} записей ({val.index[0]} - {val.index[-1]})")
        log.info(f"  Test:  {len(test)} записей ({test.index[0]} - {test.index[-1]})")

        return train.reset_index(), val.reset_index(), test.reset_index()
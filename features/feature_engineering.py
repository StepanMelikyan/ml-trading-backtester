# features/feature_engineering.py
"""
Создание признаков для машинного обучения: лаги, скользящие окна, целевые переменные.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.decorators import timer
from utils.logger import log


class FeatureEngineering:
    """
    Класс для создания признаков из временных рядов.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Инициализация с DataFrame.

        Args:
            df: исходный DataFrame
        """
        self.df = df.copy()
        self.original_columns = df.columns.tolist()

    @timer
    def create_target(self, horizon: int = 5, target_type: str = 'direction') -> pd.DataFrame:
        """
        Создает целевую переменную для прогнозирования.

        Args:
            horizon: горизонт прогноза (количество свечей вперед)
            target_type: тип цели
                - 'direction': бинарная (1 - рост, 0 - падение)
                - 'return': процентное изменение
                - 'classification': мультиклассовая (2 - сильный рост, 1 - рост, 0 - падение, -1 - сильное падение)

        Returns:
            DataFrame с добавленной колонкой 'target'
        """
        if target_type == 'direction':
            # Бинарная классификация: цена вырастет (1) или упадет (0)
            future_price = self.df['close'].shift(-horizon)
            self.df['target'] = (future_price > self.df['close']).astype(int)

            log.info(f"🎯 Создана бинарная цель (горизонт={horizon})")

        elif target_type == 'return':
            # Регрессия: процентное изменение
            self.df['target'] = (self.df['close'].shift(-horizon) / self.df['close'] - 1) * 100
            log.info(f"🎯 Создана регрессионная цель (горизонт={horizon})")

        elif target_type == 'classification':
            # Мультиклассовая классификация
            future_return = (self.df['close'].shift(-horizon) / self.df['close'] - 1) * 100

            conditions = [
                future_return > 2,  # сильный рост
                (future_return > 0.5) & (future_return <= 2),  # умеренный рост
                (future_return >= -0.5) & (future_return <= 0.5),  # боковик
                (future_return < -0.5) & (future_return >= -2),  # умеренное падение
                future_return < -2  # сильное падение
            ]
            choices = [2, 1, 0, -1, -2]

            self.df['target'] = np.select(conditions, choices, default=0)
            log.info(f"🎯 Создана мультиклассовая цель (горизонт={horizon})")

        return self.df

    @timer
    def create_lag_features(self, columns: List[str], lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """
        Создает лаговые признаки для указанных колонок.

        Args:
            columns: список колонок для создания лагов
            lags: список значений лагов

        Returns:
            DataFrame с добавленными лаговыми признаками
        """
        for col in columns:
            if col not in self.df.columns:
                log.warning(f"⚠️ Колонка {col} не найдена, пропускаем")
                continue

            for lag in lags:
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)

        log.info(f"⏳ Создано {len(columns) * len(lags)} лаговых признаков")
        return self.df

    @timer
    def create_rolling_features(self, columns: List[str],
                                windows: List[int] = [5, 10, 20, 50],
                                functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Создает скользящие статистики.

        Args:
            columns: список колонок
            windows: список окон
            functions: список функций ('mean', 'std', 'min', 'max', 'skew', 'kurt')

        Returns:
            DataFrame с добавленными признаками
        """
        func_map = {
            'mean': lambda x, w: x.rolling(w).mean(),
            'std': lambda x, w: x.rolling(w).std(),
            'min': lambda x, w: x.rolling(w).min(),
            'max': lambda x, w: x.rolling(w).max(),
            'skew': lambda x, w: x.rolling(w).skew(),
            'kurt': lambda x, w: x.rolling(w).kurt()
        }

        count = 0
        for col in columns:
            if col not in self.df.columns:
                continue

            for window in windows:
                for func_name in functions:
                    if func_name in func_map:
                        new_col = f'{col}_{func_name}_{window}'
                        self.df[new_col] = func_map[func_name](self.df[col], window)
                        count += 1

        log.info(f"📊 Создано {count} скользящих признаков")
        return self.df

    @timer
    def create_ratio_features(self, pairs: List[tuple]) -> pd.DataFrame:
        """
        Создает признаки-отношения между колонками.

        Args:
            pairs: список пар колонок [(col1, col2), ...]

        Returns:
            DataFrame с добавленными отношениями
        """
        for col1, col2 in pairs:
            if col1 in self.df.columns and col2 in self.df.columns:
                self.df[f'{col1}_{col2}_ratio'] = self.df[col1] / (self.df[col2] + 1e-10)
                self.df[f'{col1}_{col2}_diff'] = self.df[col1] - self.df[col2]

        return self.df

    @timer
    def create_interaction_features(self, columns: List[str], max_interactions: int = 10) -> pd.DataFrame:
        """
        Создает признаки взаимодействия (произведения) для топ-N важных колонок.

        Args:
            columns: список колонок
            max_interactions: максимальное количество взаимодействий

        Returns:
            DataFrame с добавленными признаками
        """
        # Берем только числовые колонки
        numeric_cols = [c for c in columns if c in self.df.columns and
                        pd.api.types.is_numeric_dtype(self.df[c])]

        if len(numeric_cols) < 2:
            return self.df

        # Создаем взаимодействия для первых max_interactions колонок
        import itertools
        count = 0
        for col1, col2 in itertools.combinations(numeric_cols[:min(len(numeric_cols), 10)], 2):
            if count >= max_interactions:
                break
            self.df[f'{col1}_x_{col2}'] = self.df[col1] * self.df[col2]
            count += 1

        return self.df

    @timer
    def create_statistical_features(self, window: int = 20) -> pd.DataFrame:
        """
        Создает статистические признаки для цены.

        Args:
            window: окно для расчета

        Returns:
            DataFrame с добавленными признаками
        """
        # Z-score цены
        price_mean = self.df['close'].rolling(window).mean()
        price_std = self.df['close'].rolling(window).std()
        self.df['price_zscore'] = (self.df['close'] - price_mean) / (price_std + 1e-10)

        # Процентили
        self.df['price_percentile'] = self.df['close'].rolling(window).apply(
            lambda x: (x.iloc[-1] < x).sum() / len(x) * 100
        )

        # Коэффициент вариации
        self.df['price_cv'] = price_std / (price_mean + 1e-10)

        return self.df

    @timer
    def create_volatility_features(self) -> pd.DataFrame:
        """
        Создает признаки волатильности на основе доходностей.

        Returns:
            DataFrame с добавленными признаками
        """
        # Доходности
        self.df['returns'] = self.df['close'].pct_change()
        self.df['log_returns'] = np.log(self.df['close'] / self.df['close'].shift(1))

        # Реализованная волатильность
        for period in [5, 10, 20]:
            self.df[f'volatility_{period}'] = self.df['returns'].rolling(period).std() * np.sqrt(252) * 100

        # Амплитуда свечи
        self.df['candle_range'] = (self.df['high'] - self.df['low']) / self.df['close'] * 100

        # GAPы
        self.df['gap'] = (self.df['open'] - self.df['close'].shift(1)) / self.df['close'].shift(1) * 100

        return self.df

    def clean_data(self, drop_na: bool = True) -> pd.DataFrame:
        """
        Очищает данные от NaN и бесконечных значений.

        Args:
            drop_na: удалять ли строки с NaN

        Returns:
            Очищенный DataFrame
        """
        # Заменяем бесконечности на NaN
        self.df = self.df.replace([np.inf, -np.inf], np.nan)

        if drop_na:
            before = len(self.df)
            self.df = self.df.dropna()
            after = len(self.df)
            if before - after > 0:
                log.info(f"🧹 Удалено {before - after} строк с NaN")

        return self.df

    def get_feature_names(self) -> List[str]:
        """
        Возвращает список созданных признаков.

        Returns:
            Список новых колонок
        """
        return [c for c in self.df.columns if c not in self.original_columns]
# features/feature_selector.py
"""
Автоматический отбор наиболее важных признаков для машинного обучения.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.decorators import timer
from utils.logger import log

warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Автоматический отбор наиболее важных признаков.
    Использует несколько методов для надежности.
    """

    def __init__(self, n_features: int = 30, correlation_threshold: float = 0.85,
                 task: str = 'classification'):
        """
        Инициализация селектора признаков.

        Args:
            n_features: количество признаков для отбора
            correlation_threshold: порог корреляции для удаления коллинеарных
            task: тип задачи ('classification' или 'regression')
        """
        self.n_features = n_features
        self.correlation_threshold = correlation_threshold
        self.task = task
        self.selected_features = None
        self.feature_importance = None
        self.removed_features = []

    @timer
    def select_features(self, df: pd.DataFrame, feature_cols: List[str],
                        target_col: str) -> Tuple[List[str], Dict]:
        """
        Полный пайплайн отбора признаков.

        Args:
            df: DataFrame с данными
            feature_cols: список признаков
            target_col: название целевой колонки

        Returns:
            (список отобранных признаков, статистика)
        """
        log.info(f"🔍 Запуск автоматического отбора признаков")
        log.info(f"  Исходное количество признаков: {len(feature_cols)}")

        stats = {}
        current_features = feature_cols.copy()

        # Шаг 1: Удаление константных признаков
        current_features, const_removed = self._remove_constant_features(df, current_features)
        stats['removed_constant'] = len(const_removed)

        # Шаг 2: Удаление признаков с низкой дисперсией
        current_features, low_var_removed = self._remove_low_variance_features(df, current_features)
        stats['removed_low_variance'] = len(low_var_removed)

        # Шаг 3: Удаление коллинеарных признаков
        current_features, collinear_removed = self._remove_collinear_features(df, current_features)
        stats['removed_collinear'] = len(collinear_removed)

        # Шаг 4: Отбор по важности (комбинируем методы)
        selected_by_mi = self._select_by_mutual_info(df, current_features, target_col)
        selected_by_rf = self._select_by_random_forest(df, current_features, target_col)

        # Берем пересечение методов (наиболее стабильные признаки)
        selected = list(set(selected_by_mi) & set(selected_by_rf))

        # Если слишком мало, расширяем объединением
        if len(selected) < 10:
            selected = list(set(selected_by_mi) | set(selected_by_rf))

        # Ограничиваем количество
        selected = selected[:self.n_features]

        # Шаг 5: Финальная сортировка по важности
        if self.feature_importance:
            selected = sorted(selected,
                              key=lambda x: abs(self.feature_importance.get(x, 0)),
                              reverse=True)

        self.selected_features = selected

        # Статистика
        stats.update({
            'initial': len(feature_cols),
            'after_constant': len(feature_cols) - len(const_removed),
            'after_variance': len(current_features) + len(const_removed),
            'after_collinear': len(current_features),
            'final': len(selected)
        })

        log.info(f"✅ Отбор признаков завершен:")
        log.info(f"  Было признаков: {stats['initial']}")
        log.info(f"  Удалено константных: {stats['removed_constant']}")
        log.info(f"  Удалено низковариативных: {stats['removed_low_variance']}")
        log.info(f"  Удалено коллинеарных: {stats['removed_collinear']}")
        log.info(f"  Стало признаков: {stats['final']}")
        log.info(f"  Сокращение: {(1 - stats['final'] / stats['initial']) * 100:.1f}%")

        return selected, stats

    def _remove_constant_features(self, df: pd.DataFrame,
                                  feature_cols: List[str]) -> Tuple[List[str], List[str]]:
        """Удаляет константные признаки."""
        removed = []
        remaining = []

        for col in feature_cols:
            if col in df.columns:
                if df[col].nunique() <= 1:
                    removed.append(col)
                    self.removed_features.append(('constant', col))
                else:
                    remaining.append(col)

        if removed:
            log.debug(f"  Удалено {len(removed)} константных признаков")

        return remaining, removed

    def _remove_low_variance_features(self, df: pd.DataFrame,
                                      feature_cols: List[str],
                                      threshold: float = 0.01) -> Tuple[List[str], List[str]]:
        """Удаляет признаки с низкой дисперсией."""
        removed = []
        remaining = []

        for col in feature_cols:
            if col in df.columns:
                variance = df[col].var()
                if variance < threshold:
                    removed.append(col)
                    self.removed_features.append(('low_variance', col))
                else:
                    remaining.append(col)

        if removed:
            log.debug(f"  Удалено {len(removed)} низковариативных признаков")

        return remaining, removed

    def _remove_collinear_features(self, df: pd.DataFrame,
                                   feature_cols: List[str]) -> Tuple[List[str], List[str]]:
        """Удаляет сильно коррелирующие признаки."""
        if len(feature_cols) < 2:
            return feature_cols, []

        # Вычисляем корреляционную матрицу
        corr_matrix = df[feature_cols].corr().abs()

        # Верхний треугольник матрицы
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Находим признаки для удаления
        to_drop = set()
        for column in upper.columns:
            # Находим признаки с корреляцией выше порога
            correlated = upper[column][upper[column] > self.correlation_threshold].index.tolist()

            for corr_col in correlated:
                # Оставляем признак с большей дисперсией
                if column in df.columns and corr_col in df.columns:
                    var_col = df[column].var()
                    var_corr = df[corr_col].var()

                    if var_col > var_corr:
                        to_drop.add(corr_col)
                        self.removed_features.append(('collinear', corr_col))
                    else:
                        to_drop.add(column)
                        self.removed_features.append(('collinear', column))

        remaining = [f for f in feature_cols if f not in to_drop]
        removed = list(to_drop)

        if removed:
            log.debug(f"  Удалено {len(removed)} коллинеарных признаков")

        return remaining, removed

    def _select_by_mutual_info(self, df: pd.DataFrame,
                               feature_cols: List[str],
                               target_col: str) -> List[str]:
        """Отбор признаков по взаимной информации."""
        X = df[feature_cols].fillna(0)
        y = df[target_col]

        # Определяем количество признаков для отбора
        k = min(self.n_features * 2, len(feature_cols))

        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X, y)

        # Получаем важность
        importance = dict(zip(feature_cols, selector.scores_))
        self.feature_importance = importance

        # Сортируем и возвращаем топ
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        return [f[0] for f in sorted_features[:self.n_features]]

    def _select_by_random_forest(self, df: pd.DataFrame,
                                 feature_cols: List[str],
                                 target_col: str) -> List[str]:
        """Отбор признаков с помощью Random Forest."""
        X = df[feature_cols].fillna(0)
        y = df[target_col]

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Получаем важность
        importance = dict(zip(feature_cols, rf.feature_importances_))

        # Обновляем общую важность
        if self.feature_importance is None:
            self.feature_importance = importance
        else:
            for k, v in importance.items():
                if k in self.feature_importance:
                    self.feature_importance[k] = (self.feature_importance[k] + v) / 2
                else:
                    self.feature_importance[k] = v

        # Сортируем и возвращаем топ
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        return [f[0] for f in sorted_features[:self.n_features]]

    def _select_by_lasso(self, df: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str) -> List[str]:
        """Отбор признаков с помощью Lasso (для регрессии)."""
        if self.task != 'regression':
            return feature_cols

        X = df[feature_cols].fillna(0)
        y = df[target_col]

        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X, y)

        # Выбираем признаки с ненулевыми коэффициентами
        selected = [feature_cols[i] for i in range(len(feature_cols))
                    if abs(lasso.coef_[i]) > 1e-5]

        return selected

    def get_removed_features_report(self) -> pd.DataFrame:
        """
        Возвращает отчет об удаленных признаках.

        Returns:
            DataFrame с информацией об удаленных признаках
        """
        if not self.removed_features:
            return pd.DataFrame()

        df = pd.DataFrame(self.removed_features, columns=['reason', 'feature'])
        summary = df.groupby('reason').size().reset_index(name='count')

        log.info(f"📊 Отчет об удаленных признаках:")
        for _, row in summary.iterrows():
            log.info(f"  {row['reason']}: {row['count']}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет отбор признаков к DataFrame.

        Args:
            df: исходный DataFrame

        Returns:
            DataFrame только с отобранными признаками
        """
        if self.selected_features is None:
            raise ValueError("Сначала нужно запустить select_features()")

        # Берем только существующие колонки
        available_features = [f for f in self.selected_features if f in df.columns]

        return df[available_features]
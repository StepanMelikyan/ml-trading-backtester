# models/ensemble_models.py
"""
Ансамблевые модели: Random Forest, XGBoost, LightGBM.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import time
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import lightgbm as lgb

sys.path.append(str(Path(__file__).parent.parent))
from config.model_config import MODEL_CONFIG
from .base_model import BaseModel
from utils.logger import log
from utils.decorators import timer


class RandomForestModel(BaseModel):
    """
    Random Forest Classifier.
    Ансамбль решающих деревьев с бэггингом.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Инициализация Random Forest.

        Args:
            symbol: торговый инструмент
            **kwargs: параметры модели
        """
        super().__init__("RandomForest", symbol)

        # Загружаем параметры из конфига
        config = MODEL_CONFIG['random_forest']['params'].copy()
        config.update(kwargs)
        self.params = config

    def build(self, **params):
        """
        Построение модели.

        Args:
            **params: параметры для обновления
        """
        self.params.update(params)
        self.model = RandomForestClassifier(**self.params)
        log.info(f"🌲 Построена модель Random Forest с параметрами: {self.params}")
        return self

    @timer
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Обучение модели.

        Args:
            X_train: обучающие признаки
            y_train: обучающие цели
            X_val: валидационные признаки (не используется в RF)
            y_val: валидационные цели
            **kwargs: дополнительные параметры
        """
        start_time = time.time()

        if self.model is None:
            self.build()

        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        self.features = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.is_trained = True

        log.info(f"✅ Random Forest обучен за {self.training_time:.2f} сек")

        return self

    def predict(self, X):
        """Предсказание классов."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Предсказание вероятностей."""
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """Важность признаков."""
        if hasattr(self.model, 'feature_importances_') and self.features:
            return pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

    def grid_search(self, X_train, y_train, param_grid=None, cv=5):
        """
        Поиск по сетке гиперпараметров.

        Args:
            X_train: обучающие признаки
            y_train: обучающие цели
            param_grid: сетка параметров
            cv: количество фолдов
        """
        if param_grid is None:
            param_grid = MODEL_CONFIG['random_forest']['grid_search']

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        log.info(f"🏆 Лучшие параметры: {grid.best_params_}")
        log.info(f"🏆 Лучший score: {grid.best_score_:.4f}")

        self.params.update(grid.best_params_)
        self.model = grid.best_estimator_

        return grid


class XGBoostModel(BaseModel):
    """
    XGBoost Classifier.
    Градиентный бустинг с регуляризацией.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Инициализация XGBoost.

        Args:
            symbol: торговый инструмент
            **kwargs: параметры модели
        """
        super().__init__("XGBoost", symbol)

        # Загружаем параметры из конфига
        config = MODEL_CONFIG['xgboost']['params'].copy()
        config.update(kwargs)
        self.params = config

    def build(self, **params):
        """
        Построение модели.

        Args:
            **params: параметры для обновления
        """
        self.params.update(params)
        self.model = xgb.XGBClassifier(**self.params)
        log.info(f"🚀 Построена модель XGBoost с параметрами: {self.params}")
        return self

    @timer
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Обучение модели.

        Args:
            X_train: обучающие признаки
            y_train: обучающие цели
            X_val: валидационные признаки
            y_val: валидационные цели
            **kwargs: дополнительные параметры
        """
        start_time = time.time()

        if self.model is None:
            self.build()

        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            **kwargs
        )

        self.training_time = time.time() - start_time
        self.features = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.is_trained = True

        log.info(f"✅ XGBoost обучен за {self.training_time:.2f} сек")

        return self

    def predict(self, X):
        """Предсказание классов."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Предсказание вероятностей."""
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """Важность признаков."""
        if hasattr(self.model, 'feature_importances_') and self.features:
            return pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

    def grid_search(self, X_train, y_train, param_grid=None, cv=5):
        """
        Поиск по сетке гиперпараметров.
        """
        if param_grid is None:
            param_grid = MODEL_CONFIG['xgboost']['grid_search']

        grid = GridSearchCV(
            xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        log.info(f"🏆 Лучшие параметры: {grid.best_params_}")
        log.info(f"🏆 Лучший score: {grid.best_score_:.4f}")

        self.params.update(grid.best_params_)
        self.model = grid.best_estimator_

        return grid


class LightGBMModel(BaseModel):
    """
    LightGBM Classifier.
    Быстрый градиентный бустинг с Leaf-wise ростом.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Инициализация LightGBM.

        Args:
            symbol: торговый инструмент
            **kwargs: параметры модели
        """
        super().__init__("LightGBM", symbol)

        # Загружаем параметры из конфига
        config = MODEL_CONFIG['lightgbm']['params'].copy()
        config.update(kwargs)
        self.params = config

    def build(self, **params):
        """
        Построение модели.

        Args:
            **params: параметры для обновления
        """
        self.params.update(params)
        self.model = lgb.LGBMClassifier(**self.params)
        log.info(f"⚡ Построена модель LightGBM с параметрами: {self.params}")
        return self

    @timer
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Обучение модели.

        Args:
            X_train: обучающие признаки
            y_train: обучающие цели
            X_val: валидационные признаки
            y_val: валидационные цели
            **kwargs: дополнительные параметры
        """
        start_time = time.time()

        if self.model is None:
            self.build()

        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            **kwargs
        )

        self.training_time = time.time() - start_time
        self.features = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.is_trained = True

        log.info(f"✅ LightGBM обучен за {self.training_time:.2f} сек")

        return self

    def predict(self, X):
        """Предсказание классов."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Предсказание вероятностей."""
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """Важность признаков."""
        if hasattr(self.model, 'feature_importances_') and self.features:
            return pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

    def grid_search(self, X_train, y_train, param_grid=None, cv=5):
        """
        Поиск по сетке гиперпараметров.
        """
        if param_grid is None:
            param_grid = MODEL_CONFIG['lightgbm']['grid_search']

        grid = GridSearchCV(
            lgb.LGBMClassifier(verbose=-1),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        log.info(f"🏆 Лучшие параметры: {grid.best_params_}")
        log.info(f"🏆 Лучший score: {grid.best_score_:.4f}")

        self.params.update(grid.best_params_)
        self.model = grid.best_estimator_

        return grid


class EnsembleModel(BaseModel):
    """
    Ансамбль из нескольких моделей (голосование).
    """

    def __init__(self, symbol: str, models: Optional[List[BaseModel]] = None):
        """
        Инициализация ансамбля.

        Args:
            symbol: торговый инструмент
            models: список моделей для ансамбля
        """
        super().__init__("Ensemble", symbol)
        self.models = models or []
        self.weights = None
        self.voting = 'soft'  # 'soft' или 'hard'

    def add_model(self, model: BaseModel, weight: float = 1.0):
        """
        Добавить модель в ансамбль.

        Args:
            model: экземпляр модели
            weight: вес модели при голосовании
        """
        self.models.append(model)
        if self.weights is None:
            self.weights = [weight]
        else:
            self.weights.append(weight)

        log.info(f"➕ Добавлена модель {model.name} с весом {weight}")

        return self

    def build(self, **params):
        """Построение (ничего не делает)."""
        pass

    @timer
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Обучение всех моделей в ансамбле.

        Args:
            X_train: обучающие признаки
            y_train: обучающие цели
            X_val: валидационные признаки
            y_val: валидационные цели
            **kwargs: дополнительные параметры
        """
        start_time = time.time()

        if not self.models:
            raise ValueError("Ансамбль пуст. Добавьте модели через add_model()")

        # Обучаем каждую модель
        for i, model in enumerate(self.models):
            log.info(f"  Обучение {model.name} ({i + 1}/{len(self.models)})...")
            model.train(X_train, y_train, X_val, y_val, **kwargs)

        # Оптимизация весов на валидации (если есть)
        if X_val is not None and len(self.models) > 1:
            self._optimize_weights(X_val, y_val)

        self.training_time = time.time() - start_time
        self.features = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.is_trained = True

        log.info(f"✅ Ансамбль обучен за {self.training_time:.2f} сек")

        return self

    def _optimize_weights(self, X_val, y_val):
        """
        Оптимизация весов моделей на валидации.

        Args:
            X_val: валидационные признаки
            y_val: валидационные цели
        """
        from scipy.optimize import minimize
        from sklearn.metrics import f1_score

        # Получаем предсказания всех моделей
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X_val)
            if pred.shape[1] > 1:
                pred = pred[:, 1]  # Берем вероятность положительного класса
            predictions.append(pred)

        # Функция для оптимизации
        def objective(weights):
            weights = np.array(weights)
            weights = weights / (weights.sum() + 1e-10)  # Нормализация

            # Взвешенное голосование
            ensemble_pred = np.zeros(len(y_val))
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * pred

            # Бинаризация
            ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)

            # Минимизируем -f1 (чтобы максимизировать f1)
            return -f1_score(y_val, ensemble_pred_binary, average='weighted')

        # Начальные веса (равные)
        initial_weights = [1.0 / len(self.models)] * len(self.models)

        # Оптимизация
        bounds = [(0, 1)] * len(self.models)
        result = minimize(objective, initial_weights, bounds=bounds, method='SLSQP')

        if result.success:
            self.weights = result.x / (result.x.sum() + 1e-10)
            log.info(f"  Оптимальные веса: {dict(zip([m.name for m in self.models], self.weights))}")

    def predict(self, X):
        """
        Предсказание (взвешенное голосование).

        Args:
            X: признаки

        Returns:
            Предсказанные классы
        """
        if not self.models:
            raise ValueError("Ансамбль пуст")

        if self.voting == 'soft':
            # Мягкое голосование (по вероятностям)
            if self.weights is None:
                weights = [1.0 / len(self.models)] * len(self.models)
            else:
                weights = self.weights

            ensemble_proba = np.zeros((len(X), 2))

            for model, weight in zip(self.models, weights):
                proba = model.predict_proba(X)
                ensemble_proba += weight * proba

            return np.argmax(ensemble_proba, axis=1)

        else:
            # Жесткое голосование (по классам)
            predictions = np.array([model.predict(X) for model in self.models])

            if self.weights is None:
                # Простое большинство
                return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
            else:
                # Взвешенное голосование
                weighted_pred = np.zeros((len(X), len(np.unique(predictions))))
                for i, (model, weight) in enumerate(zip(self.models, self.weights)):
                    for j, pred in enumerate(predictions[i]):
                        weighted_pred[j, pred] += weight

                return np.argmax(weighted_pred, axis=1)

    def predict_proba(self, X):
        """
        Предсказание вероятностей.

        Args:
            X: признаки

        Returns:
            Вероятности классов
        """
        if not self.models:
            raise ValueError("Ансамбль пуст")

        if self.weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        else:
            weights = self.weights

        ensemble_proba = np.zeros((len(X), 2))

        for model, weight in zip(self.models, weights):
            proba = model.predict_proba(X)
            ensemble_proba += weight * proba

        return ensemble_proba
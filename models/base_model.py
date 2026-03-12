# models/base_model.py
"""
Базовый абстрактный класс для всех моделей машинного обучения.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import MODELS_DIR
from utils.logger import log
from utils.decorators import timer


class BaseModel(ABC):
    """
    Абстрактный базовый класс для всех моделей.
    Определяет единый интерфейс для обучения, предсказания и сохранения.
    """

    def __init__(self, name: str, symbol: str):
        """
        Инициализация базовой модели.

        Args:
            name: название модели
            symbol: торговый инструмент
        """
        self.name = name
        self.symbol = symbol
        self.model = None
        self.features = None
        self.metrics = {}
        self.training_time = None
        self.created_at = datetime.now()
        self.is_trained = False

    @abstractmethod
    def build(self, **params):
        """Построение архитектуры модели."""
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Обучение модели."""
        pass

    @abstractmethod
    def predict(self, X):
        """Предсказание."""
        pass

    def predict_proba(self, X):
        """
        Вероятности предсказаний (если поддерживается).

        Args:
            X: признаки

        Returns:
            Вероятности классов
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Конвертируем предсказания в вероятности
            preds = self.predict(X)
            if len(np.unique(preds)) > 2:
                # Мультикласс
                n_classes = len(np.unique(preds))
                proba = np.zeros((len(preds), n_classes))
                for i, p in enumerate(preds):
                    proba[i, p] = 1.0
                return proba
            else:
                # Бинарный
                return np.column_stack([1 - preds, preds])

    @timer
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Оценка модели на тестовых данных.

        Args:
            X_test: тестовые признаки
            y_test: тестовые цели

        Returns:
            Словарь с метриками
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     mean_squared_error, mean_absolute_error, r2_score)

        y_pred = self.predict(X_test)

        # Определяем тип задачи по целевой переменной
        if y_test.dtype in ['int64', 'int32', 'bool'] or len(np.unique(y_test)) < 10:
            # Классификация
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }

            # ROC-AUC для бинарной классификации
            if len(np.unique(y_test)) == 2:
                try:
                    y_proba = self.predict_proba(X_test)
                    if y_proba.shape[1] > 1:
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                except:
                    pass

            # Матрица ошибок
            cm = confusion_matrix(y_test, y_pred)
            if cm.size == 4:  # Бинарная
                tn, fp, fn, tp = cm.ravel()
                metrics['true_negative'] = int(tn)
                metrics['false_positive'] = int(fp)
                metrics['false_negative'] = int(fn)
                metrics['true_positive'] = int(tp)

        else:
            # Регрессия
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

        self.metrics = metrics
        log.info(f"📊 Метрики {self.name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                log.info(f"  {k}: {v:.4f}")

        return metrics

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Сохранение модели и метаданных.

        Args:
            path: путь для сохранения

        Returns:
            Путь к сохраненной модели
        """
        if path is None:
            # Создаем уникальное имя
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = MODELS_DIR / f"{self.symbol}_{self.name}_{timestamp}"
        else:
            model_dir = Path(path)

        model_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем модель
        model_path = model_dir / "model.joblib"
        joblib.dump(self.model, model_path)

        # Сохраняем метаданные
        metadata = {
            'name': self.name,
            'symbol': self.symbol,
            'created_at': self.created_at.isoformat(),
            'training_time': self.training_time,
            'metrics': self.metrics,
            'features': self.features if self.features is not None else None,  # ← ИСПРАВЛЕНО
            'is_trained': self.is_trained
        }

        with open(model_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Сохраняем конфигурацию если есть
        if hasattr(self, 'params'):
            with open(model_dir / "params.json", 'w', encoding='utf-8') as f:
                json.dump(self.params, f, indent=2, default=str)

        size_mb = sum(f.stat().st_size for f in model_dir.glob('**/*') if f.is_file()) / (1024 * 1024)
        log.info(f"💾 Модель сохранена в {model_dir} ({size_mb:.2f} MB)")

        return model_dir

    def load(self, path: Path):
        """
        Загрузка модели.

        Args:
            path: путь к модели

        Returns:
            self
        """
        model_path = Path(path) / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.model = joblib.load(model_path)

        # Загружаем метаданные
        metadata_path = Path(path) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.name = metadata.get('name', self.name)
                self.symbol = metadata.get('symbol', self.symbol)
                self.created_at = datetime.fromisoformat(metadata['created_at']) if 'created_at' in metadata else None
                self.training_time = metadata.get('training_time')
                self.metrics = metadata.get('metrics', {})
                self.features = metadata.get('features')
                self.is_trained = metadata.get('is_trained', True)

        # Загружаем параметры
        params_path = Path(path) / "params.json"
        if params_path.exists() and hasattr(self, 'params'):
            with open(params_path, 'r', encoding='utf-8') as f:
                self.params.update(json.load(f))

        log.info(f"📂 Модель загружена из {path}")

        return self

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Возвращает важность признаков (если доступно).

        Returns:
            DataFrame с важностью признаков
        """
        if hasattr(self.model, 'feature_importances_') and self.features is not None:
            importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        elif hasattr(self.model, 'coef_') and self.features is not None:
            # Для линейных моделей
            importance = pd.DataFrame({
                'feature': self.features,
                'importance': np.abs(self.model.coef_).flatten()
            }).sort_values('importance', ascending=False)
            return importance
        return None

    def summary(self):
        """Выводит краткое описание модели."""
        print(f"\n{'=' * 50}")
        print(f"МОДЕЛЬ: {self.name} [{self.symbol}]")
        print(f"{'=' * 50}")
        print(f"Создана: {self.created_at}")
        print(f"Обучена: {'Да' if self.is_trained else 'Нет'}")
        if self.training_time:
            print(f"Время обучения: {self.training_time:.2f} сек")

        if self.metrics:
            print(f"\nМетрики:")
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        importance = self.get_feature_importance()
        if importance is not None:
            print(f"\nТоп-5 важных признаков:")
            for i, row in importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
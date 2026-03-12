# models/deep_learning.py
"""
Глубокое обучение: LSTM, GRU, Transformer модели для временных рядов.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import time
import sys
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input,
                                     Bidirectional, Attention, concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

sys.path.append(str(Path(__file__).parent.parent))
from config.model_config import MODEL_CONFIG
from .base_model import BaseModel
from utils.logger import log
from utils.decorators import timer


class LSTMModel(BaseModel):
    """
    LSTM модель для анализа временных рядов.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Инициализация LSTM.

        Args:
            symbol: торговый инструмент
            **kwargs: параметры модели
        """
        super().__init__("LSTM", symbol)

        config = MODEL_CONFIG['lstm']['params'].copy()
        config.update(kwargs)
        self.params = config

        self.sequence_length = self.params.get('sequence_length', 60)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.history = None

    def build(self, **params):
        """
        Построение архитектуры LSTM.

        Args:
            **params: параметры для обновления
        """
        self.params.update(params)

        model = Sequential()

        # Первый LSTM слой
        model.add(LSTM(
            self.params.get('lstm_units', [100])[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.params.get('n_features', 1))
        ))
        model.add(Dropout(self.params.get('dropout', 0.2)))

        # Второй LSTM слой
        if len(self.params.get('lstm_units', [100])) > 1:
            model.add(LSTM(
                self.params.get('lstm_units')[1],
                return_sequences=False
            ))
            model.add(Dropout(self.params.get('dropout', 0.2)))

        # Выходной слой
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        self.model = model
        log.info(f"🧠 Построена LSTM модель")
        model.summary(print_fn=lambda x: log.debug(x))

        return self

    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str],
                          target_col: str = 'target', test_size: float = 0.2):
        """
        Подготовка последовательностей для LSTM.

        Args:
            df: DataFrame с данными
            feature_cols: список признаков
            target_col: целевая колонка
            test_size: размер тестовой выборки

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        # Масштабируем признаки
        X_scaled = self.scaler_X.fit_transform(df[feature_cols].values)
        y_scaled = self.scaler_y.fit_transform(df[[target_col]].values).flatten()

        # Создаем последовательности
        X, y = [], []
        for i in range(self.sequence_length, len(X_scaled)):
            X.append(X_scaled[i - self.sequence_length:i])
            y.append(y_scaled[i])

        X = np.array(X)
        y = np.array(y)

        # Разделение с учетом времени
        split_idx = int(len(X) * (1 - test_size))

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        log.info(f"📊 Форма обучающих данных: {X_train.shape}")
        log.info(f"📊 Форма тестовых данных: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    @timer
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Обучение LSTM.

        Args:
            X_train: обучающие последовательности
            y_train: обучающие цели
            X_val: валидационные последовательности
            y_val: валидационные цели
            **kwargs: дополнительные параметры
        """
        start_time = time.time()

        if self.model is None:
            # Определяем количество признаков
            self.params['n_features'] = X_train.shape[2]
            self.build()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.params.get('patience', 15),
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.params.get('epochs', 100),
            batch_size=self.params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )

        self.training_time = time.time() - start_time
        self.is_trained = True

        log.info(f"✅ LSTM обучена за {self.training_time:.2f} сек")

        return self.history

    def predict(self, X):
        """
        Предсказание классов.

        Args:
            X: последовательности

        Returns:
            Предсказанные классы (0 или 1)
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        """
        Предсказание вероятностей.

        Args:
            X: последовательности

        Returns:
            Вероятности классов
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        predictions = self.model.predict(X, verbose=0)
        return np.column_stack([1 - predictions, predictions])

    def save(self, path=None):
        """
        Сохранение LSTM модели.

        Args:
            path: путь для сохранения
        """
        save_path = super().save(path)

        # Дополнительно сохраняем scaler
        import joblib
        joblib.dump(self.scaler_X, save_path / "scaler_X.pkl")
        joblib.dump(self.scaler_y, save_path / "scaler_y.pkl")

        return save_path

    def load(self, path):
        """
        Загрузка LSTM модели.

        Args:
            path: путь к модели
        """
        super().load(path)

        # Загружаем scaler
        import joblib
        scaler_X_path = Path(path) / "scaler_X.pkl"
        scaler_y_path = Path(path) / "scaler_y.pkl"

        if scaler_X_path.exists():
            self.scaler_X = joblib.load(scaler_X_path)
        if scaler_y_path.exists():
            self.scaler_y = joblib.load(scaler_y_path)

        return self

    def plot_training_history(self):
        """
        Визуализация истории обучения.
        """
        if self.history is None:
            log.warning("Нет истории обучения")
            return

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # График потерь
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Потери при обучении')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # График точности
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Точность при обучении')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig


class GRUModel(LSTMModel):
    """
    GRU модель - упрощенная версия LSTM.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Инициализация GRU.
        """
        super().__init__(symbol, **kwargs)
        self.name = "GRU"

    def build(self, **params):
        """
        Построение архитектуры GRU.
        """
        self.params.update(params)

        model = Sequential()

        # Первый GRU слой
        model.add(GRU(
            self.params.get('lstm_units', [100])[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.params.get('n_features', 1))
        ))
        model.add(Dropout(self.params.get('dropout', 0.2)))

        # Второй GRU слой
        if len(self.params.get('lstm_units', [100])) > 1:
            model.add(GRU(
                self.params.get('lstm_units')[1],
                return_sequences=False
            ))
            model.add(Dropout(self.params.get('dropout', 0.2)))

        # Выходной слой
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        self.model = model
        log.info(f"🧠 Построена GRU модель")

        return self


class BidirectionalLSTM(LSTMModel):
    """
    Двунаправленная LSTM модель.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Инициализация Bidirectional LSTM.
        """
        super().__init__(symbol, **kwargs)
        self.name = "BiLSTM"

    def build(self, **params):
        """
        Построение архитектуры Bidirectional LSTM.
        """
        self.params.update(params)

        model = Sequential()

        # Двунаправленный LSTM слой
        model.add(Bidirectional(
            LSTM(self.params.get('lstm_units', [100])[0], return_sequences=True),
            input_shape=(self.sequence_length, self.params.get('n_features', 1))
        ))
        model.add(Dropout(self.params.get('dropout', 0.2)))

        # Второй слой
        if len(self.params.get('lstm_units', [100])) > 1:
            model.add(Bidirectional(
                LSTM(self.params.get('lstm_units')[1], return_sequences=False)
            ))
            model.add(Dropout(self.params.get('dropout', 0.2)))
        else:
            model.add(Flatten())

        # Выходной слой
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        self.model = model
        log.info(f"🧠 Построена Bidirectional LSTM модель")

        return self
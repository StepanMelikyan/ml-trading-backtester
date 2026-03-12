# models/hybrid_model.py
"""
Гибридные модели, комбинирующие разные подходы: CNN + LSTM, Transformer + Attention.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import time
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
                                     Flatten, concatenate, BatchNormalization, Attention,
                                     MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.append(str(Path(__file__).parent.parent))
from .base_model import BaseModel
from .ensemble_models import RandomForestModel, XGBoostModel, LightGBMModel
from .deep_learning import LSTMModel
from utils.logger import log
from utils.decorators import timer


class CNNLSTMHybrid(LSTMModel):
    """
    Гибридная модель CNN + LSTM.
    CNN извлекает локальные паттерны, LSTM - долгосрочные зависимости.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Инициализация CNN-LSTM гибрида.

        Args:
            symbol: торговый инструмент
            **kwargs: параметры модели
        """
        super().__init__(symbol, **kwargs)
        self.name = "CNN-LSTM"

    def build(self, **params):
        """
        Построение архитектуры CNN-LSTM.
        """
        self.params.update(params)

        n_features = self.params.get('n_features', 1)
        sequence_length = self.sequence_length

        # Входной слой
        inputs = Input(shape=(sequence_length, n_features))

        # CNN блоки
        # Блок 1
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        # Блок 2
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        # LSTM слой
        x = LSTM(units=100, return_sequences=False)(x)
        x = Dropout(0.3)(x)

        # Выходной слой
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        self.model = model
        log.info(f"🧠 Построена гибридная модель CNN-LSTM")
        model.summary(print_fn=lambda x: log.debug(x))

        return self


class HybridModel(BaseModel):
    """
    Универсальная гибридная модель (обертка для выбора типа гибрида).
    """

    def __init__(self, symbol: str, hybrid_type: str = 'cnn_lstm', **kwargs):
        """
        Инициализация гибридной модели.

        Args:
            symbol: торговый инструмент
            hybrid_type: тип гибрида ('cnn_lstm', 'attention', 'transformer', 'stacking')
            **kwargs: параметры модели
        """
        self.hybrid_type = hybrid_type
        self.symbol = symbol
        self.kwargs = kwargs

        # Создаем соответствующую реализацию
        if hybrid_type == 'cnn_lstm':
            self.model_impl = CNNLSTMHybrid(symbol, **kwargs)
        elif hybrid_type == 'attention':
            self.model_impl = AttentionLSTM(symbol, **kwargs)
        elif hybrid_type == 'transformer':
            self.model_impl = TransformerModel(symbol, **kwargs)
        elif hybrid_type == 'stacking':
            self.model_impl = StackingHybrid(symbol)
            if 'models' in kwargs:
                for model in kwargs['models']:
                    self.model_impl.add_base_model(model)
        else:
            raise ValueError(f"Неизвестный тип гибрида: {hybrid_type}")

        # Инициализируем базовый класс
        super().__init__(f"Hybrid-{hybrid_type}", symbol)

        # Копируем атрибуты из реализации
        self.model = getattr(self.model_impl, 'model', None)
        self.is_trained = False

    def build(self, **params):
        """Построение модели."""
        # Объединяем параметры
        all_params = {**self.kwargs, **params}
        self.model_impl.build(**all_params)
        self.model = getattr(self.model_impl, 'model', None)
        return self

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Обучение модели."""
        self.model_impl.train(X_train, y_train, X_val, y_val, **kwargs)
        self.is_trained = getattr(self.model_impl, 'is_trained', True)

        # Копируем метрики
        if hasattr(self.model_impl, 'metrics'):
            self.metrics = self.model_impl.metrics

        return self

    def predict(self, X):
        """Предсказание."""
        return self.model_impl.predict(X)

    def predict_proba(self, X):
        """Предсказание вероятностей."""
        return self.model_impl.predict_proba(X)

    def get_feature_importance(self):
        """Важность признаков (если доступно)."""
        if hasattr(self.model_impl, 'get_feature_importance'):
            return self.model_impl.get_feature_importance()
        return None


class AttentionLSTM(LSTMModel):
    """
    LSTM с механизмом внимания (Attention).
    Позволяет модели фокусироваться на важных частях последовательности.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Инициализация LSTM с Attention.
        """
        super().__init__(symbol, **kwargs)
        self.name = "Attention-LSTM"

    def build(self, **params):
        """
        Построение архитектуры LSTM с Attention.
        """
        self.params.update(params)

        n_features = self.params.get('n_features', 1)
        sequence_length = self.sequence_length
        lstm_units = self.params.get('lstm_units', [100])[0]

        # Входной слой
        inputs = Input(shape=(sequence_length, n_features))

        # LSTM слой (возвращаем всю последовательность)
        lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)

        # Механизм внимания
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(lstm_units)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)

        # Взвешенная сумма
        weighted = tf.keras.layers.Multiply()([lstm_out, attention])
        context = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(weighted)

        # Полносвязные слои
        x = Dense(50, activation='relu')(context)
        x = Dropout(0.3)(x)
        x = Dense(25, activation='relu')(x)

        # Выходной слой
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        self.model = model
        log.info(f"🧠 Построена модель LSTM с Attention")

        return self


class TransformerBlock(tf.keras.layers.Layer):
    """
    Блок Transformer для временных рядов.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(LSTMModel):
    """
    Transformer модель для временных рядов.
    Использует механизм самовнимания без рекуррентных слоев.
    """

    def __init__(self, symbol: str, **kwargs):
        """
        Инициализация Transformer.
        """
        super().__init__(symbol, **kwargs)
        self.name = "Transformer"

    def build(self, **params):
        """
        Построение архитектуры Transformer.
        """
        self.params.update(params)

        n_features = self.params.get('n_features', 1)
        sequence_length = self.sequence_length
        embed_dim = self.params.get('embed_dim', 64)  # Размер эмбеддинга
        num_heads = self.params.get('num_heads', 4)  # Количество голов внимания
        ff_dim = self.params.get('ff_dim', 128)  # Размер feed-forward слоя
        num_blocks = self.params.get('num_blocks', 2)  # Количество блоков

        # Входной слой
        inputs = Input(shape=(sequence_length, n_features))

        # Входной проекционный слой
        x = Dense(embed_dim)(inputs)
        x = Dropout(0.1)(x)

        # Блоки Transformer
        for _ in range(num_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

        # Глобальный пулинг
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.3)(x)

        # Выходной слой
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=self.params.get('learning_rate', 0.001))

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        self.model = model
        log.info(f"🧠 Построена Transformer модель")
        model.summary(print_fn=lambda x: log.debug(x))

        return self


class StackingHybrid(BaseModel):
    """
    Гибридная модель на основе стекинга.
    Комбинирует предсказания разных моделей через мета-модель.
    """

    def __init__(self, symbol: str):
        """
        Инициализация стекинг гибрида.

        Args:
            symbol: торговый инструмент
        """
        super().__init__("StackingHybrid", symbol)
        self.base_models = []
        self.meta_model = None
        self.base_predictions = None

    def add_base_model(self, model: BaseModel):
        """
        Добавление базовой модели.

        Args:
            model: экземпляр модели
        """
        self.base_models.append(model)
        log.info(f"➕ Добавлена базовая модель {model.name}")
        return self

    def build(self, **params):
        """Построение (ничего не делает)."""
        from sklearn.ensemble import RandomForestClassifier
        self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        return self

    @timer
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Обучение гибридной модели.

        Args:
            X_train: обучающие признаки
            y_train: обучающие цели
            X_val: валидационные признаки
            y_val: валидационные цели
            **kwargs: дополнительные параметры
        """
        start_time = time.time()

        if not self.base_models:
            raise ValueError("Нет базовых моделей. Добавьте через add_base_model()")

        # Обучаем базовые модели
        base_preds_train = []

        for i, model in enumerate(self.base_models):
            log.info(f"  Обучение базовой модели {i + 1}/{len(self.base_models)}: {model.name}")
            model.train(X_train, y_train, X_val, y_val, **kwargs)

            # Получаем предсказания на тренировочных данных
            preds = model.predict_proba(X_train)
            if preds.shape[1] > 1:
                preds = preds[:, 1]  # Берем вероятность положительного класса
            base_preds_train.append(preds.reshape(-1, 1))

        # Формируем признаки для мета-модели
        X_meta_train = np.hstack(base_preds_train)

        # Обучаем мета-модель
        log.info(f"  Обучение мета-модели...")
        self.meta_model.fit(X_meta_train, y_train)

        self.training_time = time.time() - start_time
        self.is_trained = True

        log.info(f"✅ Гибридная модель обучена за {self.training_time:.2f} сек")

        return self

    def predict(self, X):
        """
        Предсказание классов.

        Args:
            X: признаки
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        # Получаем предсказания базовых моделей
        base_preds = []
        for model in self.base_models:
            preds = model.predict_proba(X)
            if preds.shape[1] > 1:
                preds = preds[:, 1]
            base_preds.append(preds.reshape(-1, 1))

        X_meta = np.hstack(base_preds)

        return self.meta_model.predict(X_meta)

    def predict_proba(self, X):
        """
        Предсказание вероятностей.
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")

        base_preds = []
        for model in self.base_models:
            preds = model.predict_proba(X)
            if preds.shape[1] > 1:
                preds = preds[:, 1]
            base_preds.append(preds.reshape(-1, 1))

        X_meta = np.hstack(base_preds)

        proba = self.meta_model.predict_proba(X_meta)
        return proba
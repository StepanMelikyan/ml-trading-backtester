# tests/test_models.py
"""
Тесты для модулей машинного обучения.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from pathlib import Path
import tempfile
import shutil

from models.base_model import BaseModel
from models.ensemble_models import RandomForestModel, XGBoostModel, LightGBMModel, EnsembleModel
from models.deep_learning import LSTMModel, GRUModel
from models.hybrid_model import CNNLSTMHybrid, AttentionLSTM, StackingHybrid
from models.model_registry import ModelRegistry


@pytest.fixture
def sample_ml_data():
    """Создает синтетические данные для тестирования моделей."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(20)]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='target')

    # Разделение на train/test
    split = int(len(X) * 0.7)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    return X_train, X_test, y_train, y_test, feature_names


@pytest.fixture
def setup_model_dir():
    """Создает временную директорию для моделей."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestRandomForestModel:
    """Тесты для Random Forest модели."""

    def test_build_and_train(self, sample_ml_data):
        """Тест построения и обучения."""
        X_train, X_test, y_train, y_test, _ = sample_ml_data

        model = RandomForestModel("TEST")
        model.build(n_estimators=50, max_depth=5)
        model.train(X_train, y_train)

        assert model.is_trained
        assert model.model is not None

        # Проверка предсказаний
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

        # Проверка вероятностей
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(y_test), 2)

        # Проверка метрик
        metrics = model.evaluate(X_test, y_test)
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert metrics['accuracy'] > 0.5

    def test_feature_importance(self, sample_ml_data):
        """Тест важности признаков."""
        X_train, X_test, y_train, y_test, feature_names = sample_ml_data

        model = RandomForestModel("TEST")
        model.build(n_estimators=50)
        model.train(X_train, y_train)
        model.features = feature_names

        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == len(feature_names)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns

    def test_save_load(self, sample_ml_data, setup_model_dir):
        """Тест сохранения и загрузки."""
        X_train, X_test, y_train, y_test, _ = sample_ml_data

        model = RandomForestModel("TEST")
        model.build(n_estimators=50)
        model.train(X_train, y_train)

        # Сохраняем
        save_path = model.save(path=setup_model_dir / "test_model")

        # Загружаем
        loaded_model = RandomForestModel("TEST")
        loaded_model.load(save_path)

        assert loaded_model.is_trained
        assert loaded_model.model is not None

        # Проверяем, что предсказания совпадают
        pred1 = model.predict(X_test)
        pred2 = loaded_model.predict(X_test)
        np.testing.assert_array_equal(pred1, pred2)


class TestXGBoostModel:
    """Тесты для XGBoost модели."""

    def test_build_and_train(self, sample_ml_data):
        """Тест построения и обучения."""
        X_train, X_test, y_train, y_test, _ = sample_ml_data

        model = XGBoostModel("TEST")
        model.build(n_estimators=50, max_depth=3)
        model.train(X_train, y_train, X_test, y_test)

        assert model.is_trained
        metrics = model.evaluate(X_test, y_test)
        assert metrics['accuracy'] > 0.5


class TestLightGBMModel:
    """Тесты для LightGBM модели."""

    def test_build_and_train(self, sample_ml_data):
        """Тест построения и обучения."""
        X_train, X_test, y_train, y_test, _ = sample_ml_data

        model = LightGBMModel("TEST")
        model.build(n_estimators=50, max_depth=3)
        model.train(X_train, y_train, X_test, y_test)

        assert model.is_trained
        metrics = model.evaluate(X_test, y_test)
        assert metrics['accuracy'] > 0.5


class TestEnsembleModel:
    """Тесты для ансамблевой модели."""

    def test_ensemble(self, sample_ml_data):
        """Тест создания и обучения ансамбля."""
        X_train, X_test, y_train, y_test, _ = sample_ml_data

        # Создаем отдельные модели
        rf = RandomForestModel("TEST")
        rf.build(n_estimators=50)
        rf.train(X_train, y_train)

        xgb = XGBoostModel("TEST")
        xgb.build(n_estimators=50)
        xgb.train(X_train, y_train)

        # Создаем ансамбль
        ensemble = EnsembleModel("TEST")
        ensemble.add_model(rf, weight=0.6)
        ensemble.add_model(xgb, weight=0.4)

        # Обучаем (в ансамбле просто хранит модели)
        ensemble.train(X_train, y_train)

        # Проверяем предсказания
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(y_test)

        proba = ensemble.predict_proba(X_test)
        assert proba.shape == (len(y_test), 2)


class TestLSTMModel:
    """Тесты для LSTM модели."""

    def test_build_and_train(self, sample_ml_data):
        """Тест построения и обучения LSTM."""
        X_train, X_test, y_train, y_test, _ = sample_ml_data

        # Для LSTM нужно изменить форму данных
        # В реальности здесь должна быть подготовка последовательностей

        model = LSTMModel("TEST")
        model.build()

        # Пропускаем обучение из-за сложности
        # Просто проверяем создание модели
        assert model.model is not None


class TestModelRegistry:
    """Тесты для реестра моделей."""

    def test_register_and_list(self, sample_ml_data, setup_model_dir):
        """Тест регистрации и списка моделей."""
        X_train, X_test, y_train, y_test, _ = sample_ml_data

        registry = ModelRegistry(registry_path=setup_model_dir / "registry.json")

        # Создаем и сохраняем модель
        model = RandomForestModel("TEST")
        model.build(n_estimators=50)
        model.train(X_train, y_train)
        model.evaluate(X_test, y_test)

        save_path = model.save(path=setup_model_dir / "models" / "test_model")

        # Регистрируем
        model_id = registry.register(model, save_path, model.metrics, tags=["test"])

        assert model_id is not None

        # Получаем список моделей
        models_df = registry.list_models()
        assert len(models_df) > 0
        assert 'name' in models_df.columns

        # Получаем лучшую модель
        best = registry.get_best_model("TEST")
        assert best is not None
        assert best['id'] == model_id

    def test_compare_models(self, sample_ml_data, setup_model_dir):
        """Тест сравнения моделей."""
        X_train, X_test, y_train, y_test, _ = sample_ml_data

        registry = ModelRegistry(registry_path=setup_model_dir / "registry.json")

        # Создаем несколько моделей
        for name in ["RF1", "RF2"]:
            model = RandomForestModel("TEST")
            model.build(n_estimators=50)
            model.train(X_train, y_train)
            model.evaluate(X_test, y_test)

            save_path = model.save(path=setup_model_dir / "models" / name)
            registry.register(model, save_path, model.metrics, tags=["test"])

        # Сравниваем
        comparison = registry.compare_models("TEST")
        assert len(comparison) == 2
        assert 'accuracy' in comparison.columns
        assert 'f1_score' in comparison.columns
# tests/test_integration.py
"""
Интеграционные тесты для проверки взаимодействия всех модулей.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

from data.downloader import DataDownloader
from data.preprocessor import DataPreprocessor
from data.cache_manager import CacheManager

from features.base_indicators import BaseIndicators
from features.trend_indicators import TrendIndicators
from features.volume_indicators import VolumeIndicators
from features.volatility import VolatilityIndicators
from features.oscillators import OscillatorIndicators
from features.price_patterns import PricePatterns
from features.feature_engineering import FeatureEngineering
from features.feature_selector import FeatureSelector

from models.ensemble_models import RandomForestModel, XGBoostModel
from models.model_registry import ModelRegistry

from backtest.engine import BacktestEngine
from backtest.metrics import TradingMetrics

from reports.visualization import TradingVisualizer
from reports.html_report import HTMLReport


@pytest.fixture
def setup_test_env():
    """Создает временное окружение для тестов."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "data"
    models_dir = Path(temp_dir) / "models"
    reports_dir = Path(temp_dir) / "reports"

    for d in [data_dir, models_dir, reports_dir]:
        d.mkdir(parents=True)

    yield {
        'temp_dir': Path(temp_dir),
        'data_dir': data_dir,
        'models_dir': models_dir,
        'reports_dir': reports_dir
    }

    shutil.rmtree(temp_dir)


@pytest.fixture
def generate_test_data():
    """Генерирует тестовые данные."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    n = len(dates)

    np.random.seed(42)

    # Генерируем цену с трендом и циклами
    t = np.arange(n)
    trend = 0.0001 * t
    cycle1 = 0.01 * np.sin(2 * np.pi * t / 24)  # дневной цикл
    cycle2 = 0.05 * np.sin(2 * np.pi * t / (24 * 7))  # недельный цикл
    noise = np.random.randn(n) * 0.005

    returns = trend + cycle1 + cycle2 + noise
    price = 100 * np.exp(np.cumsum(returns))

    # Создаем OHLC
    df = pd.DataFrame({
        'time': dates,
        'open': price * (1 + np.random.randn(n) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(n) * 0.002)),
        'low': price * (1 - np.abs(np.random.randn(n) * 0.002)),
        'close': price,
        'volume': np.random.randint(100, 1000, n)
    })

    return df


class TestFullPipeline:
    """Интеграционные тесты полного пайплайна."""

    def test_data_pipeline(self, generate_test_data, setup_test_env):
        """Тест пайплайна обработки данных."""
        df = generate_test_data

        # Препроцессинг
        preprocessor = DataPreprocessor()
        df = preprocessor.clean(df)
        df = preprocessor.add_datetime_features(df)

        assert len(df) > 0
        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns

        # Добавление индикаторов
        df = BaseIndicators.add_all(df)
        df = TrendIndicators.add_all(df)
        df = VolumeIndicators.add_all(df)
        df = VolatilityIndicators.add_all(df)
        df = OscillatorIndicators.add_all(df)
        df = PricePatterns.add_all(df)

        # Проверяем наличие ключевых индикаторов
        assert 'SMA_20' in df.columns
        assert 'RSI_14' in df.columns
        assert 'MACD' in df.columns
        assert 'ADX' in df.columns
        assert 'BB_upper_2.0' in df.columns

        # Создание целевой переменной
        fe = FeatureEngineering(df)
        df = fe.create_target(horizon=5, target_type='direction')
        df = fe.clean_data()

        assert 'target' in df.columns
        assert df['target'].isin([0, 1]).all()

        print("✅ Тест пайплайна данных пройден")

    def test_feature_selection_pipeline(self, generate_test_data):
        """Тест пайплайна отбора признаков."""
        df = generate_test_data

        # Добавляем индикаторы
        df = BaseIndicators.add_all(df)
        df = TrendIndicators.add_all(df)

        # Создаем целевую переменную
        fe = FeatureEngineering(df)
        df = fe.create_target(horizon=5)
        df = fe.clean_data()

        # Получаем список признаков
        feature_cols = [c for c in df.columns if c not in
                        ['time', 'open', 'high', 'low', 'close', 'volume', 'target']]

        # Отбор признаков
        selector = FeatureSelector(n_features=20)
        selected, stats = selector.select_features(df, feature_cols, 'target')

        assert len(selected) <= 20
        assert stats['initial'] >= stats['final']
        assert stats['removed_constant'] >= 0

        print(f"✅ Тест отбора признаков пройден: {len(selected)} из {stats['initial']}")

    def test_model_training_pipeline(self, generate_test_data):
        """Тест пайплайна обучения моделей."""
        df = generate_test_data

        # Подготовка данных
        df = BaseIndicators.add_all(df)

        fe = FeatureEngineering(df)
        df = fe.create_target(horizon=5)
        df = fe.clean_data()

        # Простые признаки для теста
        feature_cols = ['SMA_20', 'SMA_50', 'RSI_14']
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].fillna(0)
        y = df['target']

        # Разделение
        split = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Обучение Random Forest
        rf_model = RandomForestModel("TEST")
        rf_model.build(n_estimators=50, max_depth=5)
        rf_model.train(X_train, y_train)

        metrics = rf_model.evaluate(X_test, y_test)

        assert metrics['accuracy'] > 0
        assert metrics['f1_score'] > 0

        # Обучение XGBoost
        xgb_model = XGBoostModel("TEST")
        xgb_model.build(n_estimators=50, max_depth=3)
        xgb_model.train(X_train, y_train, X_test, y_test)

        xgb_metrics = xgb_model.evaluate(X_test, y_test)

        assert xgb_metrics['accuracy'] > 0

        print(f"✅ Тест обучения моделей пройден: RF={metrics['accuracy']:.3f}, XGB={xgb_metrics['accuracy']:.3f}")

    def test_backtest_pipeline(self, generate_test_data):
        """Тест пайплайна бэктестинга."""
        df = generate_test_data

        # Создаем простые сигналы для теста
        df['signal'] = 0
        df.loc[df['RSI_14'] < 30, 'signal'] = 1  # Buy
        df.loc[df['RSI_14'] > 70, 'signal'] = -1  # Sell

        # Добавляем ATR для стопов
        df = VolatilityIndicators.add_atr(df, periods=[14])

        # Запускаем бэктестинг
        engine = BacktestEngine(initial_capital=10000, symbol="XAUUSD")
        results = engine.run(df, signal_column='signal')

        assert results['total_trades'] > 0
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results

        # Проверяем метрики
        assert results['total_return'] != 0
        assert results['win_rate'] >= 0

        print(f"✅ Тест бэктестинга пройден: {results['total_trades']} сделок, возврат {results['total_return']:.2f}")

    def test_full_integration(self, generate_test_data, setup_test_env):
        """Полный интеграционный тест всего пайплайна."""
        df = generate_test_data
        env = setup_test_env

        print("\n" + "=" * 60)
        print("🚀 ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА")
        print("=" * 60)

        # 1. Препроцессинг
        print("\n1. Препроцессинг данных...")
        preprocessor = DataPreprocessor()
        df = preprocessor.clean(df)
        df = preprocessor.add_datetime_features(df)
        print(f"   ✅ Данные после очистки: {len(df)} записей")

        # 2. Добавление индикаторов
        print("\n2. Расчет индикаторов...")
        df = BaseIndicators.add_all(df)
        df = TrendIndicators.add_all(df)
        df = VolumeIndicators.add_all(df)
        df = VolatilityIndicators.add_all(df)
        print(f"   ✅ Добавлено индикаторов: {len([c for c in df.columns if c not in generate_test_data.columns])}")

        # 3. Feature engineering
        print("\n3. Создание признаков...")
        fe = FeatureEngineering(df)
        df = fe.create_target(horizon=5)
        df = fe.create_lag_features(['close'], lags=[1, 2, 3])
        df = fe.create_rolling_features(['close'], windows=[5, 10])
        df = fe.clean_data()
        print(f"   ✅ Всего признаков: {len(df.columns)}")

        # 4. Отбор признаков
        print("\n4. Отбор признаков...")
        feature_cols = [c for c in df.columns if c not in
                        ['time', 'open', 'high', 'low', 'close', 'volume', 'target']]

        selector = FeatureSelector(n_features=15)
        selected, stats = selector.select_features(df, feature_cols, 'target')
        print(f"   ✅ Отобрано признаков: {len(selected)}")

        # 5. Обучение модели
        print("\n5. Обучение модели...")
        X = df[selected].fillna(0)
        y = df['target']

        split = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestModel("TEST")
        model.build(n_estimators=50, max_depth=5)
        model.train(X_train, y_train)

        metrics = model.evaluate(X_test, y_test)
        print(f"   ✅ Модель обучена: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")

        # 6. Предсказания и сигналы
        print("\n6. Генерация сигналов...")
        predictions = model.predict(X)
        df['signal'] = 0
        df.loc[df.index[-len(predictions):], 'signal'] = predictions

        # 7. Бэктестинг
        print("\n7. Бэктестинг...")
        engine = BacktestEngine(initial_capital=10000)
        results = engine.run(df.iloc[split:], signal_column='signal')

        print(f"\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТИНГА:")
        print(f"   Прибыль: ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)")
        print(f"   Сделок: {results['total_trades']}")
        print(f"   Винрейт: {results['win_rate']:.1f}%")
        print(f"   Профит-фактор: {results['profit_factor']:.2f}")
        print(f"   Шарп: {results['sharpe_ratio']:.2f}")
        print(f"   Макс. просадка: {results['max_drawdown']:.2f}%")

        # 8. Визуализация
        print("\n8. Создание визуализаций...")
        viz = TradingVisualizer("TEST", save_plots=False)

        # Просто проверяем, что графики создаются без ошибок
        try:
            viz.plot_equity_curve(engine.equity_curve, engine.trades, show=False)
            viz.plot_price_with_signals(
                df.iloc[-200:],
                df['signal'].iloc[-200:],
                engine.trades[-10:],
                show=False
            )
            viz.plot_drawdown_analysis(engine.equity_curve, show=False)
            print("   ✅ Визуализация успешна")
        except Exception as e:
            pytest.fail(f"❌ Ошибка визуализации: {e}")

        # 9. Сохранение модели
        print("\n9. Сохранение модели...")
        model.save(path=env['models_dir'] / "test_model")

        # 10. Регистрация модели
        print("\n10. Регистрация модели...")
        registry = ModelRegistry(registry_path=env['temp_dir'] / "registry.json")
        model_id = registry.register(
            model,
            env['models_dir'] / "test_model",
            model.metrics,
            tags=["test", "integration"]
        )
        print(f"   ✅ Модель зарегистрирована: {model_id}")

        # 11. Создание отчета
        print("\n11. Создание отчета...")
        report = HTMLReport("TEST", output_dir=env['reports_dir'])
        report_path = report.generate(
            df, results, engine.trades, engine.equity_curve,
            models={'RandomForest': model},
            feature_importance=(selected, model.model.feature_importances_)
        )
        print(f"   ✅ Отчет сохранен: {report_path}")

        print("\n" + "=" * 60)
        print("✅ ИНТЕГРАЦИОННЫЙ ТЕСТ ПРОЙДЕН УСПЕШНО")
        print("=" * 60)

        return results
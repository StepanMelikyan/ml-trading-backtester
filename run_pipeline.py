# run_pipeline.py
"""
Полный пайплайн для анализа и торговли.
Загружает данные, рассчитывает индикаторы, обучает модели и проводит бэктестинг.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

'''from data.downloader import DataDownloader'''
from data.file_downloader import FileDownloader as DataDownloader
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

from models.ensemble_models import RandomForestModel, XGBoostModel, LightGBMModel, EnsembleModel
from models.deep_learning import LSTMModel
from models.hybrid_model import CNNLSTMHybrid
from models.model_registry import ModelRegistry

from backtest.engine import BacktestEngine
from backtest.metrics import TradingMetrics
from backtest.risk_metrics import AdvancedRiskMetrics

from reports.visualization import TradingVisualizer
from reports.html_report import HTMLReport
from reports.telegram_bot import telegram

from config.settings import SYMBOLS, DEFAULT_TIMEFRAME, DEFAULT_YEARS, BACKTEST_CONFIG
from utils.logger import log
from utils.decorators import timer


class TradingPipeline:
    """
    Главный класс, управляющий всем пайплайном анализа и торговли.
    """

    def __init__(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME,
                 years: int = DEFAULT_YEARS, use_cache: bool = True,
                 use_lstm: bool = True, quick_mode: bool = False):
        """
        Инициализация пайплайна.

        Args:
            symbol: торговый инструмент
            timeframe: таймфрейм
            years: количество лет истории
            use_cache: использовать ли кэш
            use_lstm: использовать ли LSTM модели
            quick_mode: быстрый режим (меньше данных, простые модели)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.years = years if not quick_mode else min(years, 2)
        self.use_cache = use_cache
        self.use_lstm = use_lstm and not quick_mode
        self.quick_mode = quick_mode

        self.data = None
        self.features = None
        self.selected_features = None
        self.models = {}
        self.best_model = None
        self.backtest_results = None
        self.trades = None
        self.equity_curve = None

        symbol_name = SYMBOLS.get(symbol, {}).get('name', symbol)

        print(f"\n{'=' * 60}")
        print(f"🚀 ЗАПУСК ТОРГОВОГО ПАЙПЛАЙНА: {symbol_name}")
        print(f"{'=' * 60}")
        print(f"Инструмент: {symbol}")
        print(f"Таймфрейм:  {timeframe}")
        print(f"Лет истории: {self.years}")
        print(f"Режим:      {'Быстрый' if quick_mode else 'Полный'}")
        print(f"Кэш:        {'Вкл' if use_cache else 'Выкл'}")
        print(f"LSTM:       {'Вкл' if use_lstm else 'Выкл'}")
        print("=" * 60)

    @timer
    def step1_load_data(self):
        """Шаг 1: Загрузка данных."""
        print("\n📥 ШАГ 1: Загрузка данных")

        downloader = DataDownloader(
            self.symbol,
            timeframe=self.timeframe,
            years=self.years,
            use_cache=self.use_cache
        )
        self.data = downloader.download()

        if self.data is None:
            raise Exception(f"❌ Не удалось загрузить данные для {self.symbol}")

        # Предобработка
        preprocessor = DataPreprocessor()
        self.data = preprocessor.clean(self.data)
        self.data = preprocessor.add_datetime_features(self.data)

        print(f"  ✅ Загружено {len(self.data)} записей")
        print(f"  📅 Период: {self.data['time'].iloc[0]} - {self.data['time'].iloc[-1]}")

        return self

    @timer
    def step2_calculate_indicators(self):
        """Шаг 2: Расчет всех индикаторов."""
        print("\n📊 ШАГ 2: Расчет технических индикаторов")

        # Базовые индикаторы
        print("  Базовые индикаторы...")
        self.data = BaseIndicators.add_all(self.data)

        # Трендовые
        if not self.quick_mode:
            print("  Трендовые индикаторы...")
            self.data = TrendIndicators.add_all(self.data)

        # Объемные
        if 'volume' in self.data.columns and not self.quick_mode:
            print("  Объемные индикаторы...")
            self.data = VolumeIndicators.add_all(self.data)

        # Волатильность
        print("  Индикаторы волатильности...")
        self.data = VolatilityIndicators.add_all(self.data)

        # Осцилляторы
        if not self.quick_mode:
            print("  Осцилляторы...")
            self.data = OscillatorIndicators.add_all(self.data)

        # Паттерны
        if not self.quick_mode:
            print("  Ценовые паттерны...")
            self.data = PricePatterns.add_all(self.data)

        indicator_count = len([c for c in self.data.columns if c not in
                               ['time', 'open', 'high', 'low', 'close', 'volume']])
        print(f"  ✅ Всего индикаторов: {indicator_count}")

        return self

    @timer
    def step3_create_features(self):
        """Шаг 3: Создание признаков для ML."""
        print("\n🔧 ШАГ 3: Создание признаков")

        feature_eng = FeatureEngineering(self.data)

        # Создаем целевую переменную
        horizon = 3 if self.quick_mode else 5
        self.data = feature_eng.create_target(horizon=horizon, target_type='direction')

        # Создаем лаговые признаки
        if not self.quick_mode:
            self.data = feature_eng.create_lag_features(
                ['close'], lags=[1, 2, 3, 5]
            )

            # Создаем скользящие статистики
            self.data = feature_eng.create_rolling_features(
                ['close'], windows=[5, 10], functions=['mean', 'std']
            )

        # Очищаем от NaN
        self.data = feature_eng.clean_data()

        print(f"  ✅ Всего признаков: {len(self.data.columns)}")
        print(f"  🎯 Целевая переменная: target (горизонт={horizon})")

        return self

    @timer
    def step4_select_features(self):
        """Шаг 4: Отбор лучших признаков."""
        print("\n🎯 ШАГ 4: Отбор признаков")

        # Получаем список всех признаков
        exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [c for c in self.data.columns if c not in exclude_cols]

        if len(feature_cols) == 0:
            print("  ⚠️ Нет признаков для отбора")
            self.selected_features = []
            return self

        # Отбор признаков
        n_features = 15 if self.quick_mode else 30
        selector = FeatureSelector(n_features=n_features)
        selected_features, stats = selector.select_features(
            self.data, feature_cols, 'target'
        )

        self.selected_features = selected_features
        self.feature_selector = selector

        print(f"\n  ✅ Отобрано {len(selected_features)} признаков из {stats['initial']}")

        return self

    @timer
    def step5_train_models(self):
        """Шаг 5: Обучение моделей."""
        print("\n🤖 ШАГ 5: Обучение моделей")

        if not self.selected_features:
            print("  ⚠️ Нет признаков для обучения")
            return self

        # Подготовка данных
        X = self.data[self.selected_features].fillna(0)
        y = self.data['target']

        # Разделение на train/val/test
        train_size = int(len(X) * 0.6)
        val_size = int(len(X) * 0.2)

        X_train = X.iloc[:train_size]
        X_val = X.iloc[train_size:train_size + val_size]
        X_test = X.iloc[train_size + val_size:]

        y_train = y.iloc[:train_size]
        y_val = y.iloc[train_size:train_size + val_size]
        y_test = y.iloc[train_size + val_size:]

        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")

        # Random Forest
        print("\n  🌲 Обучение Random Forest...")
        rf_model = RandomForestModel(self.symbol)
        rf_params = {'n_estimators': 50, 'max_depth': 5} if self.quick_mode else {}
        rf_model.build(**rf_params)
        rf_model.train(X_train, y_train)
        rf_metrics = rf_model.evaluate(X_test, y_test)
        self.models['RandomForest'] = rf_model

        # XGBoost (если не быстрый режим)
        if not self.quick_mode:
            print("\n  🚀 Обучение XGBoost...")
            xgb_model = XGBoostModel(self.symbol)
            xgb_params = {'n_estimators': 50, 'max_depth': 3} if self.quick_mode else {}
            xgb_model.build(**xgb_params)
            xgb_model.train(X_train, y_train, X_val, y_val)
            xgb_metrics = xgb_model.evaluate(X_test, y_test)
            self.models['XGBoost'] = xgb_model

        # LightGBM
        if not self.quick_mode:
            print("\n  ⚡ Обучение LightGBM...")
            lgb_model = LightGBMModel(self.symbol)
            lgb_model.build()
            lgb_model.train(X_train, y_train, X_val, y_val)
            lgb_metrics = lgb_model.evaluate(X_test, y_test)
            self.models['LightGBM'] = lgb_model

        # Ансамбль (если есть несколько моделей)
        if len(self.models) > 1:
            print("\n  🔀 Создание ансамбля...")
            ensemble = EnsembleModel(self.symbol)
            for name, model in self.models.items():
                ensemble.add_model(model, weight=1.0 / len(self.models))
            ensemble.train(X_train, y_train)
            ensemble_metrics = ensemble.evaluate(X_test, y_test)
            self.models['Ensemble'] = ensemble

        # Выбираем лучшую модель по F1-score
        best_model_name = max(self.models.keys(),
                              key=lambda name: self.models[name].metrics.get('f1_score', 0))
        self.best_model = self.models[best_model_name]

        print(f"\n  🏆 Лучшая модель: {best_model_name}")
        print(f"     F1-Score: {self.best_model.metrics['f1_score']:.4f}")
        print(f"     Accuracy: {self.best_model.metrics['accuracy']:.4f}")

        return self

    @timer
    def step6_backtest(self):
        """Шаг 6: Бэктестинг лучшей модели."""
        print("\n📈 ШАГ 6: Бэктестинг")

        if self.best_model is None:
            print("  ⚠️ Нет обученной модели")
            return self

        # Получаем предсказания на всех данных
        X_all = self.data[self.selected_features].fillna(0)
        predictions = self.best_model.predict(X_all)

        # Добавляем сигналы в DataFrame
        self.data['signal'] = 0
        # Заполняем сигналы, начиная с индекса, где есть все признаки
        start_idx = len(self.data) - len(predictions)
        self.data.iloc[start_idx:, self.data.columns.get_loc('signal')] = predictions

        # Запускаем бэктестинг
        engine = BacktestEngine(
            initial_capital=BACKTEST_CONFIG['initial_capital'],
            commission=BACKTEST_CONFIG['commission'],
            slippage=BACKTEST_CONFIG.get('slippage', 0.0001),
            risk_per_trade=BACKTEST_CONFIG['risk_per_trade'],
            symbol=self.symbol
        )

        # Используем только тестовую часть для бэктестинга
        test_size = int(len(self.data) * 0.2)
        test_data = self.data.iloc[-test_size:].copy()

        self.backtest_results = engine.run(test_data, signal_column='signal')

        # Сохраняем сделки и кривую капитала
        self.trades = engine.trades
        self.equity_curve = engine.equity_curve

        # Выводим основные результаты
        print(f"\n📊 Результаты бэктестинга:")
        print(f"  Прибыль: ${self.backtest_results.get('total_return', 0):.2f}")
        print(f"  Сделок: {self.backtest_results.get('num_trades', 0)}")
        print(f"  Винрейт: {self.backtest_results.get('win_rate', 0):.1f}%")

        return self

    @timer
    def step7_visualize(self):
        """Шаг 7: Визуализация результатов."""
        print("\n📊 ШАГ 7: Визуализация")

        if self.data is None or self.trades is None:
            print("  ⚠️ Нет данных для визуализации")
            return self

        viz = TradingVisualizer(self.symbol)

        # График цены с сигналами
        viz.plot_price_with_signals(
            self.data.iloc[-500:],
            self.data['signal'].iloc[-500:] if 'signal' in self.data.columns else None,
            self.trades[-30:] if self.trades else []
        )

        # Кривая капитала
        if self.equity_curve:
            viz.plot_equity_curve(self.equity_curve, self.trades)

        # Анализ просадок
        if self.equity_curve:
            viz.plot_drawdown_analysis(self.equity_curve)

        # Распределение сделок
        if self.trades:
            viz.plot_trades_distribution(self.trades)

        # Важность признаков
        if self.best_model and self.selected_features:
            importance = self.best_model.get_feature_importance()
            if importance is not None:
                viz.plot_feature_importance(
                    importance['feature'].values,
                    importance['importance'].values
                )

        return self

    @timer
    def step8_generate_report(self):
        """Шаг 8: Генерация отчета."""
        print("\n📝 ШАГ 8: Генерация отчета")

        if self.backtest_results is None:
            print("  ⚠️ Нет результатов для отчета")
            return self

        # Печать метрик
        TradingMetrics.print_report(self.backtest_results, self.symbol)

        # Сохранение HTML отчета
        report = HTMLReport(self.symbol)
        report_path = report.generate(
            df=self.data,
            metrics=self.backtest_results,
            trades=self.trades if self.trades else [],
            equity_curve=self.equity_curve if self.equity_curve else [],
            models=self.models,
            feature_importance=None
        )

        # Отправка в Telegram (если настроено)
        try:
            if telegram.bot:
                # Отправляем уведомление о завершении
                summary = self.backtest_results
                telegram.send_daily_report({
                    'balance': summary.get('final_capital', 0),
                    'daily_pnl': summary.get('total_return', 0),
                    'trades_today': summary.get('num_trades', 0),
                    'win_rate': summary.get('win_rate', 0) / 100,
                    'drawdown': summary.get('max_drawdown', 0),
                    'sharpe': summary.get('sharpe_ratio', 0)
                })

                # Отправляем отчет
                if report_path.exists():
                    import asyncio
                    asyncio.create_task(telegram.send_document(
                        report_path,
                        f"📊 Отчет {self.symbol}"
                    ))
        except Exception as e:
            log.warning(f"Не удалось отправить в Telegram: {e}")

        # Сохранение лучшей модели
        if self.best_model:
            self.best_model.save()

        print(f"\n  ✅ Отчет сохранен: {report_path}")

        return self

    def run(self):
        """Запуск полного пайплайна."""
        try:
            self.step1_load_data()
            self.step2_calculate_indicators()
            self.step3_create_features()
            self.step4_select_features()
            self.step5_train_models()
            self.step6_backtest()
            self.step7_visualize()
            self.step8_generate_report()

            print(f"\n{'=' * 60}")
            print("✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН")
            print(f"{'=' * 60}")

            return self.backtest_results

        except KeyboardInterrupt:
            print("\n\n⚠️ Прервано пользователем")
            return None

        except Exception as e:
            log.exception(f"❌ Ошибка в пайплайне: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Главная функция для запуска пайплайна."""
    import argparse

    parser = argparse.ArgumentParser(description='Запуск торгового пайплайна')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Торговый инструмент')
    parser.add_argument('--timeframe', type=str, default='H1', help='Таймфрейм')
    parser.add_argument('--years', type=int, default=5, help='Лет истории')
    parser.add_argument('--quick', action='store_true', help='Быстрый режим')
    parser.add_argument('--no-cache', action='store_true', help='Не использовать кэш')
    parser.add_argument('--no-lstm', action='store_true', help='Не использовать LSTM')

    args = parser.parse_args()

    # Запуск пайплайна
    pipeline = TradingPipeline(
        symbol=args.symbol,
        timeframe=args.timeframe,
        years=args.years,
        use_cache=not args.no_cache,
        use_lstm=not args.no_lstm,
        quick_mode=args.quick
    )

    results = pipeline.run()

    if results:
        print("\n🎯 Итоговые результаты:")
        print(f"  Прибыль: ${results.get('total_return', 0):.2f}")
        print(f"  Сделок: {results.get('num_trades', 0)}")
        print(f"  Винрейт: {results.get('win_rate', 0):.1f}%")
        print(f"  Шарп: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Макс. просадка: {results.get('max_drawdown', 0):.2f}%")

    return 0


if __name__ == "__main__":
    main()
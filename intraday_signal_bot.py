# intraday_signal_bot.py
"""
Внутридневной торговый бот с МИНУТНЫМ анализом
Модель обучена на H1, но анализирует M1 данные в реальном времени
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
import asyncio
import json
from collections import deque

sys.path.append(str(Path(__file__).parent))

from data.downloader import DataDownloader
from features.base_indicators import BaseIndicators
from features.trend_indicators import TrendIndicators
from features.volume_indicators import VolumeIndicators
from features.volatility import VolatilityIndicators
from features.oscillators import OscillatorIndicators
from features.price_patterns import PricePatterns
from features.feature_engineering import FeatureEngineering
from models.ensemble_models import RandomForestModel, XGBoostModel, LightGBMModel
from reports.telegram_bot import telegram
from config.settings import SYMBOLS
from utils.logger import log


class IntradaySignalBot:
    """
    Внутридневной бот для МИНУТНОГО анализа рынка
    Модель обучена на H1, но применяется к M1 данным
    """

    def __init__(self, symbol="XAUUSD", model_type="RandomForest",
                 confidence_threshold=0.75, min_trade_interval=15,
                 use_real_mt5=True):
        """
        Инициализация внутридневного бота

        Args:
            symbol: торговый инструмент
            model_type: тип модели
            confidence_threshold: порог уверенности (0.0-1.0)
            min_trade_interval: минимальный интервал между сделками (минуты)
            use_real_mt5: использовать реальный MT5
        """
        self.symbol = symbol
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.min_trade_interval = min_trade_interval
        self.use_real_mt5 = use_real_mt5
        self.symbol_name = SYMBOLS.get(symbol, {}).get('name', symbol)

        # Пути к моделям
        self.models_dir = Path("models/saved")
        self.current_model = None
        self.model_path = None
        self.features_list = None
        self.model_metrics = {}

        # Для отслеживания последних сигналов
        self.last_trade_time = None
        self.last_signal = None
        self.signal_history = deque(maxlen=100)  # храним последние 100 сигналов

        # Для статистики
        self.total_checks = 0
        self.signals_sent = 0
        self.start_time = datetime.now()

        # Загружаем последнюю модель
        self._load_latest_model()

        print(f"\n{'=' * 70}")
        print(f"🚀 ВНУТРИДНЕВНОЙ БОТ ЗАПУЩЕН (МИНУТНЫЙ АНАЛИЗ)")
        print(f"{'=' * 70}")
        print(f"Инструмент: {self.symbol_name}")
        print(f"Модель: {self.model_type} (обучена на H1)")
        print(f"Анализ: M1 (каждую минуту)")
        print(f"Порог уверенности: {self.confidence_threshold * 100:.0f}%")
        print(f"Мин. интервал между сделками: {self.min_trade_interval} мин")
        print(f"Данные: {'Реальные MT5' if use_real_mt5 else 'Тестовые файлы'}")
        print(f"{'=' * 70}\n")

    def _load_latest_model(self):
        """Загружает самую свежую модель для символа"""
        print(f"🔍 Поиск последней модели для {self.symbol}...")

        symbol_models = list(self.models_dir.glob(f"{self.symbol}_{self.model_type}_*"))

        if not symbol_models:
            print(f"❌ МОДЕЛИ НЕ НАЙДЕНЫ!")
            print(f"   Сначала обучите модель: python run_pipeline.py --symbol {self.symbol} --years 1")
            return False

        # Берем самую свежую модель
        latest_model_dir = max(symbol_models, key=lambda p: p.stat().st_ctime)
        self.model_path = latest_model_dir

        # Загружаем метаданные
        metadata_path = latest_model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.features_list = metadata.get('features')
                self.model_metrics = metadata.get('metrics', {})

        # Создаем модель нужного типа
        if self.model_type == "RandomForest":
            self.current_model = RandomForestModel(self.symbol)
        elif self.model_type == "XGBoost":
            self.current_model = XGBoostModel(self.symbol)
        elif self.model_type == "LightGBM":
            self.current_model = LightGBMModel(self.symbol)
        else:
            print(f"❌ Неизвестный тип модели: {self.model_type}")
            return False

        # Загружаем модель
        try:
            self.current_model.load(latest_model_dir)
            print(f"✅ МОДЕЛЬ ЗАГРУЖЕНА: {latest_model_dir.name}")
            print(f"   Accuracy: {self.model_metrics.get('accuracy', 0):.3f}")
            print(f"   F1-Score: {self.model_metrics.get('f1_score', 0):.3f}")
            print(f"   Признаков: {len(self.features_list) if self.features_list else 0}")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def _get_live_data(self, bars_needed=200):
        """
        Загружает последние МИНУТНЫЕ данные для анализа

        Args:
            bars_needed: количество необходимых свечей (200 минут = ~3.3 часа)
        """
        if self.use_real_mt5:
            # Используем реальный MT5
            from data.downloader import DataDownloader
            downloader = DataDownloader(
                self.symbol,
                timeframe="M1",  # МИНУТНЫЕ свечи для реального времени
                years=0.1,
                use_cache=False
            )

            # Загружаем последние бары
            import MetaTrader5 as mt5
            if not mt5.initialize():
                print("❌ Ошибка инициализации MT5")
                return None

            rates = mt5.copy_rates_from_pos(
                self.symbol,
                mt5.TIMEFRAME_M1,  # МИНУТНЫЙ таймфрейм
                0,
                bars_needed
            )
            mt5.shutdown()

            if rates is None or len(rates) == 0:
                print("❌ Не удалось загрузить данные из MT5")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Настраиваем volume для модели
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
                df['real_volume'] = df['tick_volume']  # для признака real_volume
                df['tick_volume'] = df['tick_volume']  # для признака tick_volume

            print(f"📊 Загружено {len(df)} МИНУТНЫХ свечей из MT5")
            print(f"   Период: {df['time'].min()} - {df['time'].max()}")
            return df

        else:
            # Используем файловый загрузчик (для тестов)
            from data.file_downloader import FileDownloader
            downloader = FileDownloader(
                self.symbol,
                timeframe="H1",
                years=1,
                use_cache=False
            )
            df = downloader.download()
            if df is not None:
                print(f"📊 Загружено {len(df)} свечей из файла")
            return df

    def _prepare_features(self, df):
        """Подготавливает признаки для модели на МИНУТНЫХ данных"""
        if df is None or len(df) == 0:
            return None

        df = df.copy()
        original_len = len(df)

        # Добавляем временные признаки
        df['time'] = pd.to_datetime(df['time'])
        df['day_of_month'] = df['time'].dt.day
        df['week_of_year'] = df['time'].dt.isocalendar().week
        df['day_of_week'] = df['time'].dt.dayofweek

        # Базовые индикаторы (на минутных данных)
        print("   Расчёт индикаторов на M1...")
        df = BaseIndicators.add_sma(df, windows=[10, 20, 200])
        df = BaseIndicators.add_rsi(df, periods=[14])
        df = BaseIndicators.add_macd(df)
        df = VolatilityIndicators.add_bollinger_bands(df, period=20, std_dev=2.0)

        # Feature engineering (лаги и скользящие)
        print("   Feature engineering...")
        fe = FeatureEngineering(df)
        df = fe.create_lag_features(['close'], lags=[1, 2, 3, 5])
        df = fe.create_rolling_features(['close'], windows=[5, 10],
                                        functions=['mean', 'std'])

        # Удаляем строки с NaN
        df = df.dropna()

        print(f"   После обработки: {len(df)} строк (было {original_len})")

        # Оставляем только те признаки, которые нужны модели
        if hasattr(self, 'features_list') and self.features_list:
            available_features = [f for f in self.features_list if f in df.columns]
            print(f"   Признаков для модели: {len(available_features)} из {len(self.features_list)}")

            if len(available_features) < len(self.features_list):
                missing = set(self.features_list) - set(available_features)
                print(f"   ⚠️ Отсутствуют: {missing}")

            # Возвращаем все строки, не только последние
            return df[available_features].copy()

        return df.copy()

    def analyze_market(self):
        """
        Анализирует рынок на МИНУТНЫХ данных и возвращает сигнал
        """
        self.total_checks += 1

        # Загружаем данные
        df = self._get_live_data(bars_needed=300)
        if df is None:
            return None, "❌ Нет данных"

        # Сохраняем исходные цены до обработки
        original_prices = df[['time', 'close']].copy()

        # Подготавливаем признаки
        df_features = self._prepare_features(df)
        if df_features is None or len(df_features) == 0:
            return None, "❌ Недостаточно данных для расчета"

        # Берем последние 3 строки признаков
        latest = df_features.iloc[-3:].copy()

        # Берем соответствующие цены из исходных данных
        # Индексы в df_features соответствуют индексам в original_prices после обработки
        last_idx = latest.index[-1]
        current_price = original_prices.loc[last_idx, 'close']
        current_time = original_prices.loc[last_idx, 'time']

        # Получаем предсказания для последних 3 свечей
        predictions = []
        confidences = []

        for i in range(len(latest)):
            try:
                X_i = latest.iloc[i:i + 1]
                pred = self.current_model.predict(X_i)[0]
                proba = self.current_model.predict_proba(X_i)[0]

                if pred == 1:
                    confidence = proba[1] if len(proba) > 1 else proba[0]
                else:
                    confidence = proba[0] if len(proba) > 1 else 1 - proba[0]

                predictions.append(pred)
                confidences.append(confidence)
            except Exception as e:
                predictions.append(None)
                confidences.append(0)

        # Текущий сигнал
        current_pred = predictions[-1]
        current_conf = confidences[-1]

        # Проверяем интервал между сделками
        can_trade = True
        if self.last_trade_time:
            minutes_since_last = (datetime.now() - self.last_trade_time).total_seconds() / 60
            if minutes_since_last < self.min_trade_interval:
                can_trade = False

        # Определяем тренд
        trend = self._detect_trend(predictions, confidences)

        # Формируем результат
        result = {
            'current_time': current_time,
            'last_price': current_price,
            'prediction': current_pred,
            'confidence': current_conf,
            'predictions_3': predictions,
            'confidences_3': confidences,
            'can_trade': can_trade,
            'trend': trend,
            'check_number': self.total_checks
        }

        self.signal_history.append(result)

        # Краткий лог
        signal_str = "BUY" if current_pred == 1 else "SELL"
        conf_pct = (current_conf * 100) if current_conf else 0
        print(f"[{self.total_checks:4d}] {current_time.strftime('%H:%M:%S')} - {signal_str} @ {current_price:.2f} | "
              f"Уверенность: {conf_pct:.1f}% | {trend}")

        return result, "✅ Анализ завершен"


    def _detect_trend(self, predictions, confidences):
        """Определяет тренд на основе последовательности сигналов"""
        if len(predictions) < 2:
            return "NEUTRAL"

        # Убираем None
        valid = [(p, c) for p, c in zip(predictions, confidences) if p is not None]

        if len(valid) < 2:
            return "NEUTRAL"

        # Проверяем последовательность
        if all(p == 1 for p, _ in valid) and all(c > 0.6 for _, c in valid):
            return "STRONG_UPTREND"
        elif all(p == 0 for p, _ in valid) and all(c > 0.6 for _, c in valid):
            return "STRONG_DOWNTREND"
        elif valid[-1][0] == 1 and valid[-2][0] == 1:
            return "UPTREND"
        elif valid[-1][0] == 0 and valid[-2][0] == 0:
            return "DOWNTREND"
        else:
            return "NEUTRAL"

    def format_signal_message(self, signal):
        """Форматирует сигнал для Telegram"""
        if signal is None:
            return "❌ Нет сигнала"

        action = "ПОКУПКА" if signal['prediction'] == 1 else "ПРОДАЖА"
        emoji = "🟢" if signal['prediction'] == 1 else "🔴"

        confidence_val = signal['confidence'] if signal['confidence'] else 0

        # Статус сделки
        if signal['can_trade'] and confidence_val >= self.confidence_threshold:
            trade_status = "✅ МОЖНО ТОРГОВАТЬ"
        else:
            trade_status = "⏺ ОЖИДАНИЕ"

        # Цвет уверенности
        if confidence_val > 0.8:
            conf_color = "🟢"
        elif confidence_val > 0.6:
            conf_color = "🟡"
        else:
            conf_color = "🔴"

        # Статистика за сегодня
        today_signals = len([s for s in self.signal_history
                             if s['current_time'].date() == datetime.now().date()
                             and s['confidence'] and s['confidence'] >= self.confidence_threshold])

        message = (
            f"{emoji} <b>ТОРГОВЫЙ СИГНАЛ: {self.symbol_name}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Действие:</b> {action}\n"
            f"💰 <b>Цена:</b> ${signal['last_price']:.2f}\n"
            f"{conf_color} <b>Уверенность:</b> {confidence_val * 100:.1f}%\n"
            f"📈 <b>Тренд:</b> {signal['trend']}\n"
            f"⏰ <b>Время:</b> {signal['current_time'].strftime('%H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{trade_status}\n"
            f"🤖 <b>Модель:</b> {self.model_type} (H1 → M1)\n"
            f"📊 <b>Проверок сегодня:</b> {self.total_checks}\n"
            f"🎯 <b>Сигналов сегодня:</b> {today_signals}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <i>Риск-менеджмент обязателен! SL/TP</i>"
        )

        return message

    def check_and_send_signal(self):
        """Проверяет рынок и отправляет сигнал если уверенность высокая"""
        signal, status = self.analyze_market()

        if signal is None:
            return False

        confidence_val = signal['confidence'] if signal['confidence'] else 0

        # Отправляем только если уверенность выше порога
        if confidence_val >= self.confidence_threshold:
            # Проверяем, не отправляли ли такой же сигнал недавно
            same_as_last = (self.last_signal == signal['prediction'] and
                            self.last_trade_time and
                            (datetime.now() - self.last_trade_time).total_seconds() < 300)  # 5 минут

            if not same_as_last or confidence_val > 0.9:  # Очень уверенные сигналы дублируем
                message = self.format_signal_message(signal)

                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(telegram.send_message(message))
                    loop.close()

                    self.signals_sent += 1
                    print(f"✅ СИГНАЛ ОТПРАВЛЕН! ({self.signals_sent})")

                    # Обновляем время последней сделки
                    if signal['can_trade']:
                        self.last_trade_time = datetime.now()
                        self.last_signal = signal['prediction']

                    return True
                except Exception as e:
                    print(f"❌ Ошибка отправки в Telegram: {e}")
        else:
            # Для отладки показываем и слабые сигналы
            if self.total_checks % 10 == 0:  # Каждую 10-ю проверку
                print(f"   (слабый сигнал: {confidence_val * 100:.1f}%)")

        return False

    def print_stats(self):
        """Выводит статистику работы бота"""
        elapsed = datetime.now() - self.start_time
        hours = elapsed.total_seconds() / 3600

        print(f"\n{'=' * 50}")
        print(f"📊 СТАТИСТИКА РАБОТЫ БОТА")
        print(f"{'=' * 50}")
        print(f"Время работы: {elapsed}")
        print(f"Всего проверок: {self.total_checks}")
        print(f"Сигналов отправлено: {self.signals_sent}")
        print(f"Сигналов в час: {self.signals_sent / hours:.2f}")
        print(f"Процент сигналов: {self.signals_sent / self.total_checks * 100:.2f}%")
        print(f"{'=' * 50}\n")

    def run(self):
        """Запускает непрерывный МИНУТНЫЙ мониторинг"""
        print(f"\n🚀 ЗАПУСК МИНУТНОГО МОНИТОРИНГА")
        print(f"Проверка рынка КАЖДУЮ МИНУТУ")
        print(f"Порог уверенности: {self.confidence_threshold * 100:.0f}%")
        print(f"Минимальный интервал между сделками: {self.min_trade_interval} мин")
        print("Нажмите Ctrl+C для остановки\n")

        # Отправляем стартовое сообщение
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(telegram.send_message(
                f"🚀 <b>МИНУТНЫЙ МОНИТОРИНГ ЗАПУЩЕН</b>\n"
                f"Инструмент: {self.symbol_name}\n"
                f"Модель: {self.model_type} (H1) → Анализ M1\n"
                f"Порог уверенности: {self.confidence_threshold * 100:.0f}%\n"
                f"Интервал: 1 минута"
            ))
            loop.close()
        except Exception as e:
            print(f"⚠️ Не удалось отправить стартовое сообщение: {e}")

        try:
            while True:
                self.check_and_send_signal()

                # Каждые 10 минут показываем статистику
                if self.total_checks % 10 == 0:
                    self.print_stats()

                time.sleep(60)  # Проверка КАЖДУЮ МИНУТУ

        except KeyboardInterrupt:
            print("\n\n🛑 Мониторинг остановлен")
            self.print_stats()

            # Отправляем финальное сообщение
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(telegram.send_message(
                    f"🛑 <b>Мониторинг остановлен</b>\n"
                    f"Инструмент: {self.symbol_name}\n"
                    f"Проверок: {self.total_checks}\n"
                    f"Сигналов: {self.signals_sent}"
                ))
                loop.close()
            except:
                pass


def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='Внутридневной сигнальный бот (МИНУТНЫЙ анализ)')
    parser.add_argument('--symbol', '-s', type=str, default='XAUUSD',
                        help='Торговый инструмент')
    parser.add_argument('--model', '-m', type=str, default='RandomForest',
                        choices=['RandomForest', 'XGBoost', 'LightGBM'],
                        help='Тип модели')
    parser.add_argument('--confidence', '-c', type=float, default=0.75,
                        help='Порог уверенности (0.0-1.0)')
    parser.add_argument('--interval', '-i', type=int, default=15,
                        help='Мин. интервал между сделками (минуты)')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Тестовый режим (без MT5, из файлов)')

    args = parser.parse_args()

    bot = IntradaySignalBot(
        symbol=args.symbol,
        model_type=args.model,
        confidence_threshold=args.confidence,
        min_trade_interval=args.interval,
        use_real_mt5=not args.test
    )

    bot.run()


if __name__ == "__main__":
    main()
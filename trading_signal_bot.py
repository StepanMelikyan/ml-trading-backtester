# trading_signal_bot.py
"""
Скрипт для получения торговых сигналов от обученной модели и отправки в Telegram
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys
import asyncio
import json
import joblib

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

# Импортируем ваши модули
from data.downloader import DataDownloader
from data.file_downloader import FileDownloader
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


class SignalBot:
    """
    Бот для получения торговых сигналов от обученной модели
    """

    def __init__(self, symbol="XAUUSD", model_type="RandomForest", use_real_mt5=False):
        """
        Инициализация бота сигналов

        Args:
            symbol: торговый инструмент (XAUUSD, Brent)
            model_type: тип модели (RandomForest, XGBoost, LightGBM)
            use_real_mt5: использовать реальный MT5 или файловые данные
        """
        self.symbol = symbol
        self.model_type = model_type
        self.use_real_mt5 = use_real_mt5
        self.symbol_name = SYMBOLS.get(symbol, {}).get('name', symbol)

        # Пути к моделям
        self.models_dir = Path("models/saved")
        self.current_model = None
        self.model_path = None
        self.features_list = None

        # Загружаем последнюю модель
        self._load_latest_model()

        # Для отслеживания последнего сигнала
        self.last_signal_time = None
        self.last_signal = None

    def _load_latest_model(self):
        """Загружает самую свежую модель для символа"""
        print(f"🔍 Поиск последней модели для {self.symbol}...")

        # Ищем все модели для этого символа
        symbol_models = list(self.models_dir.glob(f"{self.symbol}_{self.model_type}_*"))

        if not symbol_models:
            print(f"❌ Модели для {self.symbol} не найдены!")
            print(f"   Сначала обучите модель через run_pipeline.py")
            return False

        # Сортируем по дате создания (самые свежие первые)
        latest_model_dir = max(symbol_models, key=lambda p: p.stat().st_ctime)
        self.model_path = latest_model_dir

        # Загружаем метаданные
        metadata_path = latest_model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.features_list = metadata.get('features')
                print(f"📊 Модель обучена на {len(self.features_list) if self.features_list else '?'} признаках")

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
            print(f"✅ Модель загружена: {latest_model_dir.name}")
            print(
                f"   Метрики: Accuracy={self.current_model.metrics.get('accuracy', 0):.3f}, F1={self.current_model.metrics.get('f1_score', 0):.3f}")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False

    def _get_latest_data(self, days_back=30):
        """Загружает последние данные для анализа"""
        if self.use_real_mt5:
            # Используем реальный MT5
            downloader = DataDownloader(
                self.symbol,
                timeframe="H1",
                years=days_back / 365,  # конвертируем дни в годы
                use_cache=False
            )
        else:
            # Используем файловый загрузчик
            from data.file_downloader import FileDownloader
            downloader = FileDownloader(
                self.symbol,
                timeframe="H1",
                years=1,
                use_cache=False
            )

        df = downloader.download()
        if df is None:
            return None

        # Берем только последние days_back дней
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df = df[df['time'] >= cutoff_date]

        return df

    def _prepare_features(self, df):
        """Подготавливает признаки для модели"""
        if df is None or len(df) == 0:
            return None

        # Добавляем временные признаки
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])

        # Добавляем все индикаторы (как при обучении)
        df = BaseIndicators.add_all(df)
        df = TrendIndicators.add_all(df)
        df = VolumeIndicators.add_all(df)
        df = VolatilityIndicators.add_all(df)
        df = OscillatorIndicators.add_all(df)
        df = PricePatterns.add_all(df)

        # Feature engineering
        fe = FeatureEngineering(df)
        df = fe.create_lag_features(['close'], lags=[1, 2, 3, 5])
        df = fe.create_rolling_features(['close'], windows=[5, 10])
        df = fe.clean_data(drop_na=True)

        return df

    def get_signal(self):
        """Получает текущий сигнал от модели"""
        # Загружаем последние данные
        df = self._get_latest_data(days_back=30)
        if df is None:
            return None, None, "❌ Нет данных"

        # Подготавливаем признаки
        df_features = self._prepare_features(df)
        if df_features is None or len(df_features) == 0:
            return None, None, "❌ Недостаточно данных для расчета"

        # Берем последнюю строку
        latest = df_features.iloc[-1:].copy()

        # Определяем признаки для модели
        if self.features_list:
            # Используем признаки из метаданных модели
            available_features = [f for f in self.features_list if f in latest.columns]
            X = latest[available_features].fillna(0)
        else:
            # Используем все числовые признаки
            feature_cols = [c for c in latest.columns if c not in ['time', 'target', 'signal']]
            X = latest[feature_cols].fillna(0)

        # Получаем предсказание
        try:
            prediction = self.current_model.predict(X)[0]
            proba = self.current_model.predict_proba(X)[0]

            # Последняя цена
            last_price = df['close'].iloc[-1]
            last_time = df['time'].iloc[-1]

            # Интерпретируем сигнал
            if prediction == 1:
                signal_type = "ПОКУПКА"
                emoji = "🟢"
                confidence = proba[1] if len(proba) > 1 else proba[0]
            else:
                signal_type = "ПРОДАЖА"
                emoji = "🔴"
                confidence = proba[0] if len(proba) > 1 else 1 - proba[0]

            signal_info = {
                'type': signal_type,
                'emoji': emoji,
                'confidence': confidence,
                'price': last_price,
                'time': last_time,
                'prediction': prediction,
                'proba': proba
            }

            return signal_info, df, "✅ Сигнал получен"

        except Exception as e:
            return None, None, f"❌ Ошибка предсказания: {e}"

    def format_signal_message(self, signal_info):
        """Форматирует сигнал для отправки в Telegram"""
        if signal_info is None:
            return "❌ Не удалось получить сигнал"

        message = (
            f"{signal_info['emoji']} <b>ТОРГОВЫЙ СИГНАЛ: {self.symbol_name}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Действие:</b> {signal_info['type']}\n"
            f"🎯 <b>Цена:</b> ${signal_info['price']:.2f}\n"
            f"📈 <b>Уверенность:</b> {signal_info['confidence'] * 100:.1f}%\n"
            f"⏰ <b>Время:</b> {signal_info['time'].strftime('%Y-%m-%d %H:%M')}\n"
            f"🤖 <b>Модель:</b> {self.model_type}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <i>Торгуйте ответственно, используйте стоп-лосс!</i>"
        )

        return message

    def send_signal(self):
        """Получает сигнал и отправляет в Telegram"""
        signal_info, df, status = self.get_signal()

        if signal_info is None:
            print(status)
            return False

        # Проверяем, не отправляли ли мы такой сигнал недавно
        current_time = datetime.now()
        if (self.last_signal_time and
                self.last_signal == signal_info['prediction'] and
                (current_time - self.last_signal_time).total_seconds() < 3600):  # не чаще раза в час
            print(f"⏺ Сигнал {signal_info['type']} уже отправлялся недавно, пропускаем")
            return False

        # Форматируем и отправляем сообщение
        message = self.format_signal_message(signal_info)

        # Отправляем в Telegram
        try:
            # Используем асинхронный вызов
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(telegram.send_message(message))
            loop.close()

            print(f"✅ Сигнал отправлен в Telegram: {signal_info['type']} @ {signal_info['price']:.2f}")

            # Обновляем последний сигнал
            self.last_signal_time = current_time
            self.last_signal = signal_info['prediction']

            return True

        except Exception as e:
            print(f"❌ Ошибка отправки в Telegram: {e}")
            return False

    def run_once(self):
        """Запускает однократную проверку сигнала"""
        print(f"\n{'=' * 50}")
        print(f"📡 ПРОВЕРКА СИГНАЛА ДЛЯ {self.symbol_name}")
        print(f"{'=' * 50}")
        print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.current_model is None:
            print("❌ Модель не загружена!")
            return

        self.send_signal()

    def run_continuous(self, interval_minutes=60):
        """
        Запускает непрерывный мониторинг сигналов

        Args:
            interval_minutes: интервал проверки в минутах
        """
        print(f"\n{'=' * 50}")
        print(f"🚀 ЗАПУСК МОНИТОРИНГА СИГНАЛОВ")
        print(f"{'=' * 50}")
        print(f"Инструмент: {self.symbol_name}")
        print(f"Модель: {self.model_type}")
        print(f"Интервал: {interval_minutes} минут")
        print(f"{'=' * 50}\n")

        if self.current_model is None:
            print("❌ Модель не загружена!")
            return

        # Отправляем тестовое сообщение
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(telegram.send_message(
                f"🚀 <b>Мониторинг сигналов запущен</b>\n"
                f"Инструмент: {self.symbol_name}\n"
                f"Модель: {self.model_type}\n"
                f"Интервал: {interval_minutes} мин"
            ))
            loop.close()
        except Exception as e:
            print(f"⚠️ Не удалось отправить стартовое сообщение: {e}")

        try:
            while True:
                self.send_signal()
                print(f"⏳ Следующая проверка через {interval_minutes} минут...")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\n🛑 Мониторинг остановлен пользователем")

            # Отправляем сообщение об остановке
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(telegram.send_message(
                    f"🛑 <b>Мониторинг сигналов остановлен</b>\n"
                    f"Инструмент: {self.symbol_name}"
                ))
                loop.close()
            except:
                pass


def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='Торговый сигнальный бот')
    parser.add_argument('--symbol', '-s', type=str, default='XAUUSD',
                        help='Торговый инструмент (XAUUSD, Brent)')
    parser.add_argument('--model', '-m', type=str, default='RandomForest',
                        choices=['RandomForest', 'XGBoost', 'LightGBM'],
                        help='Тип модели')
    parser.add_argument('--interval', '-i', type=int, default=60,
                        help='Интервал проверки в минутах')
    parser.add_argument('--once', '-o', action='store_true',
                        help='Однократная проверка (без цикла)')
    parser.add_argument('--real-mt5', action='store_true',
                        help='Использовать реальный MT5 (иначе файлы)')

    args = parser.parse_args()

    # Создаем бота
    bot = SignalBot(
        symbol=args.symbol,
        model_type=args.model,
        use_real_mt5=args.real_mt5
    )

    if args.once:
        bot.run_once()
    else:
        bot.run_continuous(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
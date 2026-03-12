# dynamic_level_bot.py
"""
Многотаймфреймовый бот с динамическими уровнями входа
- Анализирует тренд на H4/D1
- Вычисляет зону входа на H1
- Ждет цену в зоне на M15
- Динамически обновляет уровень
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
import MetaTrader5 as mt5  

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


class DynamicLevelBot:
    """
    Бот с динамическими уровнями входа и иерархическим анализом
    """

    def __init__(self, symbol="XAUUSD",
                 senior_tf="H4",  # для тренда
                 medium_tf="H1",  # для зоны входа
                 junior_tf="M15",  # для ожидания цены
                 zone_width_pct=0.3,  # ширина зоны в % от цены
                 confidence_threshold=0.75,
                 use_real_mt5=True):

        self.symbol = symbol
        self.senior_tf = senior_tf
        self.medium_tf = medium_tf
        self.junior_tf = junior_tf
        self.zone_width_pct = zone_width_pct  # ширина зоны в процентах
        self.confidence_threshold = confidence_threshold
        self.use_real_mt5 = use_real_mt5
        self.symbol_name = SYMBOLS.get(symbol, {}).get('name', symbol)

        # Состояние бота
        self.current_trend = None  # текущий тренд (BUY/SELL/NEUTRAL)
        self.trend_confidence = 0.0  # уверенность в тренде
        self.entry_zone = None  # зона входа {'low': 100, 'high': 101}
        self.zone_price = None  # целевая цена
        self.zone_updated_time = None  # когда последний раз обновляли зону
        self.waiting_for_entry = False  # ждем ли вход в зону
        self.last_check_time = None  # время последней проверки

        # Модели для каждого TF
        self.models = {}
        self.features_lists = {}

        # Статистика
        self.total_checks = 0
        self.signals_sent = 0
        self.zones_created = 0
        self.zones_cancelled = 0
        self.start_time = datetime.now()

        # История для отладки
        self.zone_history = deque(maxlen=20)
        self.signal_history = deque(maxlen=100)

        # Загружаем модели
        self._load_models()

        self._print_startup_info()

    def _load_models(self):
        """Загружает модели для всех таймфреймов"""
        print(f"\n🔍 ЗАГРУЗКА МОДЕЛЕЙ")
        print(f"=" * 50)

        tf_list = [self.senior_tf, self.medium_tf, self.junior_tf]
        tf_names = {
            "H4": "СТАРШИЙ (тренд)",
            "H1": "СРЕДНИЙ (зона входа)",
            "M15": "МЛАДШИЙ (точка входа)",
            "M5": "МЛАДШИЙ (точка входа)"
        }

        for tf in tf_list:
            print(f"\n📊 {tf_names.get(tf, tf)} [{tf}]:")

            # Ищем модель для этого таймфрейма
            pattern = f"{self.symbol}_{tf}_RandomForest_*"
            model_dirs = list(Path("models/saved").glob(pattern))

            if not model_dirs:
                print(f"   ⚠️ Модель не найдена, будет использован упрощенный анализ")
                self.models[tf] = None
                continue

            latest = max(model_dirs, key=lambda p: p.stat().st_ctime)

            # Загружаем метаданные
            meta_path = latest / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    self.features_lists[tf] = meta.get('features', [])

            # Создаем и загружаем модель
            model = RandomForestModel(self.symbol)
            try:
                model.load(latest)
                self.models[tf] = model
                print(f"   ✅ Модель загружена: {latest.name}")
                print(f"   Accuracy: {model.metrics.get('accuracy', 0):.3f}")
            except Exception as e:
                print(f"   ❌ Ошибка загрузки: {e}")
                self.models[tf] = None

    def _print_startup_info(self):
        """Выводит информацию о запуске"""
        print(f"\n{'=' * 60}")
        print(f"🚀 БОТ С ДИНАМИЧЕСКИМИ УРОВНЯМИ ЗАПУЩЕН")
        print(f"{'=' * 60}")
        print(f"Инструмент: {self.symbol_name}")
        print(f"Старший TF (тренд): {self.senior_tf}")
        print(f"Средний TF (зона): {self.medium_tf}")
        print(f"Младший TF (вход): {self.junior_tf}")
        print(f"Ширина зоны: ±{self.zone_width_pct}%")
        print(f"Порог уверенности: {self.confidence_threshold * 100:.0f}%")
        print(f"Данные: {'Реальные MT5' if self.use_real_mt5 else 'Тестовые'}")
        print(f"{'=' * 60}\n")

    def _get_tf_data(self, tf, bars_needed=200):
        """Загружает данные для конкретного таймфрейма"""
        import MetaTrader5 as mt5

        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }

        if not mt5.initialize():
            print(f"❌ Ошибка инициализации MT5 для {tf}")
            return None

        try:
            rates = mt5.copy_rates_from_pos(
                self.symbol,
                tf_map.get(tf, mt5.TIMEFRAME_H1),
                0,
                bars_needed
            )
        except Exception as e:
            print(f"❌ Ошибка загрузки данных для {tf}: {e}")
            rates = None

        mt5.shutdown()

        if rates is None or len(rates) == 0:
            print(f"❌ Нет данных для {tf}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        if 'tick_volume' in df.columns:
            df['volume'] = df['tick_volume']
            df['tick_volume'] = df['tick_volume']

        return df

    def _prepare_features(self, df, tf):
        """Подготавливает признаки для конкретного таймфрейма"""
        if df is None or len(df) == 0:
            return None

        df = df.copy()

        # Временные признаки
        df['time'] = pd.to_datetime(df['time'])
        df['day_of_month'] = df['time'].dt.day
        df['week_of_year'] = df['time'].dt.isocalendar().week
        df['day_of_week'] = df['time'].dt.dayofweek

        # Индикаторы
        df = BaseIndicators.add_sma(df, windows=[10, 20, 50])
        df = BaseIndicators.add_rsi(df, periods=[14])
        df = BaseIndicators.add_macd(df)
        df = VolatilityIndicators.add_bollinger_bands(df)

        # Feature engineering
        fe = FeatureEngineering(df)
        df = fe.create_lag_features(['close'], lags=[1, 2, 3])
        df = fe.create_rolling_features(['close'], windows=[5, 10])
        df = df.dropna()

        # Оставляем только нужные признаки
        if tf in self.features_lists and self.features_lists[tf]:
            avail = [f for f in self.features_lists[tf] if f in df.columns]
            return df[avail]

        return df

    def _analyze_trend(self):
        """
        Анализирует тренд на старшем таймфрейме (H4/D1)
        Возвращает: направление тренда и уверенность
        """
        df = self._get_tf_data(self.senior_tf, bars_needed=50)
        if df is None:
            return None, 0

        if self.models[self.senior_tf] is None:
            # Упрощенный анализ по SMA
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            current = df['close'].iloc[-1]

            if current > sma_50 and sma_20 > sma_50:
                return "BUY", 0.7
            elif current < sma_50 and sma_20 < sma_50:
                return "SELL", 0.7
            else:
                return "NEUTRAL", 0.5

        # ML анализ
        features = self._prepare_features(df, self.senior_tf)
        if features is None or len(features) == 0:
            return None, 0

        last = features.iloc[-1:]
        pred = self.models[self.senior_tf].predict(last)[0]
        proba = self.models[self.senior_tf].predict_proba(last)[0]

        if pred == 1:
            confidence = proba[1] if len(proba) > 1 else proba[0]
            return "BUY", confidence
        else:
            confidence = proba[0] if len(proba) > 1 else proba[0]
            return "SELL", confidence

    def _calculate_entry_zone(self, trend):
        """
        Рассчитывает зону входа на среднем таймфрейме (H1)
        Возвращает: целевую цену и зону входа
        """
        df = self._get_tf_data(self.medium_tf, bars_needed=100)
        if df is None:
            return None, None

        current_price = df['close'].iloc[-1]

        if self.models[self.medium_tf] is None:
            # Упрощенный расчет по поддержке/сопротивлению
            if trend == "BUY":
                # Ищем ближайший уровень поддержки
                lows = df['low'].rolling(20).min()
                support = lows.iloc[-20:].max()  # ближайшая поддержка
                target = support * 1.001  # чуть выше поддержки
            elif trend == "SELL":
                # Ищем ближайший уровень сопротивления
                highs = df['high'].rolling(20).max()
                resistance = highs.iloc[-20:].min()  # ближайшее сопротивление
                target = resistance * 0.999  # чуть ниже сопротивления
            else:
                return None, None
        else:
            # ML расчет (используем предсказание модели)
            features = self._prepare_features(df, self.medium_tf)
            if features is None:
                return None, None

            # Получаем уверенность модели
            last = features.iloc[-1:]
            pred = self.models[self.medium_tf].predict(last)[0]
            proba = self.models[self.medium_tf].predict_proba(last)[0]

            if pred == 1 and trend == "BUY":
                confidence = proba[1]
                # Целевая цена = текущая + ATR * confidence
                atr = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
                target = current_price + atr * 0.3 * confidence
            elif pred == 0 and trend == "SELL":
                confidence = proba[0]
                atr = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
                target = current_price - atr * 0.3 * confidence
            else:
                return None, None

        # Рассчитываем зону входа (±zone_width_pct%)
        zone_width = target * self.zone_width_pct / 100
        zone = {
            'low': target - zone_width,
            'high': target + zone_width,
            'target': target,
            'trend': trend,
            'created_at': datetime.now(),
            'current_price': current_price
        }

        return zone, target

    def _check_entry(self, zone):
        """
        Проверяет, вошла ли цена в зону входа на младшем таймфрейме
        """
        df = self._get_tf_data(self.junior_tf, bars_needed=10)
        if df is None:
            return False

        current_price = df['close'].iloc[-1]

        # Проверяем, в зоне ли цена
        in_zone = zone['low'] <= current_price <= zone['high']

        # Проверяем подтверждение от младшей модели
        if self.models[self.junior_tf] is not None and in_zone:
            features = self._prepare_features(df, self.junior_tf)
            if features is not None:
                last = features.iloc[-1:]
                pred = self.models[self.junior_tf].predict(last)[0]
                proba = self.models[self.junior_tf].predict_proba(last)[0]

                if zone['trend'] == "BUY" and pred == 1:
                    confidence = proba[1]
                    if confidence >= self.confidence_threshold:
                        return True
                elif zone['trend'] == "SELL" and pred == 0:
                    confidence = proba[0]
                    if confidence >= self.confidence_threshold:
                        return True

        return in_zone  # если нет модели, просто по зоне

    def analyze_market(self):
        """
        Полный анализ рынка на всех таймфреймах
        """
        self.total_checks += 1
        current_time = datetime.now()

        # 1. Анализ тренда на старшем TF
        trend, trend_conf = self._analyze_trend()

        # Обновляем тренд
        trend_changed = (self.current_trend != trend)
        self.current_trend = trend
        self.trend_confidence = trend_conf

        # 2. Если тренд сильный, обновляем зону входа
        if trend and trend_conf >= self.confidence_threshold:
            # Обновляем зону (всегда, потому что рынок меняется)
            new_zone, target = self._calculate_entry_zone(trend)

            if new_zone:
                # Проверяем, изменилась ли зона существенно
                zone_changed = True
                if self.entry_zone:
                    old_target = self.entry_zone.get('target', 0)
                    change_pct = abs(target - old_target) / old_target * 100
                    zone_changed = change_pct > 0.1  # >0.1% изменение

                if zone_changed:
                    self.entry_zone = new_zone
                    self.zone_price = target
                    self.zone_updated_time = current_time
                    self.waiting_for_entry = True
                    self.zones_created += 1

                    # Логируем новую зону
                    print(f"\n🎯 НОВАЯ ЗОНА {trend}:")
                    print(f"   Цель: ${target:.2f} (±{self.zone_width_pct}%)")
                    print(f"   Зона: ${new_zone['low']:.2f} - ${new_zone['high']:.2f}")

        # 3. Проверяем вход в зону
        signal = None
        if self.waiting_for_entry and self.entry_zone:
            if self._check_entry(self.entry_zone):
                signal = {
                    'trend': self.current_trend,
                    'zone': self.entry_zone,
                    'price': self.entry_zone['target'],
                    'time': current_time,
                    'confidence': self.trend_confidence
                }

                self.signals_sent += 1
                self.waiting_for_entry = False
                self.entry_zone = None

        # 4. Проверяем актуальность зоны (не старше 4 часов)
        if self.zone_updated_time:
            age = (current_time - self.zone_updated_time).total_seconds() / 3600
            if age > 4:  # зона устарела через 4 часа
                print(f"⏳ Зона устарела ({age:.1f}ч), ожидаем новую")
                self.waiting_for_entry = False
                self.entry_zone = None
                self.zones_cancelled += 1

        return signal, trend, self.entry_zone

    def format_signal_message(self, signal):
        """Форматирует сигнал для Telegram"""
        if not signal:
            return None

        emoji = "🟢" if signal['trend'] == "BUY" else "🔴"
        action = "ПОКУПКА" if signal['trend'] == "BUY" else "ПРОДАЖА"

        zone = signal['zone']

        message = (
            f"{emoji} <b>СИГНАЛ НА ВХОД: {self.symbol_name}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Действие:</b> {action}\n"
            f"🎯 <b>Целевая цена:</b> ${zone['target']:.2f}\n"
            f"📈 <b>Зона входа:</b> ${zone['low']:.2f} - ${zone['high']:.2f}\n"
            f"💰 <b>Текущая цена:</b> ${zone['current_price']:.2f}\n"
            f"📊 <b>Уверенность:</b> {signal['confidence'] * 100:.1f}%\n"
            f"⏰ <b>Время:</b> {signal['time'].strftime('%H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 <b>Таймфреймы:</b> {self.senior_tf}→{self.medium_tf}→{self.junior_tf}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <i>Стоп-лосс: за пределами зоны</i>"
        )

        return message

    def format_zone_update_message(self, zone, trend):
        """Форматирует сообщение об обновлении зоны"""
        emoji = "🟡" if trend == "BUY" else "🔵"
        action = "ПОКУПКИ" if trend == "BUY" else "ПРОДАЖИ"

        message = (
            f"{emoji} <b>НОВАЯ ЗОНА {action}: {self.symbol_name}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Целевая цена:</b> ${zone['target']:.2f}\n"
            f"📈 <b>Зона входа:</b> ${zone['low']:.2f} - ${zone['high']:.2f}\n"
            f"💰 <b>Текущая цена:</b> ${zone['current_price']:.2f}\n"
            f"⏰ <b>Время:</b> {zone['created_at'].strftime('%H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏳ <i>Ожидаем цену в зоне...</i>"
        )

        return message

    def check_and_notify(self):
        """
        Проверяет рынок и отправляет уведомления
        """
        signal, trend, zone = self.analyze_market()

        # Логируем в консоль
        trend_str = trend if trend else "NEUTRAL"
        zone_str = f"🎯 ${zone['target']:.2f}" if zone else "⏺ нет зоны"
        print(f"[{self.total_checks:4d}] {datetime.now().strftime('%H:%M:%S')} | "
              f"Тренд: {trend_str} | {zone_str}")

        # Отправляем сигнал, если есть
        if signal:
            msg = self.format_signal_message(signal)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(telegram.send_message(msg))
                loop.close()
                print(f"✅ СИГНАЛ ОТПРАВЛЕН!")
            except Exception as e:
                print(f"❌ Ошибка Telegram: {e}")

        # Отправляем обновление зоны (каждые 30 минут или при сильном изменении)
        elif zone and (self.total_checks % 30 == 0 or self.zones_created % 5 == 0):
            msg = self.format_zone_update_message(zone, trend)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(telegram.send_message(msg))
                loop.close()
            except:
                pass

    def print_stats(self):
        """Выводит статистику"""
        elapsed = datetime.now() - self.start_time
        hours = elapsed.total_seconds() / 3600

        print(f"\n{'=' * 50}")
        print(f"📊 СТАТИСТИКА БОТА")
        print(f"{'=' * 50}")
        print(f"Время работы: {elapsed}")
        print(f"Проверок: {self.total_checks}")
        print(f"Зон создано: {self.zones_created}")
        print(f"Зон отменено: {self.zones_cancelled}")
        print(f"Сигналов: {self.signals_sent}")
        print(f"Текущий тренд: {self.current_trend}")
        print(f"{'=' * 50}\n")

    def run(self):
        """Запускает непрерывный мониторинг"""
        print(f"\n🚀 ЗАПУСК МОНИТОРИНГА")
        print(f"Проверка каждую минуту")
        print(f"Нажмите Ctrl+C для остановки\n")

        # Стартовое сообщение
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(telegram.send_message(
                f"🚀 <b>Бот с динамическими уровнями запущен</b>\n"
                f"Инструмент: {self.symbol_name}\n"
                f"Тренд: {self.senior_tf} → Зона: {self.medium_tf} → Вход: {self.junior_tf}\n"
                f"Ширина зоны: ±{self.zone_width_pct}%"
            ))
            loop.close()
        except:
            pass

        try:
            while True:
                self.check_and_notify()

                if self.total_checks % 60 == 0:  # каждый час
                    self.print_stats()

                time.sleep(60)  # проверка каждую минуту

        except KeyboardInterrupt:
            print("\n\n🛑 Остановка бота")
            self.print_stats()

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(telegram.send_message(
                    f"🛑 <b>Бот остановлен</b>\n"
                    f"Проверок: {self.total_checks}\n"
                    f"Сигналов: {self.signals_sent}"
                ))
                loop.close()
            except:
                pass


def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='Бот с динамическими уровнями')
    parser.add_argument('--symbol', '-s', default='XAUUSD', help='Инструмент')
    parser.add_argument('--senior', default='H4', help='Старший TF (тренд)')
    parser.add_argument('--medium', default='H1', help='Средний TF (зона)')
    parser.add_argument('--junior', default='M15', help='Младший TF (вход)')
    parser.add_argument('--zone-width', '-w', type=float, default=0.3,
                        help='Ширина зоны в %')
    parser.add_argument('--confidence', '-c', type=float, default=0.75,
                        help='Порог уверенности')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Тестовый режим')

    args = parser.parse_args()

    bot = DynamicLevelBot(
        symbol=args.symbol,
        senior_tf=args.senior,
        medium_tf=args.medium,
        junior_tf=args.junior,
        zone_width_pct=args.zone_width,
        confidence_threshold=args.confidence,
        use_real_mt5=not args.test
    )

    bot.run()


if __name__ == "__main__":
    main()
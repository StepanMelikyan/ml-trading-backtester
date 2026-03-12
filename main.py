# main.py
"""
Главный скрипт для запуска анализа и бэктестинга.
Поддерживает различные режимы работы.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from run_pipeline import TradingPipeline
from utils.logger import log


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='ML Trading Backtester - анализ и бэктестинг торговых стратегий'
    )

    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='XAUUSD',
        help='Торговый инструмент (XAUUSD, Brent, WTI)'
    )

    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        default='H1',
        choices=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'],
        help='Таймфрейм'
    )

    parser.add_argument(
        '--years', '-y',
        type=int,
        default=5,
        help='Количество лет истории'
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='full',
        choices=['full', 'data', 'features', 'train', 'backtest', 'report'],
        help='Режим работы'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Не использовать кэш'
    )

    parser.add_argument(
        '--no-lstm',
        action='store_true',
        help='Не использовать LSTM модели'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Быстрый режим (меньше данных, простые модели)'
    )

    return parser.parse_args()


def main():
    """Главная функция."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("🚀 ML TRADING BACKTESTER")
    print("=" * 60)
    print(f"Symbol:     {args.symbol}")
    print(f"Timeframe:  {args.timeframe}")
    print(f"Years:      {args.years}")
    print(f"Mode:       {args.mode}")
    print(f"Cache:      {'OFF' if args.no_cache else 'ON'}")
    print("=" * 60)

    start_time = datetime.now()

    try:
        if args.mode == 'full':
            # Полный пайплайн
            pipeline = TradingPipeline(
                symbol=args.symbol,
                timeframe=args.timeframe,
                years=args.years,
                use_cache=not args.no_cache,
                use_lstm=not args.no_lstm,
                quick_mode=args.quick
            )
            results = pipeline.run()

        elif args.mode == 'data':
            # Только загрузка данных
            from data.downloader import DataDownloader
            downloader = DataDownloader(
                args.symbol,
                timeframe=args.timeframe,
                years=args.years,
                use_cache=not args.no_cache
            )
            df = downloader.download()
            if df is not None:
                print(f"\n✅ Данные загружены: {len(df)} записей")
                print(df.head())

        elif args.mode == 'features':
            # Только расчет индикаторов
            from data.downloader import DataDownloader
            from features.base_indicators import BaseIndicators

            downloader = DataDownloader(
                args.symbol,
                timeframe=args.timeframe,
                years=args.years,
                use_cache=not args.no_cache
            )
            df = downloader.download()

            if df is not None:
                df = BaseIndicators.add_all(df)
                print(f"\n✅ Индикаторы рассчитаны: {len(df.columns)} колонок")
                print(
                    f"Новые колонки: {[c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']][:10]}")

        elif args.mode == 'train':
            # Только обучение модели
            from data.downloader import DataDownloader
            from features.base_indicators import BaseIndicators
            from features.feature_engineering import FeatureEngineering
            from models.ensemble_models import RandomForestModel

            downloader = DataDownloader(
                args.symbol,
                timeframe=args.timeframe,
                years=min(args.years, 3),  # Для обучения меньше данных
                use_cache=not args.no_cache
            )
            df = downloader.download()

            if df is not None:
                df = BaseIndicators.add_all(df)

                fe = FeatureEngineering(df)
                df = fe.create_target(horizon=5)
                df = fe.clean_data()

                feature_cols = [c for c in df.columns if c.startswith(('SMA', 'RSI', 'MACD'))]
                X = df[feature_cols].fillna(0)
                y = df['target']

                split = int(len(X) * 0.7)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]

                model = RandomForestModel(args.symbol)
                model.build()
                model.train(X_train, y_train)

                metrics = model.evaluate(X_test, y_test)
                print(f"\n✅ Модель обучена: Accuracy={metrics['accuracy']:.3f}")

        elif args.mode == 'backtest':
            # Только бэктестинг
            from data.downloader import DataDownloader
            from features.base_indicators import BaseIndicators
            from features.volatility import VolatilityIndicators
            from backtest.engine import BacktestEngine

            downloader = DataDownloader(
                args.symbol,
                timeframe=args.timeframe,
                years=args.years,
                use_cache=not args.no_cache
            )
            df = downloader.download()

            if df is not None:
                df = BaseIndicators.add_rsi(df)
                df = VolatilityIndicators.add_atr(df)

                # Простая стратегия на RSI
                df['signal'] = 0
                df.loc[df['RSI_14'] < 30, 'signal'] = 1
                df.loc[df['RSI_14'] > 70, 'signal'] = -1

                engine = BacktestEngine(initial_capital=10000, symbol=args.symbol)
                results = engine.run(df, signal_column='signal')

                from backtest.metrics import TradingMetrics
                TradingMetrics.print_report(results, args.symbol)

        elif args.mode == 'report':
            # Только генерация отчета (из已有的 файлов)
            import pandas as pd
            from reports.html_report import HTMLReport

            # Ищем последний файл с данными
            data_dir = Path("data")
            csv_files = list(data_dir.glob(f"{args.symbol}*.csv"))

            if csv_files:
                latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                df = pd.read_csv(latest_file)

                report = HTMLReport(args.symbol)
                # Здесь нужны trades и metrics, но для простоты создаем заглушку
                report.generate(df, {}, [], [], {})
                print(f"\n✅ Отчет создан")
            else:
                print(f"❌ Файлы не найдены в {data_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
        sys.exit(0)

    except Exception as e:
        log.exception(f"❌ Ошибка: {e}")
        sys.exit(1)

    elapsed = datetime.now() - start_time
    print(f"\n⏱ Время выполнения: {elapsed}")
    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
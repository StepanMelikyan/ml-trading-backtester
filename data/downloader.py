# data/downloader.py
"""
Модуль для загрузки данных с MetaTrader 5.
Поддерживает множественные символы и таймфреймы.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import MT5_CONFIG, TIMEFRAMES
from utils.logger import log
from .cache_manager import CacheManager


class DataDownloader:
    """
    Загрузчик данных с MetaTrader 5 с поддержкой кэширования.
    """

    def __init__(self, symbol: str, timeframe: str = "H1", years: int = 5, use_cache: bool = True):
        """
        Инициализация загрузчика.

        Args:
            symbol: Тикер инструмента (XAUUSD, Brent и т.д.)
            timeframe: Таймфрейм (M1, M5, H1, D1)
            years: Количество лет истории
            use_cache: Использовать ли кэширование
        """
        self.symbol = symbol
        self.timeframe = TIMEFRAMES.get(timeframe, 60)
        self.timeframe_name = timeframe
        self.years = years
        self.use_cache = use_cache
        self.cache = CacheManager() if use_cache else None

    def download(self, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Загружает данные. Если force_download=False, сначала проверяет кэш.

        Returns:
            DataFrame с колонками: time, open, high, low, close, tick_volume, spread
        """
        # Проверяем кэш
        if self.use_cache and not force_download:
            cached_data = self.cache.get(self.symbol, self.timeframe_name, self.years)
            if cached_data is not None:
                log.info(f"📂 Загружено из кэша: {self.symbol} ({len(cached_data)} записей)")
                return cached_data

        # Инициализация MT5
        if not self._initialize_mt5():
            return None

        # Расчет дат
        to_date = datetime.now()
        from_date = to_date - timedelta(days=365 * self.years)

        log.info(f"📥 Загрузка {self.symbol} {self.timeframe_name} за {self.years} лет...")
        log.info(f"   Период: {from_date} - {to_date}")

        # Загрузка данных с повторными попытками
        rates = self._download_with_retry(from_date, to_date)

        mt5.shutdown()

        if rates is None or len(rates) == 0:
            log.error(f"❌ Не удалось загрузить {self.symbol}")
            return None

        # Преобразование в DataFrame
        df = self._rates_to_dataframe(rates)

        # Фильтруем по нужному периоду
        df = df[(df['time'] >= from_date) & (df['time'] <= to_date)]

        # Сохраняем в кэш
        if self.use_cache:
            self.cache.save(df, self.symbol, self.timeframe_name, self.years)

        log.info(f"✅ Загружено {len(df)} записей для {self.symbol}")
        log.info(f"   Период: {df['time'].min()} - {df['time'].max()}")
        return df


    def _initialize_mt5(self) -> bool:
        """Инициализация и авторизация MT5"""

        # Пробуем разные способы инициализации
        if not mt5.initialize():
            # Если не получилось, пробуем с указанием пути
            mt5_path = "C:\\Program Files\\FusionMarkets MT5\\terminal64.exe"
            if not mt5.initialize(path=mt5_path):
                log.error(f"❌ Ошибка инициализации MT5: {mt5.last_error()}")
                return False

        log.info("✅ MT5 инициализирован")

        # Авторизация если нужна
        if MT5_CONFIG['login'] and MT5_CONFIG['password']:
            authorized = mt5.login(
                login=int(MT5_CONFIG['login']),
                password=MT5_CONFIG['password'],
                server=MT5_CONFIG['server']
            )
            if not authorized:
                log.error(f"❌ Ошибка авторизации MT5: {mt5.last_error()}")
                return False
            log.info("✅ Авторизация в MT5 успешна")

        return True

    def test_connection(self):
        """Тест подключения к MT5 и доступности символа"""
        import MetaTrader5 as mt5

        if not mt5.initialize():
            return {"error": f"Не удалось инициализировать MT5: {mt5.last_error()}"}

        # Проверяем символ
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            mt5.shutdown()
            return {"error": f"Символ {self.symbol} не найден"}

        # Проверяем, включен ли символ для торговли
        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)
            symbol_info = mt5.symbol_info(self.symbol)

        # Пробуем получить последние данные
        ticks = mt5.copy_ticks_from(self.symbol, datetime.now(), 100, mt5.COPY_TICKS_ALL)
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 10)

        mt5.shutdown()

        return {
            "symbol": self.symbol,
            "symbol_found": symbol_info is not None,
            "trade_mode": symbol_info.trade_mode if symbol_info else None,
            "spread": symbol_info.spread if symbol_info else None,
            "ticks": len(ticks) if ticks is not None else 0,
            "rates": len(rates) if rates is not None else 0
        }

    def _download_with_retry(self, from_date: datetime, to_date: datetime,
                             max_attempts: int = 3) -> Optional[pd.DataFrame]:
        """Загрузка с повторными попытками - загружаем по частям"""

        # Максимальное количество свечей за один запрос
        MAX_BARS_PER_REQUEST = 5000

        try:
            all_rates = []
            current_date = to_date

            log.info(f"Загружаем данные по частям с {from_date} по {to_date}")

            while current_date > from_date:
                # Загружаем порцию данных
                rates = mt5.copy_rates_from(
                    self.symbol,
                    self.timeframe,
                    current_date,
                    MAX_BARS_PER_REQUEST
                )

                if rates is None or len(rates) == 0:
                    log.warning(f"Не удалось загрузить порцию данных до {current_date}")
                    break

                # Преобразуем в DataFrame для удобства
                df_chunk = pd.DataFrame(rates)
                df_chunk['time'] = pd.to_datetime(df_chunk['time'], unit='s')

                # Добавляем к общему списку
                all_rates.append(df_chunk)

                # Обновляем текущую дату для следующей порции
                current_date = df_chunk['time'].min()

                log.info(f"  Загружено {len(df_chunk)} свечей до {current_date}")

                # Небольшая пауза, чтобы не перегружать сервер
                time.sleep(0.5)

            if all_rates:
                # Объединяем все порции
                final_df = pd.concat(all_rates, ignore_index=True)

                # Удаляем дубликаты и сортируем
                final_df = final_df.drop_duplicates(subset=['time'])
                final_df = final_df.sort_values('time')

                # Фильтруем по нужному периоду
                final_df = final_df[(final_df['time'] >= from_date) & (final_df['time'] <= to_date)]

                log.info(f"✅ Всего загружено {len(final_df)} свечей методом copy_rates_from (по частям)")

                # Преобразуем обратно в формат mt5
                return final_df.to_records(index=False)

        except Exception as e:
            log.warning(f"Ошибка при загрузке по частям: {e}")

        # Если метод с частями не сработал, пробуем старый метод
        for attempt in range(max_attempts):
            try:
                rates = mt5.copy_rates_range(
                    self.symbol,
                    self.timeframe,
                    from_date,
                    to_date
                )

                if rates is not None and len(rates) > 0:
                    log.info(f"✅ Данные получены методом copy_rates_range: {len(rates)} записей")
                    return rates

            except Exception as e:
                log.warning(f"Ошибка при загрузке: {e}")

            log.warning(f"⏳ Попытка {attempt + 1}/{max_attempts} не удалась, повтор через 2 сек...")
            time.sleep(2)

        return None

    def _rates_to_dataframe(self, rates: tuple) -> pd.DataFrame:
        """Преобразование данных MT5 в DataFrame"""
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Переименовываем колонки для удобства
        df = df.rename(columns={
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        })

        return df

    def download_multiple(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Загружает данные для нескольких символов.

        Args:
            symbols: Список тикеров

        Returns:
            Словарь {symbol: DataFrame}
        """
        results = {}
        for symbol in symbols:
            self.symbol = symbol
            df = self.download()
            if df is not None:
                results[symbol] = df
        return results
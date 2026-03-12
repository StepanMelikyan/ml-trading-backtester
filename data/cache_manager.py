# data/cache_manager.py
"""
Менеджер кэширования данных для ускорения повторных запусков.
Сохраняет данные в сжатом формате с метаданными.
"""

import pandas as pd
import pickle
import gzip
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
import hashlib
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DATA_DIR
from utils.logger import log


class CacheManager:
    """
    Управляет кэшированием данных для ускорения повторных запусков.
    Поддерживает сжатие, метаданные и автоматическую очистку.
    """

    def __init__(self, cache_dir: Path = None, compress: bool = True):
        """
        Инициализация менеджера кэша.

        Args:
            cache_dir: Директория для кэша (по умолчанию data/cache)
            compress: Использовать ли сжатие gzip
        """
        self.cache_dir = cache_dir or (DATA_DIR / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress

        log.info(f"📁 Кэш-директория: {self.cache_dir}")

    def _get_cache_key(self, symbol: str, timeframe: str, years: int, **kwargs) -> str:
        """
        Генерирует уникальный ключ для кэша на основе параметров.

        Args:
            symbol: Тикер инструмента
            timeframe: Таймфрейм
            years: Количество лет
            **kwargs: Дополнительные параметры

        Returns:
            Уникальный хеш-ключ
        """
        # Создаем строку из всех параметров
        key_parts = [symbol, timeframe, str(years)]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        key_str = "_".join(key_parts)

        # Создаем хеш
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_paths(self, cache_key: str) -> tuple:
        """
        Возвращает пути к файлам кэша.

        Returns:
            (data_path, meta_path)
        """
        if self.compress:
            data_path = self.cache_dir / f"{cache_key}.pkl.gz"
        else:
            data_path = self.cache_dir / f"{cache_key}.pkl"

        meta_path = self.cache_dir / f"{cache_key}_meta.json"

        return data_path, meta_path

    def get(self, symbol: str, timeframe: str, years: int,
            max_age_hours: int = 24, **kwargs) -> Optional[pd.DataFrame]:
        """
        Получает данные из кэша, если они не устарели.

        Args:
            symbol: Тикер инструмента
            timeframe: Таймфрейм
            years: Количество лет
            max_age_hours: Максимальный возраст кэша в часах
            **kwargs: Дополнительные параметры для ключа

        Returns:
            DataFrame или None, если кэш отсутствует или устарел
        """
        cache_key = self._get_cache_key(symbol, timeframe, years, **kwargs)
        data_path, meta_path = self._get_cache_paths(cache_key)

        # Проверяем существование файлов
        if not data_path.exists() or not meta_path.exists():
            log.debug(f"⏺ Кэш не найден: {cache_key}")
            return None

        # Загружаем метаданные
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception as e:
            log.warning(f"⚠️ Ошибка загрузки метаданных кэша: {e}")
            return None

        # Проверяем возраст кэша
        cache_time = datetime.fromisoformat(meta['cached_at'])
        age = datetime.now() - cache_time

        if age > timedelta(hours=max_age_hours):
            log.debug(f"⏳ Кэш устарел ({age.days}д {age.seconds // 3600}ч)")
            return None

        # Загружаем данные
        try:
            if self.compress:
                with gzip.open(data_path, 'rb') as f:
                    df = pickle.load(f)
            else:
                with open(data_path, 'rb') as f:
                    df = pickle.load(f)

            log.info(f"📦 Загружено из кэша: {symbol} {timeframe} ({len(df)} записей, возраст {age.seconds // 3600}ч)")

            # Проверяем целостность
            if self._validate_data(df, meta):
                return df
            else:
                log.warning("⚠️ Данные кэша повреждены, удаляем...")
                self._delete_cache_files(data_path, meta_path)
                return None

        except Exception as e:
            log.warning(f"⚠️ Ошибка загрузки кэша: {e}")
            return None

    def save(self, df: pd.DataFrame, symbol: str, timeframe: str, years: int, **kwargs):
        """
        Сохраняет данные в кэш.

        Args:
            df: DataFrame для сохранения
            symbol: Тикер инструмента
            timeframe: Таймфрейм
            years: Количество лет
            **kwargs: Дополнительные параметры для ключа
        """
        cache_key = self._get_cache_key(symbol, timeframe, years, **kwargs)
        data_path, meta_path = self._get_cache_paths(cache_key)

        # Создаем метаданные
        meta = {
            'symbol': symbol,
            'timeframe': timeframe,
            'years': years,
            'cached_at': datetime.now().isoformat(),
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['time'].min().isoformat() if 'time' in df.columns else None,
            'end_date': df['time'].max().isoformat() if 'time' in df.columns else None,
            'data_hash': self._calculate_hash(df),
            'kwargs': kwargs
        }

        # Сохраняем данные
        try:
            if self.compress:
                with gzip.open(data_path, 'wb') as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(data_path, 'wb') as f:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Сохраняем метаданные
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            size_mb = data_path.stat().st_size / (1024 * 1024)
            log.info(f"💾 Сохранено в кэш: {symbol} {timeframe} ({len(df)} записей, {size_mb:.2f} MB)")

        except Exception as e:
            log.error(f"❌ Ошибка сохранения кэша: {e}")

    def _calculate_hash(self, df: pd.DataFrame) -> str:
        """Вычисляет хеш данных для проверки целостности"""
        # Берем первые 1000 строк и основные колонки для хеша
        sample = df[['open', 'high', 'low', 'close']].head(1000).values
        return hashlib.md5(sample.tobytes()).hexdigest()

    def _validate_data(self, df: pd.DataFrame, meta: Dict) -> bool:
        """Проверяет целостность данных"""
        # Проверяем количество строк
        if len(df) != meta['rows']:
            return False

        # Проверяем наличие основных колонок
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return False

        # Проверяем, что нет NaN в основных колонках
        if df[required_cols].isnull().any().any():
            return False

        return True

    def _delete_cache_files(self, data_path: Path, meta_path: Path):
        """Удаляет файлы кэша"""
        try:
            if data_path.exists():
                data_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
        except Exception as e:
            log.warning(f"⚠️ Ошибка удаления кэша: {e}")

    def clear_old(self, max_age_days: int = 7) -> int:
        """
        Очищает старые кэш-файлы.

        Args:
            max_age_days: Максимальный возраст в днях

        Returns:
            Количество удаленных файлов
        """
        now = datetime.now()
        removed = 0

        for meta_file in self.cache_dir.glob("*_meta.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                cache_time = datetime.fromisoformat(meta['cached_at'])
                age = now - cache_time

                if age > timedelta(days=max_age_days):
                    # Удаляем meta и соответствующий data файл
                    cache_key = meta_file.stem.replace('_meta', '')

                    if self.compress:
                        data_file = self.cache_dir / f"{cache_key}.pkl.gz"
                    else:
                        data_file = self.cache_dir / f"{cache_key}.pkl"

                    if data_file.exists():
                        data_file.unlink()
                    meta_file.unlink()
                    removed += 1

                    log.debug(f"🗑 Удален устаревший кэш: {cache_key}")

            except Exception as e:
                log.warning(f"⚠️ Ошибка при очистке кэша: {e}")

        if removed > 0:
            log.info(f"🧹 Очищено {removed} устаревших кэш-файлов")

        return removed

    def get_stats(self) -> Dict:
        """
        Возвращает статистику по кэшу.

        Returns:
            Словарь со статистикой
        """
        meta_files = list(self.cache_dir.glob("*_meta.json"))

        if not meta_files:
            return {'total_files': 0, 'total_size_mb': 0}

        total_size = 0
        symbols = set()
        timeframes = set()

        for meta_file in meta_files:
            try:
                cache_key = meta_file.stem.replace('_meta', '')

                if self.compress:
                    data_file = self.cache_dir / f"{cache_key}.pkl.gz"
                else:
                    data_file = self.cache_dir / f"{cache_key}.pkl"

                if data_file.exists():
                    total_size += data_file.stat().st_size

                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    symbols.add(meta.get('symbol', 'unknown'))
                    timeframes.add(meta.get('timeframe', 'unknown'))

            except Exception:
                pass

        return {
            'total_files': len(meta_files),
            'total_size_mb': total_size / (1024 * 1024),
            'symbols': list(symbols),
            'timeframes': list(timeframes)
        }
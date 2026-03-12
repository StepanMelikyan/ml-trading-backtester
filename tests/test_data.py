# tests/test_data.py
"""
Тесты для модулей загрузки и обработки данных.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from data.downloader import DataDownloader
from data.preprocessor import DataPreprocessor
from data.cache_manager import CacheManager


@pytest.fixture
def sample_data():
    """Создает тестовые данные."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    n = len(dates)

    # Генерируем случайные цены
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_price = close.shift(1).fillna(close[0])
    volume = np.random.randint(100, 1000, n)

    df = pd.DataFrame({
        'time': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'tick_volume': volume
    })

    return df


@pytest.fixture
def setup_cache():
    """Создает временную директорию для кэша."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestDataPreprocessor:
    """Тесты для DataPreprocessor."""

    def test_clean_data(self, sample_data):
        """Тест очистки данных."""
        preprocessor = DataPreprocessor()

        # Добавляем некорректные данные
        df = sample_data.copy()
        df.loc[10, 'high'] = df.loc[10, 'low'] - 1  # high < low

        cleaned = preprocessor.clean(df)

        assert len(cleaned) <= len(df)
        assert (cleaned['high'] >= cleaned['low']).all()
        assert not cleaned.isnull().any().any()

    def test_add_datetime_features(self, sample_data):
        """Тест добавления временных признаков."""
        preprocessor = DataPreprocessor()
        df = preprocessor.add_datetime_features(sample_data)

        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
        assert 'month' in df.columns
        assert 'hour_sin' in df.columns
        assert 'hour_cos' in df.columns

    def test_split_data(self, sample_data):
        """Тест разделения данных."""
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.split_data(sample_data, train_ratio=0.6, val_ratio=0.2)

        assert len(train) + len(val) + len(test) == len(sample_data)
        assert len(train) > len(val) > 0


class TestCacheManager:
    """Тесты для CacheManager."""

    def test_save_and_load(self, sample_data, setup_cache):
        """Тест сохранения и загрузки кэша."""
        cache = CacheManager(cache_dir=setup_cache, compress=False)

        # Сохраняем
        cache.save(sample_data, "XAUUSD", "H1", 1)

        # Загружаем
        loaded = cache.get("XAUUSD", "H1", 1, max_age_hours=24)

        assert loaded is not None
        assert len(loaded) == len(sample_data)
        assert (loaded['close'] == sample_data['close']).all()

    def test_cache_expiry(self, sample_data, setup_cache):
        """Тест устаревания кэша."""
        cache = CacheManager(cache_dir=setup_cache, compress=False)

        cache.save(sample_data, "XAUUSD", "H1", 1)

        # Пытаемся загрузить с маленьким max_age
        loaded = cache.get("XAUUSD", "H1", 1, max_age_hours=0)

        assert loaded is None

    def test_clear_old(self, sample_data, setup_cache):
        """Тест очистки старых файлов."""
        cache = CacheManager(cache_dir=setup_cache, compress=False)

        cache.save(sample_data, "XAUUSD", "H1", 1)
        cache.save(sample_data, "Brent", "H1", 1)

        # Очищаем (в тестах файлы только что созданы, поэтому не должны удаляться)
        removed = cache.clear_old(max_age_days=0)

        assert removed == 0

        # Проверяем статистику
        stats = cache.get_stats()
        assert stats['total_files'] == 2


class TestDataDownloader:
    """Тесты для DataDownloader."""

    def test_initialization(self):
        """Тест инициализации."""
        downloader = DataDownloader("XAUUSD", timeframe="H1", years=1, use_cache=False)

        assert downloader.symbol == "XAUUSD"
        assert downloader.timeframe_name == "H1"
        assert downloader.years == 1
        assert downloader.use_cache is False

    @pytest.mark.skip(reason="Требуется подключение к MT5")
    def test_download(self):
        """Тест загрузки данных (требуется MT5)."""
        downloader = DataDownloader("XAUUSD", timeframe="H1", years=1, use_cache=False)
        df = downloader.download()

        assert df is not None
        assert len(df) > 0
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
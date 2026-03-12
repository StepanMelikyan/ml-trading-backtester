# tests/test_features.py
"""
Тесты для модулей расчета технических индикаторов.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from features.base_indicators import BaseIndicators
from features.trend_indicators import TrendIndicators
from features.volume_indicators import VolumeIndicators
from features.volatility import VolatilityIndicators
from features.oscillators import OscillatorIndicators
from features.price_patterns import PricePatterns
from features.feature_engineering import FeatureEngineering
from features.feature_selector import FeatureSelector


@pytest.fixture
def sample_data():
    """Создает тестовые данные."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
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
        'volume': volume
    })

    return df


class TestBaseIndicators:
    """Тесты для базовых индикаторов."""

    def test_sma(self, sample_data):
        """Тест SMA."""
        df = BaseIndicators.add_sma(sample_data.copy(), windows=[10, 20])

        assert 'SMA_10' in df.columns
        assert 'SMA_20' in df.columns
        assert 'SMA_10_slope' in df.columns
        assert 'price_to_SMA_10' in df.columns

    def test_rsi(self, sample_data):
        """Тест RSI."""
        df = BaseIndicators.add_rsi(sample_data.copy(), periods=[14])

        assert 'RSI_14' in df.columns
        assert df['RSI_14'].min() >= 0
        assert df['RSI_14'].max() <= 100

    def test_macd(self, sample_data):
        """Тест MACD."""
        df = BaseIndicators.add_macd(sample_data.copy())

        assert 'MACD' in df.columns
        assert 'MACD_signal' in df.columns
        assert 'MACD_histogram' in df.columns
        assert 'MACD_cross' in df.columns


class TestTrendIndicators:
    """Тесты для трендовых индикаторов."""

    def test_adx(self, sample_data):
        """Тест ADX."""
        df = TrendIndicators.add_adx(sample_data.copy())

        assert 'ADX' in df.columns
        assert 'DI_plus' in df.columns
        assert 'DI_minus' in df.columns
        assert df['ADX'].min() >= 0
        assert df['ADX'].max() <= 100

    def test_parabolic_sar(self, sample_data):
        """Тест Parabolic SAR."""
        df = TrendIndicators.add_parabolic_sar(sample_data.copy())

        assert 'PSAR' in df.columns
        assert 'PSAR_trend' in df.columns
        assert 'PSAR_distance' in df.columns


class TestVolumeIndicators:
    """Тесты для объемных индикаторов."""

    def test_obv(self, sample_data):
        """Тест OBV."""
        df = VolumeIndicators.add_obv(sample_data.copy())

        assert 'OBV' in df.columns
        assert 'OBV_MA' in df.columns

    def test_mfi(self, sample_data):
        """Тест MFI."""
        df = VolumeIndicators.add_mfi(sample_data.copy())

        assert 'MFI' in df.columns
        assert df['MFI'].min() >= 0
        assert df['MFI'].max() <= 100

    def test_vwap(self, sample_data):
        """Тест VWAP."""
        df = VolumeIndicators.add_vwap(sample_data.copy())

        assert 'VWAP' in df.columns
        assert 'VWAP_deviation' in df.columns


class TestVolatilityIndicators:
    """Тесты для индикаторов волатильности."""

    def test_bollinger_bands(self, sample_data):
        """Тест Bollinger Bands."""
        df = VolatilityIndicators.add_bollinger_bands(sample_data.copy())

        assert 'BB_upper_2.0' in df.columns
        assert 'BB_lower_2.0' in df.columns
        assert 'BB_width_2.0' in df.columns

    def test_atr(self, sample_data):
        """Тест ATR."""
        df = VolatilityIndicators.add_atr(sample_data.copy(), periods=[14])

        assert 'ATR_14' in df.columns
        assert 'ATR_pct_14' in df.columns


class TestOscillatorIndicators:
    """Тесты для осцилляторов."""

    def test_stochastic(self, sample_data):
        """Тест Stochastic."""
        df = OscillatorIndicators.add_stochastic(sample_data.copy())

        assert 'Stoch_K' in df.columns
        assert 'Stoch_D' in df.columns
        assert 'Stoch_overbought' in df.columns

    def test_cci(self, sample_data):
        """Тест CCI."""
        df = OscillatorIndicators.add_cci(sample_data.copy())

        assert 'CCI' in df.columns
        assert 'CCI_overbought' in df.columns


class TestPricePatterns:
    """Тесты для ценовых паттернов."""

    def test_candle_patterns(self, sample_data):
        """Тест свечных паттернов."""
        df = PricePatterns.add_candle_patterns(sample_data.copy())

        assert 'Doji' in df.columns
        assert 'Hammer' in df.columns
        assert 'BullishEngulfing' in df.columns

    def test_fractals(self, sample_data):
        """Тест фракталов."""
        df = PricePatterns.add_fractals(sample_data.copy())

        assert 'FractalUp' in df.columns
        assert 'FractalDown' in df.columns


class TestFeatureEngineering:
    """Тесты для feature engineering."""

    def test_create_target(self, sample_data):
        """Тест создания целевой переменной."""
        fe = FeatureEngineering(sample_data.copy())
        df = fe.create_target(horizon=5, target_type='direction')

        assert 'target' in df.columns
        assert df['target'].isin([0, 1]).all()

    def test_lag_features(self, sample_data):
        """Тест создания лагов."""
        fe = FeatureEngineering(sample_data.copy())
        df = fe.create_lag_features(['close'], lags=[1, 2, 3])

        assert 'close_lag_1' in df.columns
        assert 'close_lag_2' in df.columns
        assert 'close_lag_3' in df.columns

    def test_rolling_features(self, sample_data):
        """Тест скользящих статистик."""
        fe = FeatureEngineering(sample_data.copy())
        df = fe.create_rolling_features(['close'], windows=[5, 10],
                                        functions=['mean', 'std'])

        assert 'close_mean_5' in df.columns
        assert 'close_std_10' in df.columns


class TestFeatureSelector:
    """Тесты для отбора признаков."""

    def test_feature_selection(self, sample_data):
        """Тест отбора признаков."""
        # Добавляем индикаторы
        df = BaseIndicators.add_all(sample_data.copy())
        df = TrendIndicators.add_all(df)

        # Создаем целевую переменную
        fe = FeatureEngineering(df)
        df = fe.create_target(horizon=5)
        df = fe.clean_data()

        # Отбираем признаки
        feature_cols = [c for c in df.columns if c not in
                        ['time', 'open', 'high', 'low', 'close', 'volume', 'target']]

        selector = FeatureSelector(n_features=10)
        selected, stats = selector.select_features(df, feature_cols, 'target')

        assert len(selected) <= 10
        assert stats['final'] > 0
        assert stats['initial'] >= stats['final']
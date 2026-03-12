# features/volume_indicators.py
import pandas as pd
import numpy as np
from typing import Optional


class VolumeIndicators:
    """Индикаторы на основе объема торгов"""

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """On-Balance Volume - связь цены и объема"""
        if 'tick_volume' not in df.columns:
            print("⚠️ 'tick_volume' не найден, пропускаем OBV")
            return df

        obv = np.zeros(len(df))
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv[i] = obv[i - 1] + df['tick_volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv[i] = obv[i - 1] - df['tick_volume'].iloc[i]
            else:
                obv[i] = obv[i - 1]

        df['OBV'] = obv
        df['OBV_MA'] = df['OBV'].rolling(20).mean()
        df['OBV_slope'] = df['OBV'].diff(5)

        # Дивергенция OBV и цены
        df['OBV_divergence'] = df['close'] - df['OBV']
        return df

    @staticmethod
    def add_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Money Flow Index - RSI с учетом объема"""
        if 'tick_volume' not in df.columns:
            return df

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['tick_volume']

        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)

        pos_mf = pd.Series(positive_flow).rolling(period).sum()
        neg_mf = pd.Series(negative_flow).rolling(period).sum()

        money_ratio = pos_mf / neg_mf
        df['MFI'] = 100 - (100 / (1 + money_ratio))

        # MFI сигналы
        df['MFI_overbought'] = (df['MFI'] > 80).astype(int)
        df['MFI_oversold'] = (df['MFI'] < 20).astype(int)
        return df

    @staticmethod
    def add_volume_profile(df: pd.DataFrame, num_bins: int = 24) -> pd.DataFrame:
        """Volume Profile - распределение объема по ценам"""
        if 'tick_volume' not in df.columns:
            return df

        # Определяем ценовые уровни
        price_min = df['low'].min()
        price_max = df['high'].max()
        bin_size = (price_max - price_min) / num_bins

        # Создаем bins
        df['price_bin'] = ((df['close'] - price_min) / bin_size).astype(int)
        df['price_bin'] = df['price_bin'].clip(0, num_bins - 1)

        # Объем по бинам
        volume_by_bin = df.groupby('price_bin')['tick_volume'].sum()

        # Находим POC (Point of Control) - цену с максимальным объемом
        poc_bin = volume_by_bin.idxmax()
        df['POC_price'] = price_min + (poc_bin + 0.5) * bin_size

        # Value Area (70% объема)
        sorted_bins = volume_by_bin.sort_values(ascending=False)
        cumulative_volume = 0
        total_volume = volume_by_bin.sum()
        value_area_bins = []

        for bin_idx, vol in sorted_bins.items():
            if cumulative_volume / total_volume < 0.7:
                value_area_bins.append(bin_idx)
                cumulative_volume += vol
            else:
                break

        if value_area_bins:
            va_low = min(value_area_bins) * bin_size + price_min
            va_high = (max(value_area_bins) + 1) * bin_size + price_min
            df['VA_low'] = va_low
            df['VA_high'] = va_high
            df['in_value_area'] = ((df['close'] >= va_low) & (df['close'] <= va_high)).astype(int)

        return df

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Volume Weighted Average Price"""
        if 'tick_volume' not in df.columns:
            return df

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['VWAP'] = (typical_price * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()

        # Отклонение от VWAP
        df['VWAP_deviation'] = (df['close'] - df['VWAP']) / df['VWAP']
        return df
# config/indicators_config.py
"""
Конфигурация всех технических индикаторов.
Централизованное управление параметрами индикаторов.
"""

# ========== ТРЕНДОВЫЕ ИНДИКАТОРЫ ==========
TREND_INDICATORS = {
    "moving_averages": {
        "sma": {"windows": [10, 20, 50, 100, 200]},
        "ema": {"windows": [12, 26, 50, 200]},
        "wma": {"windows": [20]},
        "hma": {"windows": [20]},
        "vwma": {"windows": [20]}
    },
    "adx": {
        "period": 14,
        "thresholds": [25, 50],
        "smoothing": 14
    },
    "parabolic_sar": {
        "start": 0.02,
        "increment": 0.02,
        "maximum": 0.2
    },
    "ichimoku": {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_b_period": 52,
        "displacement": 26
    },
    "supertrend": {
        "period": 10,
        "multiplier": 3.0
    },
    "vwap": {
        "period": "D1",
        "anchor": "session"
    }
}

# ========== ОСЦИЛЛЯТОРЫ ==========
OSCILLATORS = {
    "rsi": {
        "periods": [7, 14, 21],
        "overbought": 70,
        "oversold": 30,
        "smoothing": "sma"
    },
    "stochastic": {
        "k_period": 14,
        "d_period": 3,
        "slowing": 3,
        "overbought": 80,
        "oversold": 20,
        "ma_type": "sma"
    },
    "macd": {
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "ma_type": "ema",
        "histogram": True
    },
    "cci": {
        "periods": [20],
        "overbought": 100,
        "oversold": -100,
        "ma_type": "sma"
    },
    "williams_r": {
        "periods": [14],
        "overbought": -20,
        "oversold": -80
    },
    "ultimate_oscillator": {
        "periods": [7, 14, 28],
        "weights": [4, 2, 1],
        "overbought": 70,
        "oversold": 30
    },
    "awesome_oscillator": {
        "fast": 5,
        "slow": 34,
        "ma_type": "sma"
    }
}

# ========== ИНДИКАТОРЫ ВОЛАТИЛЬНОСТИ ==========
VOLATILITY_INDICATORS = {
    "bollinger_bands": {
        "period": 20,
        "std_dev": [2.0, 2.5, 3.0],
        "ma_type": "sma",
        "use_ema": False
    },
    "keltner_channels": {
        "ema_period": 20,
        "atr_period": 10,
        "multiplier": [1.5, 2.0, 2.5]
    },
    "atr": {
        "periods": [7, 14, 21],
        "smoothing": "sma"
    },
    "chaikin_volatility": {
        "periods": [10],
        "ma_type": "ema",
        "smoothing": 10
    },
    "donchian_channels": {
        "periods": [20],
        "use_percent": False
    },
    "historical_volatility": {
        "periods": [20, 50, 100],
        "annualization": 252
    },
    "ulcer_index": {
        "period": 14,
        "ma_type": "sma"
    }
}

# ========== ОБЪЕМНЫЕ ИНДИКАТОРЫ ==========
VOLUME_INDICATORS = {
    "obv": {
        "use": True,
        "ma_period": 20,
        "smoothing": "ema"
    },
    "mfi": {
        "periods": [14],
        "overbought": 80,
        "oversold": 20,
        "volume_type": "tick"
    },
    "volume_profile": {
        "num_bins": 24,
        "value_area_pct": 70,
        "poc_type": "high"
    },
    "vwap": {
        "periods": ["D1", "W1"],
        "anchor": ["session", "week"]
    },
    "accumulation_distribution": {
        "use": True,
        "smoothing": "ema",
        "period": 20
    },
    "chaikin_money_flow": {
        "period": 20,
        "threshold": 0
    },
    "ease_of_movement": {
        "period": 14,
        "ma_type": "ema",
        "volume_divisor": 10000
    }
}

# ========== ЦЕНОВЫЕ ПАТТЕРНЫ ==========
PRICE_PATTERNS = {
    "candle_patterns": {
        "patterns": [
            "doji", "hammer", "shooting_star", "hanging_man",
            "engulfing", "harami", "morning_star", "evening_star",
            "three_white_soldiers", "three_black_crows",
            "piercing_pattern", "dark_cloud_cover",
            "marubozu", "spinning_top", "long_legged_doji"
        ],
        "require_confirmation": True,
        "confirmation_period": 1
    },
    "fractals": {
        "period": 5,
        "require_clear": True,
        "sensitivity": "normal"
    },
    "support_resistance": {
        "window": 20,
        "num_levels": 5,
        "merge_threshold": 0.01,  # 1%
        "strength_threshold": 2
    },
    "pivot_points": {
        "methods": ["classic", "fibonacci", "woodie", "camarilla"],
        "default_method": "classic"
    }
}

# ========== ОБЩАЯ КОНФИГУРАЦИЯ ==========
INDICATORS_CONFIG = {
    "trend": TREND_INDICATORS,
    "oscillators": OSCILLATORS,
    "volatility": VOLATILITY_INDICATORS,
    "volume": VOLUME_INDICATORS,
    "patterns": PRICE_PATTERNS
}

# Какие категории индикаторов использовать
ENABLED_CATEGORIES = {
    "trend": True,
    "oscillators": True,
    "volatility": True,
    "volume": True,
    "patterns": True
}

# Настройки для feature engineering
FEATURE_CONFIG = {
    "max_lags": 20,
    "rolling_windows": [5, 10, 20, 50],
    "rolling_functions": ["mean", "std", "min", "max", "skew", "kurt"],
    "use_diff": True,
    "use_pct_change": True,
    "use_log": True,
    "interaction_features": True,
    "max_interactions": 50
}
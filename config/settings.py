# config/settings.py
"""
Глобальные настройки проекта.
Пути, параметры по умолчанию, конфигурация.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# ========== ПУТИ ==========
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Создаем директории
for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========== ТОРГОВЫЕ ИНСТРУМЕНТЫ ==========
SYMBOLS = {
    "XAUUSD": {
        "name": "Gold",
        "pip_size": 0.01,
        "spread": 0.3,
        "margin_required": 0.01,
        "contract_size": 100,
        "min_volume": 0.01,
        "max_volume": 100,
        "volume_step": 0.01
    },
    "Brent": {
        "name": "Brent Crude Oil",
        "pip_size": 0.01,
        "spread": 0.02,
        "margin_required": 0.01,
        "contract_size": 1000,
        "min_volume": 0.1,
        "max_volume": 100,
        "volume_step": 0.1
    },
    "WTI": {
        "name": "WTI Crude Oil",
        "pip_size": 0.01,
        "spread": 0.02,
        "margin_required": 0.01,
        "contract_size": 1000,
        "min_volume": 0.1,
        "max_volume": 100,
        "volume_step": 0.1
    }
}

# ========== ТАЙМФРЕЙМЫ ==========
TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200
}

DEFAULT_TIMEFRAME = "H1"
DEFAULT_YEARS = 5

# ========== НАСТРОЙКИ MT5 ==========
MT5_CONFIG = {
    "login": os.getenv("MT5_LOGIN", ""),
    "password": os.getenv("MT5_PASSWORD", ""),
    "server": os.getenv("MT5_SERVER", ""),
    "path": os.getenv("MT5_PATH", "")
}

# ========== НАСТРОЙКИ БЭКТЕСТИНГА ==========
BACKTEST_CONFIG = {
    "initial_capital": 10000,
    "commission": 0.001,  # 0.1%
    "slippage": 0.0001,   # 0.01%
    "risk_per_trade": 0.02,  # 2%
    "leverage": 1.0,
    "max_open_positions": 1,
    "use_fractional_sizing": True
}

# ========== НАСТРОЙКИ ЛОГИРОВАНИЯ ==========
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file": LOGS_DIR / "trading.log",
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5
}

# ========== НАСТРОЙКИ КЭШИРОВАНИЯ ==========
CACHE_CONFIG = {
    "enabled": True,
    "max_age_hours": 24,
    "compress": True,
    "auto_clean": True,
    "clean_age_days": 7
}

# ========== НАСТРОЙКИ ВИЗУАЛИЗАЦИИ ==========
PLOT_CONFIG = {
    "style": "seaborn-v0_8-darkgrid",
    "figsize": (15, 8),
    "dpi": 150,
    "save_format": "png"
}
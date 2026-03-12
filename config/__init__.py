# config/__init__.py
"""Пакет с конфигурационными файлами"""

from .settings import *
from .indicators_config import *
from .model_config import *

__all__ = [
    'SYMBOLS',
    'TIMEFRAMES',
    'BACKTEST_CONFIG',
    'MODEL_CONFIG',
    'INDICATORS_CONFIG'
]
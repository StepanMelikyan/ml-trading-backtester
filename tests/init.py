# tests/__init__.py
"""Пакет с тестами для всех модулей проекта"""

from .test_data import *
from .test_features import *
from .test_models import *
from .test_backtest import *
from .test_integration import *

__all__ = [
    'TestDataDownloader',
    'TestDataPreprocessor',
    'TestFeatures',
    'TestModels',
    'TestBacktest',
    'TestIntegration'
]
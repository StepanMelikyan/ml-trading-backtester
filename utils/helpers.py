# utils/helpers.py
"""
Вспомогательные функции общего назначения.
"""

import pandas as pd
import numpy as np
import random
import string
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import hashlib


def generate_id(length: int = 8) -> str:
    """
    Генерирует случайный ID.

    Args:
        length: длина ID

    Returns:
        Случайная строка
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Безопасное деление с обработкой нуля.

    Args:
        a: числитель
        b: знаменатель
        default: значение по умолчанию при делении на ноль

    Returns:
        Результат деления или default
    """
    if b == 0 or np.isnan(b) or np.isinf(b):
        return default
    return a / b


def calculate_pip_value(symbol: str, price: float, volume: float = 1.0) -> float:
    """
    Расчет стоимости пипса для разных инструментов.

    Args:
        symbol: тикер инструмента
        price: текущая цена
        volume: объем в лотах

    Returns:
        Стоимость одного пипса в долларах
    """
    pip_values = {
        'XAUUSD': 0.01,  # Золото
        'Brent': 0.01,  # Нефть Brent
        'WTI': 0.01,  # Нефть WTI
    }

    pip_size = pip_values.get(symbol, 0.0001)
    return pip_size * volume * price


def round_to_pip(price: float, symbol: str) -> float:
    """
    Округление до целого пипса.

    Args:
        price: цена
        symbol: тикер инструмента

    Returns:
        Цена, округленная до пипса
    """
    pip_sizes = {
        'XAUUSD': 0.01,
        'Brent': 0.01,
        'WTI': 0.01,
    }

    pip_size = pip_sizes.get(symbol, 0.0001)
    return round(price / pip_size) * pip_size


def json_serialize(obj: Any) -> str:
    """
    Сериализация объектов в JSON с обработкой специальных типов.

    Args:
        obj: объект для сериализации

    Returns:
        JSON-совместимое представление
    """
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data: Any, path: Path, indent: int = 2):
    """
    Сохраняет данные в JSON файл.

    Args:
        data: данные для сохранения
        path: путь к файлу
        indent: отступы для форматирования
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=json_serialize, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """
    Загружает данные из JSON файла.

    Args:
        path: путь к файлу

    Returns:
        Загруженные данные
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_session_stats(df: pd.DataFrame) -> Dict:
    """
    Рассчитывает статистику по торговым сессиям.

    Args:
        df: DataFrame с колонкой 'hour'

    Returns:
        Словарь со статистикой по сессиям
    """
    if 'hour' not in df.columns:
        return {}

    # Определение сессий (время UTC)
    sessions = {
        'Asian': list(range(0, 9)),  # 00:00 - 09:00
        'European': list(range(9, 17)),  # 09:00 - 17:00
        'American': list(range(17, 24))  # 17:00 - 00:00
    }

    stats = {}

    for session_name, hours in sessions.items():
        session_data = df[df['hour'].isin(hours)]

        if len(session_data) > 0:
            stats[session_name] = {
                'count': len(session_data),
                'avg_range': (session_data['high'] - session_data['low']).mean(),
                'avg_volume': session_data['volume'].mean() if 'volume' in session_data.columns else 0,
                'volatility': session_data['close'].pct_change().std() * 100
            }

    return stats


def calculate_hash(data: Any) -> str:
    """
    Вычисляет хеш данных.

    Args:
        data: данные для хеширования

    Returns:
        MD5 хеш
    """
    if isinstance(data, pd.DataFrame):
        # Для DataFrame берем первые 1000 строк и основные колонки
        sample = data[['open', 'high', 'low', 'close']].head(1000).values
        data_bytes = sample.tobytes()
    else:
        data_bytes = str(data).encode('utf-8')

    return hashlib.md5(data_bytes).hexdigest()


def format_number(num: float, decimals: int = 2) -> str:
    """
    Форматирует число с разделителями.

    Args:
        num: число
        decimals: количество знаков после запятой

    Returns:
        Отформатированная строка
    """
    if abs(num) >= 1e6:
        return f"{num / 1e6:.{decimals}f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def ensure_list(x: Any) -> List:
    """
    Гарантирует, что объект является списком.

    Args:
        x: объект

    Returns:
        Список
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def get_memory_usage(obj: Any) -> str:
    """
    Возвращает размер объекта в человеко-читаемом формате.

    Args:
        obj: объект

    Returns:
        Строка с размером
    """
    import sys
    size = sys.getsizeof(obj)

    if size < 1024:
        return f"{size} B"
    elif size < 1024 ** 2:
        return f"{size / 1024:.2f} KB"
    elif size < 1024 ** 3:
        return f"{size / 1024 ** 2:.2f} MB"
    else:
        return f"{size / 1024 ** 3:.2f} GB"
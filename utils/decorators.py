# utils/decorators.py
"""
Декораторы для измерения времени, памяти, повторных попыток и логирования.
"""

import time
import functools
import tracemalloc
from typing import Any, Callable, Optional
from .logger import log
import pandas as pd


def timer(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения функции.

    Args:
        func: декорируемая функция

    Returns:
        Функция-обертка
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        log.debug(f"⏱ {func.__name__} выполнилась за {execution_time:.4f} сек")

        # Добавляем атрибут с временем выполнения
        wrapper.last_execution_time = execution_time

        return result

    return wrapper


def async_timer(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения асинхронной функции.

    Args:
        func: декорируемая асинхронная функция

    Returns:
        Асинхронная функция-обертка
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        log.debug(f"⏱ async {func.__name__} выполнилась за {execution_time:.4f} сек")

        return result

    return wrapper


def memory_tracker(func: Callable) -> Callable:
    """
    Декоратор для отслеживания использования памяти.

    Args:
        func: декорируемая функция

    Returns:
        Функция-обертка
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        log.debug(f"📊 {func.__name__} использует память: текущая={current / 10 ** 6:.2f}MB, пик={peak / 10 ** 6:.2f}MB")

        return result

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: tuple = (Exception,)):
    """
    Декоратор для повторных попыток выполнения функции при ошибках.

    Args:
        max_attempts: максимальное количество попыток
        delay: начальная задержка между попытками
        backoff: множитель задержки (геометрическая прогрессия)
        exceptions: кортеж исключений для повторных попыток

    Returns:
        Декоратор
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        log.error(f"❌ {func.__name__} не удалась после {max_attempts} попыток: {e}")
                        raise

                    log.warning(
                        f"⚠️ Попытка {attempt + 1}/{max_attempts} для {func.__name__} не удалась: {e}. Повтор через {current_delay:.1f}с...")
                    time.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        return wrapper

    return decorator


def cache_result(maxsize: int = 128, ttl: Optional[float] = None):
    """
    Декоратор для кэширования результатов функции.

    Args:
        maxsize: максимальный размер кэша
        ttl: время жизни кэша в секундах (None = бессрочно)

    Returns:
        Декоратор
    """

    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Создаем ключ из аргументов
            key = str(args) + str(sorted(kwargs.items()))

            # Проверяем наличие в кэше
            if key in cache:
                # Проверяем TTL
                if ttl is None or (time.time() - cache_times[key]) < ttl:
                    log.debug(f"💾 Возвращаем кэшированный результат для {func.__name__}")
                    return cache[key]

            # Вычисляем результат
            result = func(*args, **kwargs)

            # Сохраняем в кэш
            if len(cache) >= maxsize:
                # Удаляем самый старый элемент
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                del cache[oldest_key]
                del cache_times[oldest_key]

            cache[key] = result
            cache_times[key] = time.time()

            return result

        return wrapper

    return decorator


def validate_data(func: Callable) -> Callable:
    """
    Декоратор для валидации входных данных.

    Args:
        func: декорируемая функция

    Returns:
        Функция-обертка
    """

    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        # Проверяем, что DataFrame не пустой
        if df is None or len(df) == 0:
            raise ValueError("DataFrame пустой")

        # Проверяем необходимые колонки
        required_columns = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

        # Проверяем на NaN
        if df[required_columns].isnull().any().any():
            log.warning(f"⚠️ {func.__name__}: DataFrame содержит NaN значения")

        return func(df, *args, **kwargs)

    return wrapper


def log_calls(level: str = 'debug'):
    """
    Декоратор для логирования вызовов функции.

    Args:
        level: уровень логирования ('debug', 'info', 'warning')

    Returns:
        Декоратор
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_func = getattr(log, level.lower(), log.debug)

            # Логируем вызов
            log_func(f"📞 Вызов {func.__name__} с args={args}, kwargs={kwargs}")

            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                log_func(f"✅ {func.__name__} успешно завершена за {execution_time:.3f}с")
                return result

            except Exception as e:
                log.error(f"❌ Ошибка в {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


def singleton(cls):
    """
    Декоратор для реализации паттерна Singleton.

    Args:
        cls: класс

    Returns:
        Класс-синглтон
    """
    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def deprecated(message: str = ""):
    """
    Декоратор для пометки устаревших функций.

    Args:
        message: сообщение о причинах устаревания

    Returns:
        Декоратор
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log.warning(f"⚠️ {func.__name__} устарела. {message}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def suppress_errors(default_return: Any = None, log_error: bool = True):
    """
    Декоратор для подавления ошибок.

    Args:
        default_return: значение по умолчанию при ошибке
        log_error: логировать ли ошибку

    Returns:
        Декоратор
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    log.error(f"❌ Ошибка в {func.__name__} (подавлена): {e}")
                return default_return

        return wrapper

    return decorator


def measure_memory(func: Callable) -> Callable:
    """
    Декоратор для измерения пикового использования памяти.

    Args:
        func: декорируемая функция

    Returns:
        Функция-обертка
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_diff = mem_after - mem_before

        log.debug(f"📊 {func.__name__} изменила память на {mem_diff:+.2f} MB")

        return result

    return wrapper
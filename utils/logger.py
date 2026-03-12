# utils/logger.py
"""
Модуль для настройки логирования в проекте.
Поддерживает вывод в файл, консоль и ротацию логов.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional
import traceback


class Logger:
    """
    Класс для управления логированием.
    Реализует паттерн Singleton для единой конфигурации.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "trading_bot", log_dir: Optional[Path] = None):
        """
        Инициализация логгера.

        Args:
            name: имя логгера
            log_dir: директория для сохранения логов
        """
        if self._initialized:
            return

        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Создаем логгер
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Очищаем существующие handlers

        # Формат логов
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Добавляем handlers
        self._add_file_handler()
        self._add_console_handler()
        self._add_error_file_handler()

        self._initialized = True

    def _add_file_handler(self):
        """Добавляет файловый handler для всех логов."""
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"

        # Используем RotatingFileHandler для автоматической ротации
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)

    def _add_console_handler(self):
        """Добавляет консольный handler для вывода в терминал."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def _add_error_file_handler(self):
        """Добавляет отдельный файл для ошибок."""
        error_file = self.log_dir / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"

        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)
        self.logger.addHandler(error_handler)

    def debug(self, msg: str, *args, **kwargs):
        """Логирование уровня DEBUG."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Логирование уровня INFO."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Логирование уровня WARNING."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Логирование уровня ERROR."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Логирование уровня CRITICAL."""
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Логирование исключения с traceback."""
        self.logger.exception(msg, *args, **kwargs)

    def log_exception(self, e: Exception, context: str = ""):
        """
        Логирует исключение с полным traceback.

        Args:
            e: исключение
            context: дополнительный контекст
        """
        tb = traceback.format_exc()
        self.error(f"Exception in {context}: {str(e)}\n{tb}")


# Создаем глобальный экземпляр логгера
log = Logger()
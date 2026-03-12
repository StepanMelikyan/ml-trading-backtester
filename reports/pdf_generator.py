# reports/pdf_generator.py
"""
Генератор PDF отчетов из HTML.
"""

import pdfkit
from pathlib import Path
from datetime import datetime
from typing import Optional  # ← ДОБАВЛЕНО!
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import REPORTS_DIR
from utils.logger import log


class PDFReport:
    """
    Генератор PDF отчетов.
    Требует установки wkhtmltopdf.
    """

    def __init__(self, symbol: str, output_dir: Optional[Path] = None):
        """
        Инициализация генератора PDF.

        Args:
            symbol: торговый инструмент
            output_dir: директория для сохранения
        """
        self.symbol = symbol
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Проверяем наличие wkhtmltopdf
        self.wkhtmltopdf_available = self._check_wkhtmltopdf()

    def _check_wkhtmltopdf(self) -> bool:
        """Проверяет доступность wkhtmltopdf."""
        try:
            pdfkit.from_string('<html><body>Test</body></html>', False)
            return True
        except:
            log.warning("⚠️ wkhtmltopdf не найден. PDF отчеты недоступны.")
            log.warning("   Установите: sudo apt-get install wkhtmltopdf")
            return False

    def from_html(self, html_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Конвертирует HTML в PDF.

        Args:
            html_path: путь к HTML файлу
            output_path: путь для сохранения PDF

        Returns:
            Путь к PDF файлу или None
        """
        if not self.wkhtmltopdf_available:
            log.error("❌ wkhtmltopdf не доступен")
            return None

        if not html_path.exists():
            log.error(f"❌ HTML файл не найден: {html_path}")
            return None

        if output_path is None:
            output_path = self.output_dir / f"{self.symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        try:
            pdfkit.from_file(str(html_path), str(output_path))
            log.info(f"✅ PDF отчет сохранен: {output_path}")
            return output_path
        except Exception as e:
            log.error(f"❌ Ошибка создания PDF: {e}")
            return None

    def from_string(self, html_string: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Создает PDF из HTML строки.

        Args:
            html_string: HTML строка
            output_path: путь для сохранения

        Returns:
            Путь к PDF файлу или None
        """
        if not self.wkhtmltopdf_available:
            log.error("❌ wkhtmltopdf не доступен")
            return None

        if output_path is None:
            output_path = self.output_dir / f"{self.symbol}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        try:
            pdfkit.from_string(html_string, str(output_path))
            log.info(f"✅ PDF отчет сохранен: {output_path}")
            return output_path
        except Exception as e:
            log.error(f"❌ Ошибка создания PDF: {e}")
            return None
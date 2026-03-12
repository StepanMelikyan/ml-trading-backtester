# data/file_downloader.py
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import log


class FileDownloader:
    """
    Простой загрузчик данных из CSV файла (без MT5)
    """

    def __init__(self, symbol: str, timeframe: str = "H1", years: int = 5, use_cache: bool = True):
        self.symbol = symbol
        self.timeframe = timeframe
        self.years = years
        self.use_cache = use_cache
        self.data_dir = Path(__file__).parent

    def download(self, force_download: bool = False) -> pd.DataFrame:
        """
        Загружает данные из CSV файла
        """
        filename = self.data_dir / f"{self.symbol}_{self.years}years.csv"

        log.info(f"📂 Загрузка из файла: {filename}")

        if not filename.exists():
            log.error(f"❌ Файл не найден: {filename}")
            return None

        df = pd.read_csv(filename)
        df['time'] = pd.to_datetime(df['time'])

        log.info(f"✅ Загружено {len(df)} записей")
        log.info(f"   Период: {df['time'].min()} - {df['time'].max()}")

        return df
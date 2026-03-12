# models/model_registry.py
"""
Реестр для управления всеми обученными моделями.
Позволяет сохранять, загружать, сравнивать и выбирать лучшие модели.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import sys
import shutil

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import MODELS_DIR
from utils.logger import log
from .base_model import BaseModel


class ModelRegistry:
    """
    Реестр моделей для управления версиями и сравнения.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Инициализация реестра.

        Args:
            registry_path: путь к файлу реестра
        """
        self.registry_path = registry_path or (MODELS_DIR / "registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """
        Загружает реестр из файла.

        Returns:
            Словарь с данными реестра
        """
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                log.warning("⚠️ Ошибка загрузки реестра, создаем новый")

        return {
            'models': [],
            'best_models': {},
            'last_updated': datetime.now().isoformat()
        }

    def _save_registry(self):
        """Сохраняет реестр в файл."""
        self.registry['last_updated'] = datetime.now().isoformat()

        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register(self, model: BaseModel, model_path: Path,
                 metrics: Dict, tags: List[str] = None):
        """
        Регистрирует модель в реестре.

        Args:
            model: экземпляр модели
            model_path: путь к сохраненной модели
            metrics: метрики модели
            tags: теги для поиска
        """
        model_info = {
            'id': f"{model.symbol}_{model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'name': model.name,
            'symbol': model.symbol,
            'path': str(model_path),
            'created_at': model.created_at.isoformat() if model.created_at else datetime.now().isoformat(),
            'training_time': model.training_time,
            'metrics': metrics,
            'tags': tags or [],
            'features': model.features
        }

        self.registry['models'].append(model_info)

        # Обновляем лучшие модели
        self._update_best_models(model.symbol)

        self._save_registry()

        log.info(f"📝 Модель зарегистрирована: {model_info['id']}")

        return model_info['id']

    def _update_best_models(self, symbol: str, metric: str = 'f1_score'):
        """
        Обновляет список лучших моделей для символа.

        Args:
            symbol: торговый инструмент
            metric: метрика для сравнения
        """
        # Фильтруем модели по символу
        symbol_models = [m for m in self.registry['models'] if m['symbol'] == symbol]

        if not symbol_models:
            return

        # Сортируем по метрике
        sorted_models = sorted(
            symbol_models,
            key=lambda x: x['metrics'].get(metric, 0),
            reverse=True
        )

        # Сохраняем топ-3
        self.registry['best_models'][symbol] = [m['id'] for m in sorted_models[:3]]

    def get_best_model(self, symbol: str, metric: str = 'f1_score') -> Optional[Dict]:
        """
        Возвращает лучшую модель для символа.

        Args:
            symbol: торговый инструмент
            metric: метрика для сравнения

        Returns:
            Информация о лучшей модели
        """
        symbol_models = [m for m in self.registry['models'] if m['symbol'] == symbol]

        if not symbol_models:
            return None

        best = max(
            symbol_models,
            key=lambda x: x['metrics'].get(metric, 0)
        )

        return best

    def list_models(self, symbol: Optional[str] = None,
                    model_type: Optional[str] = None) -> pd.DataFrame:
        """
        Возвращает список моделей в виде DataFrame.

        Args:
            symbol: фильтр по символу
            model_type: фильтр по типу модели

        Returns:
            DataFrame с информацией о моделях
        """
        models = self.registry['models']

        if symbol:
            models = [m for m in models if m['symbol'] == symbol]

        if model_type:
            models = [m for m in models if m['name'] == model_type]

        if not models:
            return pd.DataFrame()

        df = pd.DataFrame(models)

        # Разворачиваем метрики
        if 'metrics' in df.columns:
            metrics_df = df['metrics'].apply(pd.Series)
            df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)

        return df

    def compare_models(self, symbol: str, metrics: List[str] = None) -> pd.DataFrame:
        """
        Сравнивает все модели для символа.

        Args:
            symbol: торговый инструмент
            metrics: список метрик для сравнения

        Returns:
            DataFrame со сравнением
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']

        models = self.list_models(symbol)

        if models.empty:
            return pd.DataFrame()

        # Выбираем только нужные колонки
        cols = ['name', 'created_at'] + [m for m in metrics if m in models.columns]

        comparison = models[cols].copy()
        comparison['created_at'] = pd.to_datetime(comparison['created_at']).dt.strftime('%Y-%m-%d')

        return comparison.sort_values(metrics[0] if metrics else 'f1_score', ascending=False)

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        Возвращает путь к модели по ID.

        Args:
            model_id: идентификатор модели

        Returns:
            Путь к модели
        """
        for model in self.registry['models']:
            if model['id'] == model_id:
                return Path(model['path'])

        return None

    def delete_model(self, model_id: str, delete_files: bool = True):
        """
        Удаляет модель из реестра.

        Args:
            model_id: идентификатор модели
            delete_files: удалять ли файлы
        """
        for i, model in enumerate(self.registry['models']):
            if model['id'] == model_id:
                # Удаляем из списка
                del self.registry['models'][i]

                # Удаляем файлы
                if delete_files:
                    model_path = Path(model['path'])
                    if model_path.exists():
                        shutil.rmtree(model_path)
                        log.info(f"🗑 Удалены файлы модели: {model_path}")

                # Обновляем лучшие модели
                self._update_best_models(model['symbol'])

                self._save_registry()
                log.info(f"🗑 Модель {model_id} удалена из реестра")
                return

        log.warning(f"⚠️ Модель {model_id} не найдена")

    def get_stats(self) -> Dict:
        """
        Возвращает статистику по реестру.

        Returns:
            Словарь со статистикой
        """
        models = self.registry['models']

        if not models:
            return {'total_models': 0}

        df = pd.DataFrame(models)

        stats = {
            'total_models': len(models),
            'by_symbol': df['symbol'].value_counts().to_dict(),
            'by_type': df['name'].value_counts().to_dict(),
            'avg_accuracy': df['metrics'].apply(lambda x: x.get('accuracy', 0)).mean(),
            'avg_f1': df['metrics'].apply(lambda x: x.get('f1_score', 0)).mean(),
            'last_updated': self.registry['last_updated']
        }

        return stats

    def export_to_csv(self, path: Path):
        """
        Экспортирует реестр в CSV.

        Args:
            path: путь для сохранения
        """
        df = self.list_models()
        df.to_csv(path, index=False)
        log.info(f"📤 Реестр экспортирован в {path}")

    def get_model_comparison_chart(self, symbol: str):
        """
        Создает график сравнения моделей.

        Args:
            symbol: торговый инструмент
        """
        import matplotlib.pyplot as plt

        comparison = self.compare_models(symbol)

        if comparison.empty:
            log.warning(f"Нет моделей для {symbol}")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        available_metrics = [m for m in metrics if m in comparison.columns]

        x = np.arange(len(comparison))
        width = 0.2

        for i, metric in enumerate(available_metrics):
            ax.bar(x + i * width, comparison[metric].values, width, label=metric)

        ax.set_xlabel('Модели')
        ax.set_ylabel('Значение')
        ax.set_title(f'Сравнение моделей для {symbol}')
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(comparison['name'] + '\n' + comparison['created_at'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig
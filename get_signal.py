# get_signal.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Импортируем ваши модули для загрузки данных и расчета признаков
from data.downloader import DataDownloader
from features.base_indicators import BaseIndicators
from features.trend_indicators import TrendIndicators
# ... и другие индикаторы, которые вы используете
from features.feature_engineering import FeatureEngineering
from models.ensemble_models import RandomForestModel # Или XGBoostModel

def get_live_signal(symbol="XAUUSD", model_path=None):
    print(f"📡 Получение сигнала для {symbol}...")

    # 1. Загружаем последние данные (например, за последний месяц)
    downloader = DataDownloader(symbol, timeframe="H1", years=0.1, use_cache=False) # Загружаем ~1 месяц
    df = downloader.download()
    if df is None:
        print("❌ Не удалось загрузить данные.")
        return

    # 2. Добавляем признаки (ТОЧНО ТАК ЖЕ, как при обучении!)
    df = BaseIndicators.add_all(df)
    df = TrendIndicators.add_all(df)
    # ... все остальные индикаторы
    # ВАЖНО: Убедитесь, что у вас есть функция, которая добавляет все признаки сразу,
    # как ваша step2_calculate_indicators(), но применяет её к новому df.
    # Для простоты, здесь мы перечислим основные.

    # 3. Создаем лаги и скользящие (как в step3_create_features)
    fe = FeatureEngineering(df)
    # Убедитесь, что вы используете те же параметры, что и при обучении (те же lags, windows)
    df = fe.create_lag_features(['close'], lags=[1,2,3,5])
    df = fe.create_rolling_features(['close'], windows=[5,10])
    df = fe.clean_data(drop_na=True) # Удаляем строки с NaN

    # 4. Получаем ПОСЛЕДНЮЮ строку с фичами
    if df.empty:
        print("❌ Недостаточно данных для расчета фич.")
        return
    latest_features = df.iloc[-1:]

    # 5. Загружаем лучшую модель
    if model_path is None:
        # Здесь нужна логика для поиска последней или лучшей модели
        # Например, взять самую свежую из папки models/saved/
        models_dir = Path("models/saved")
        symbol_models = list(models_dir.glob(f"{symbol}_*"))
        if not symbol_models:
            print(f"❌ Нет сохраненных моделей для {symbol}")
            return
        # Сортируем по дате создания (по имени папки) и берем последнюю
        latest_model_dir = max(symbol_models, key=lambda p: p.stat().st_ctime)
        model_path = latest_model_dir / "model.joblib"

    # Так как мы используем ваши классы моделей, лучше загружать через них
    model = RandomForestModel(symbol) # Создаем объект модели
    try:
        model.load(latest_model_dir) # Загружаем состояние в объект
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # 6. Получаем предсказание
    # Убедитесь, что в latest_features есть только те колонки, на которых обучалась модель
    # Обычно это self.selected_features в вашем пайплайне.
    # Если у вас нет списка фич, просто используйте все, кроме time и target.
    feature_cols = [c for c in latest_features.columns if c not in ['time', 'target', 'signal']]
    X_live = latest_features[feature_cols].fillna(0)

    # Если модель ожидает numpy array, преобразуем
    prediction = model.predict(X_live)[0] # Предположим, predict возвращает массив
    proba = model.predict_proba(X_live)[0] # Вероятности

    # 7. Интерпретируем сигнал
    if prediction == 1:
        signal = "📈 ПОКУПКА (BUY)"
    elif prediction == 0:
        signal = "📉 ПРОДАЖА (SELL)"
    else:
        signal = "⚪ ДЕРЖАТЬ (HOLD)"

    print(f"\n{'='*40}")
    print(f"СИГНАЛ ДЛЯ {symbol}: {signal}")
    print(f"Вероятность: {max(proba)*100:.2f}%")
    print(f"{'='*40}\n")
    return prediction, proba

if __name__ == "__main__":
    # Пример использования
    get_live_signal("XAUUSD")
    # get_live_signal("Brent")
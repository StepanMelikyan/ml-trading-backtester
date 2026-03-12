# config/model_config.py
"""
Конфигурация всех моделей машинного обучения.
"""

# ========== RANDOM FOREST ==========
RANDOM_FOREST_CONFIG = {
    "enabled": True,
    "params": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "bootstrap": True,
        "oob_score": True,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced"
    },
    "grid_search": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None]
    },
    "feature_importance": True,
    "calibration": True
}

# ========== XGBOOST ==========
XGBOOST_CONFIG = {
    "enabled": True,
    "params": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "min_child_weight": 1,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "predictor": "cpu_predictor"
    },
    "grid_search": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9, 12],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
        "reg_alpha": [0, 0.1, 1, 10],
        "reg_lambda": [0.1, 1, 10]
    },
    "early_stopping": True,
    "early_stopping_rounds": 50
}

# ========== LIGHTGBM ==========
LIGHTGBM_CONFIG = {
    "enabled": True,
    "params": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 0,
        "min_child_samples": 20,
        "min_child_weight": 0.001,
        "random_state": 42,
        "verbose": -1,
        "boosting_type": "gbdt",
        "class_weight": "balanced"
    },
    "grid_search": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9, 12, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 1, 10],
        "reg_lambda": [0, 0.1, 1, 10]
    },
    "early_stopping": True,
    "early_stopping_rounds": 50,
    "feature_importance_type": "gain"
}

# ========== LSTM ==========
LSTM_CONFIG = {
    "enabled": True,
    "params": {
        "sequence_length": 60,
        "lstm_units": [100, 50],
        "dropout": 0.2,
        "recurrent_dropout": 0.2,
        "bidirectional": False,
        "batch_size": 32,
        "epochs": 100,
        "patience": 15,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy", "precision", "recall", "auc"]
    },
    "layers": [
        {"type": "lstm", "units": 100, "return_sequences": True},
        {"type": "dropout", "rate": 0.2},
        {"type": "lstm", "units": 50, "return_sequences": False},
        {"type": "dropout", "rate": 0.2},
        {"type": "dense", "units": 1, "activation": "sigmoid"}
    ],
    "callbacks": {
        "early_stopping": {"monitor": "val_loss", "patience": 15, "mode": "min"},
        "reduce_lr": {"monitor": "val_loss", "factor": 0.5, "patience": 5, "min_lr": 1e-6},
        "model_checkpoint": {"monitor": "val_accuracy", "mode": "max", "save_best_only": True}
    }
}

# ========== GRU ==========
GRU_CONFIG = {
    "enabled": True,
    "params": {
        "sequence_length": 60,
        "gru_units": [100, 50],
        "dropout": 0.2,
        "recurrent_dropout": 0.2,
        "bidirectional": False,
        "batch_size": 32,
        "epochs": 100,
        "patience": 15,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "binary_crossentropy"
    },
    "layers": [
        {"type": "gru", "units": 100, "return_sequences": True},
        {"type": "dropout", "rate": 0.2},
        {"type": "gru", "units": 50, "return_sequences": False},
        {"type": "dropout", "rate": 0.2},
        {"type": "dense", "units": 1, "activation": "sigmoid"}
    ]
}

# ========== TRANSFORMER ==========
TRANSFORMER_CONFIG = {
    "enabled": False,  # Экспериментально
    "params": {
        "sequence_length": 60,
        "embed_dim": 64,
        "num_heads": 4,
        "ff_dim": 128,
        "num_blocks": 2,
        "dropout": 0.1,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001
    }
}

# ========== ENSEMBLE ==========
ENSEMBLE_CONFIG = {
    "enabled": True,
    "voting": "soft",  # 'soft' or 'hard'
    "weights": {
        "random_forest": 0.3,
        "xgboost": 0.4,
        "lightgbm": 0.3
    },
    "stacking": {
        "enabled": True,
        "cv": 5,
        "final_estimator": "logistic_regression"
    }
}

# ========== ГИБРИДНЫЕ МОДЕЛИ ==========
HYBRID_CONFIG = {
    "cnn_lstm": {
        "enabled": True,
        "cnn_filters": [64, 128],
        "kernel_sizes": [3, 3],
        "pool_sizes": [2, 2],
        "lstm_units": 100
    },
    "attention_lstm": {
        "enabled": True,
        "lstm_units": 100,
        "attention_type": "dot"  # 'dot', 'general', 'concat'
    }
}

# ========== ОБЩАЯ КОНФИГУРАЦИЯ ==========
MODEL_CONFIG = {
    "random_forest": RANDOM_FOREST_CONFIG,
    "xgboost": XGBOOST_CONFIG,
    "lightgbm": LIGHTGBM_CONFIG,
    "lstm": LSTM_CONFIG,
    "gru": GRU_CONFIG,
    "transformer": TRANSFORMER_CONFIG,
    "ensemble": ENSEMBLE_CONFIG,
    "hybrid": HYBRID_CONFIG
}

# Настройки обучения
TRAINING_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.2,
    "cross_validation_folds": 5,
    "use_grid_search": False,
    "grid_search_cv": 3,
    "save_best_model": True,
    "model_save_path": "models/saved/",
    "random_state": 42,
    "n_jobs": -1
}

# Настройки предсказаний
PREDICTION_CONFIG = {
    "threshold": 0.5,
    "use_probabilities": True,
    "min_confidence": 0.6,
    "ensemble_threshold": 0.5,
    "use_majority_vote": False
}

# Метрики для оценки
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "log_loss",
    "confusion_matrix"
]
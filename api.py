
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ==========================================
# 1. НАСТРОЙКА ПРИЛОЖЕНИЯ
# ==========================================

app = FastAPI(
    title="QualityPredict API",
    version="1.0.0",
    description=(
        "HTTP API для анализа производственных данных "
        "и классификации уровня дефектности."
    )
)

# Для локальной разработки.
# Если запросы идут через Go, CORS обычно
# настраивается на стороне Go-сервера.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ==========================================
# 2. ПУТИ К ФАЙЛАМ
# ==========================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "manufacturing_defect_dataset.csv"
)

MODEL_PATH = (
    ROOT
    / "src"
    / "models"
    / "saved"
    / "qualitypredict_final.joblib"
)


# ==========================================
# 3. ЗАГРУЗКА ДАННЫХ И МОДЕЛИ
# ==========================================

@lru_cache(maxsize=1)
def load_data():
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Файл датасета не найден"
        )

    try:
        return pd.read_csv(DATA_PATH)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка чтения датасета: {error}"
        )


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Файл обученной модели не найден"
        )

    try:
        # Загружать joblib можно только из
        # доверенного источника.
        artifact = joblib.load(MODEL_PATH)

        required_keys = {
            "model",
            "threshold",
            "features",
            "model_name",
        }

        if not isinstance(artifact, dict):
            raise ValueError(
                "Неверный формат файла модели"
            )

        if not required_keys.issubset(artifact):
            raise ValueError(
                "В файле модели отсутствуют "
                "обязательные поля"
            )

        if not 0 <= float(artifact["threshold"]) <= 1:
            raise ValueError(
                "Некорректный порог классификации"
            )

        return artifact

    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Ошибка загрузки модели: {error}"
        )


# ==========================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def calculate_group_statistics(group):
    total = len(group)

    if total == 0:
        return {
            "total": 0,
            "high_defect": 0,
            "low_defect": 0,
            "high_defect_share": None,
        }

    high_count = int(
        (group["DefectStatus"] == 1).sum()
    )

    low_count = int(
        (group["DefectStatus"] == 0).sum()
    )

    return {
        "total": int(total),
        "high_defect": high_count,
        "low_defect": low_count,
        "high_defect_share": round(
            high_count / total, 4
        ),
    }


# ==========================================
# 5. ПРОВЕРКА РАБОТЫ API
# ==========================================

@app.get("/")
def root():
    return {
        "service": "QualityPredict API",
        "version": "1.0.0",
        "status": "running",
    }


# ==========================================
# 6. ОБЩАЯ СТАТИСТИКА
# ==========================================

@app.get("/api/dashboard")
def get_dashboard():
    df = load_data()

    counts = df["DefectStatus"].value_counts()

    maintenance = (
        df.groupby("DefectStatus")["MaintenanceHours"]
        .mean()
    )

    volume = (
        df.groupby("DefectStatus")["ProductionVolume"]
        .mean()
    )

    return {
        "total": int(len(df)),

        "class_distribution": {
            "low": int(counts.get(0, 0)),
            "high": int(counts.get(1, 0)),
        },

        "maintenance_by_class": {
            "low": round(
                float(maintenance.get(0, 0)), 2
            ),
            "high": round(
                float(maintenance.get(1, 0)), 2
            ),
        },

        "production_volume_by_class": {
            "low": round(
                float(volume.get(0, 0)), 2
            ),
            "high": round(
                float(volume.get(1, 0)), 2
            ),
        },
    }


# ==========================================
# 7. ГРУППЫ ПО ВРЕМЕНИ ОБСЛУЖИВАНИЯ
# ==========================================

@app.get("/api/maintenance-groups")
def get_maintenance_groups():
    df = load_data()

    threshold = 11

    below = df[
        df["MaintenanceHours"] < threshold
    ]

    at_or_above = df[
        df["MaintenanceHours"] >= threshold
    ]

    return {
        "threshold_hours_per_week": threshold,

        "below_threshold":
            calculate_group_statistics(below),

        "at_or_above_threshold":
            calculate_group_statistics(at_or_above),
    }


# ==========================================
# 8. ГРУППЫ ПО ОБЪЁМУ ПРОИЗВОДСТВА
# ==========================================

@app.get("/api/production-groups")
def get_production_groups():
    df = load_data()

    threshold = 800

    below = df[
        df["ProductionVolume"] < threshold
    ]

    at_or_above = df[
        df["ProductionVolume"] >= threshold
    ]

    return {
        "threshold_units_per_day": threshold,

        "below_threshold":
            calculate_group_statistics(below),

        "at_or_above_threshold":
            calculate_group_statistics(at_or_above),
    }


# ==========================================
# 9. ФОРМАТ ВХОДНЫХ ДАННЫХ ДЛЯ ML
# ==========================================

class ProductionInput(BaseModel):
    ProductionVolume: float = Field(ge=0)
    ProductionCost: float = Field(ge=0)

    SupplierQuality: float = Field(
        ge=0, le=100
    )

    DeliveryDelay: float = Field(ge=0)
    MaintenanceHours: float = Field(ge=0)

    DowntimePercentage: float = Field(
        ge=0, le=100
    )

    InventoryTurnover: float = Field(ge=0)

    # Доля: например, 5% = 0.05
    StockoutRate: float = Field(
        ge=0, le=1
    )

    WorkerProductivity: float = Field(
        ge=0, le=100
    )

    SafetyIncidents: float = Field(ge=0)
    EnergyConsumption: float = Field(ge=0)
    EnergyEfficiency: float = Field(ge=0)
    AdditiveProcessTime: float = Field(ge=0)
    AdditiveMaterialCost: float = Field(ge=0)


# ==========================================
# 10. КЛАССИФИКАЦИЯ
# ==========================================

@app.post("/api/predict")
def predict_defect(data: ProductionInput):

    artifact = load_model()

    classifier = artifact["model"]
    threshold = float(artifact["threshold"])
    features = artifact["features"]

    values = data.model_dump()

    # Проверяем соответствие признаков
    # модели и входных данных.
    missing = [
        feature
        for feature in features
        if feature not in values
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Несовпадение признаков модели: "
                + ", ".join(missing)
            ),
        )

    # Соблюдаем порядок признаков,
    # использованный при обучении.
    X = pd.DataFrame(
        [[values[name] for name in features]],
        columns=features,
    )

    try:
        class_index = list(
            classifier.classes_
        ).index(1)

        score = float(
            classifier.predict_proba(X)[0][class_index]
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Ошибка выполнения классификации"
        )

    predicted_class = int(
        score >= threshold
    )

    label = (
        "Высокий уровень дефектности"
        if predicted_class == 1
        else "Низкий уровень дефектности"
    )

    return {
        "predicted_class": predicted_class,
        "label": label,
        "score": round(score, 4),
        "threshold": round(threshold, 4),
        "model": artifact["model_name"],
    }

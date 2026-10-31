"""
FastAPI-застосунок для демонстраційного вебінтерфейсу системи
автоматизованого діагностування хвороб рослинних культур.

Файл відповідає за:
- завантаження збережених компонентів моделі;
- відображення HTML-сторінки користувацького інтерфейсу;
- приймання зображення від користувача;
- попередню перевірку якості зображення;
- формування гібридного вектора ознак;
- отримання прогнозу моделі та рекомендації.
"""

# Робота з файловою системою потрібна для побудови шляхів до моделей.
from pathlib import Path

# OpenCV використовується для декодування завантаженого зображення.
import cv2

# joblib застосовується для завантаження збережених sklearn-моделей.
import joblib

# NumPy потрібний для перетворення байтів зображення у масив.
import numpy as np

# FastAPI забезпечує серверну частину вебзастосунку та API-маршрути.
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse

# Jinja2Templates використовується для повернення HTML-сторінки.
from fastapi.templating import Jinja2Templates

# StaticFiles дозволяє підключати CSS, JavaScript та інші статичні файли.
from fastapi.staticfiles import StaticFiles

# Імпортуємо функції попередньої обробки та формування ознак.
from src.plant_diagnosis.image_processing import (
    preprocess_image,
    check_image_quality,
    extract_full_feature_vector
)

# Модуль рекомендацій перетворює технічну назву класу у зрозумілу відповідь.
from src.plant_diagnosis.recommendations import get_recommendation


# =========================
# Global settings
# =========================

# Каталог, у якому зберігаються навчені компоненти моделі.
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_PATH = PROJECT_ROOT / "models"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Поріг впевненості: якщо максимальна ймовірність нижча,
# результат повертається зі статусом uncertain.
CONFIDENCE_THRESHOLD = 0.55


# =========================
# FastAPI application setup
# =========================

# Створення FastAPI-застосунку з описом, який видно у документації сервісу.
app = FastAPI(
    title="Plant Disease Diagnosis",
    description="Вебсайт для автоматизованого діагностування хвороб рослинних культур",
    version="2.0.0"
)

# Підключення каталогу HTML-шаблонів.
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

# Підключення каталогу static, де зберігаються styles.css та app.js.
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")


# =========================
# Loading trained components
# =========================

MODEL_FILES = (
    "hybrid_bovw_kmeans.joblib",
    "hybrid_classifier.joblib",
    "hybrid_label_encoder.joblib",
)
missing_models = [name for name in MODEL_FILES if not (MODELS_PATH / name).is_file()]
if missing_models:
    raise RuntimeError(
        "Missing trained artifacts: " + ", ".join(missing_models)
        + ". See README.md for dataset and training prerequisites."
    )

# Словник візуальних слів MiniBatchKMeans для BoVW-ознак.
kmeans = joblib.load(MODELS_PATH / "hybrid_bovw_kmeans.joblib")

# Фінальний ансамблевий класифікатор VotingClassifier.
classifier = joblib.load(MODELS_PATH / "hybrid_classifier.joblib")

# LabelEncoder потрібний для перетворення числових міток у назви класів.
label_encoder = joblib.load(MODELS_PATH / "hybrid_label_encoder.joblib")


# =========================
# Web routes
# =========================

@app.get("/")
def home(request: Request):
    """
    Відображення головної сторінки вебінтерфейсу.

    Маршрут повертає HTML-шаблон index.html, у якому користувач
    може завантажити зображення та запустити класифікацію.
    """

    return templates.TemplateResponse(
        request,
        "index.html"
    )


@app.get("/health")
def health():
    """
    Технічний маршрут для перевірки працездатності сервісу.

    Його можна використовувати для швидкої перевірки, що FastAPI
    застосунок запущений і відповідає на запити.
    """

    return {
        "status": "ok",
        "message": "Plant Disease Diagnosis web service is running"
    }


# =========================
# Image classification logic
# =========================

def classify_image(image: np.ndarray) -> dict:
    """
    Класифікація одного зображення.

    Функція використовується серверним маршрутом /predict і виконує
    повний цикл застосування моделі: перевірку якості, формування
    ознак, прогнозування класу та підготовку рекомендації.
    """

    # Попередня обробка повертає кольорове зображення та grayscale-версію.
    # Для перевірки розмитості достатньо зображення у відтінках сірого.
    _, gray = preprocess_image(image)

    # Якщо зображення занадто розмите, модель не запускається.
    # Це зменшує ризик отримання некоректного прогнозу через погану якість фото.
    if not check_image_quality(gray):
        return {
            "status": "bad_quality",
            "message": "Зображення занадто розмите або непридатне для аналізу. Зробіть фото повторно.",
            "predicted_class": None,
            "display_name": None,
            "confidence": 0.0,
            "top_predictions": [],
            "recommendation": "Спробуйте зробити чіткіше фото листка при достатньому освітленні."
        }

    # Формування гібридного вектора ознак:
    # ORB/BoVW + HSV histogram + color statistics + LBP + disease area.
    feature_vector = extract_full_feature_vector(image, kmeans).reshape(1, -1)

    # Для VotingClassifier використовується predict_proba,
    # щоб отримати ймовірності для всіх класів.
    probabilities = classifier.predict_proba(feature_vector)[0]

    # Індекс класу з найбільшою ймовірністю вважається основним прогнозом.
    predicted_index = int(np.argmax(probabilities))

    # Рівень впевненості відповідає максимальній ймовірності серед класів.
    confidence = float(probabilities[predicted_index])

    # Перетворення числової мітки назад у технічну назву класу датасету.
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]

    # Визначення трьох найбільш імовірних класів для пояснення результату.
    top_indices = np.argsort(probabilities)[::-1][:3]

    # Список top-3 прогнозів буде відображений у вебінтерфейсі.
    top_predictions = []

    for idx in top_indices:
        # Для кожного з top-3 класів отримуємо зрозумілу назву та ймовірність.
        class_name = label_encoder.inverse_transform([idx])[0]
        recommendation_data = get_recommendation(class_name)

        top_predictions.append({
            "class": class_name,
            "display_name": recommendation_data["name"],
            "confidence": round(float(probabilities[idx]), 4),
            "confidence_percent": round(float(probabilities[idx]) * 100, 2)
        })

    # Отримання української назви та рекомендації для основного прогнозу.
    recommendation_data = get_recommendation(predicted_class)

    # Якщо впевненість нижча за встановлений поріг,
    # результат не подається як остаточно надійний діагноз.
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "status": "uncertain",
            "message": "Система не має достатньої впевненості для точного визначення хвороби.",
            "predicted_class": predicted_class,
            "display_name": recommendation_data["name"],
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "top_predictions": top_predictions,
            "recommendation": "Рекомендовано зробити якісніше фото або звернутися до фахівця."
        }

    # Успішний результат повертається у форматі словника,
    # який FastAPI автоматично перетворює у JSON-відповідь.
    return {
        "status": "success",
        "message": "Класифікацію виконано успішно.",
        "predicted_class": predicted_class,
        "display_name": recommendation_data["name"],
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "top_predictions": top_predictions,
        "recommendation": recommendation_data["recommendation"]
    }


# =========================
# Prediction endpoint
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Основний серверний маршрут для класифікації зображення.

    Маршрут приймає файл від вебінтерфейсу, декодує його у формат OpenCV,
    викликає classify_image та повертає результат клієнтській частині.
    """

    # Зчитування всього завантаженого файлу у вигляді байтів.
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={
            "status": "error", "message": "Максимальний розмір файлу — 10 MiB."
        })
    if not image_bytes:
        return JSONResponse(status_code=400, content={
            "status": "error", "message": "Завантажений файл порожній."
        })

    # Перетворення байтів у NumPy-масив, який може бути декодований OpenCV.
    np_array = np.frombuffer(image_bytes, np.uint8)

    # Декодування JPG/PNG/JPEG у BGR-зображення OpenCV.
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Якщо OpenCV не зміг декодувати файл, він не є коректним зображенням.
    if image is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Не вдалося зчитати зображення. Завантажте файл у форматі JPG, JPEG або PNG."
            }
        )

    # Якщо зображення зчитане успішно, передаємо його у модельну логіку.
    result = classify_image(image)

    # Повернений словник автоматично передається браузеру як JSON.
    # JavaScript на клієнтській стороні відображає його як блок результату.
    return result

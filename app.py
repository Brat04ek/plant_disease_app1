import os
import cv2
import joblib
import numpy as np

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.plant_diagnosis.image_processing import (
    preprocess_image,
    check_image_quality,
    extract_full_feature_vector
)

from src.plant_diagnosis.recommendations import get_recommendation


MODELS_PATH = "models"
CONFIDENCE_THRESHOLD = 0.55

app = FastAPI(
    title="Plant Disease Diagnosis",
    description="Вебсайт для автоматизованого діагностування хвороб рослинних культур",
    version="2.0.0"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


kmeans = joblib.load(os.path.join(MODELS_PATH, "hybrid_bovw_kmeans.joblib"))
classifier = joblib.load(os.path.join(MODELS_PATH, "hybrid_classifier.joblib"))
label_encoder = joblib.load(os.path.join(MODELS_PATH, "hybrid_label_encoder.joblib"))


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html"
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Plant Disease Diagnosis web service is running"
    }


def classify_image(image: np.ndarray) -> dict:
    """
    Класифікація одного зображення.
    Використовується і для вебінтерфейсу, і для API.
    """

    _, gray = preprocess_image(image)

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

    feature_vector = extract_full_feature_vector(image, kmeans).reshape(1, -1)

    probabilities = classifier.predict_proba(feature_vector)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]

    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = []

    for idx in top_indices:
        class_name = label_encoder.inverse_transform([idx])[0]
        recommendation_data = get_recommendation(class_name)

        top_predictions.append({
            "class": class_name,
            "display_name": recommendation_data["name"],
            "confidence": round(float(probabilities[idx]), 4),
            "confidence_percent": round(float(probabilities[idx]) * 100, 2)
        })

    recommendation_data = get_recommendation(predicted_class)

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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Не вдалося зчитати зображення. Завантажте файл у форматі JPG, JPEG або PNG."
            }
        )

    result = classify_image(image)
    return result
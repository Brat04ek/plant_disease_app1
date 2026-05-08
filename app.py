import os
import cv2
import joblib
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from image_processing import (
    preprocess_image,
    check_image_quality,
    extract_full_feature_vector
)

from recommendations import get_recommendation


MODELS_PATH = "models"
CONFIDENCE_THRESHOLD = 0.55

app = FastAPI(
    title="Plant Disease Diagnosis API",
    description="API для автоматизованого діагностування хвороб рослинних культур",
    version="1.1.0"
)


kmeans = joblib.load(os.path.join(MODELS_PATH, "hybrid_bovw_kmeans.joblib"))
classifier = joblib.load(os.path.join(MODELS_PATH, "hybrid_classifier.joblib"))
label_encoder = joblib.load(os.path.join(MODELS_PATH, "hybrid_label_encoder.joblib"))


@app.get("/")
def root():
    return {
        "message": "Hybrid Plant Disease Diagnosis API is running"
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
                "error": "Не вдалося зчитати зображення."
            }
        )

    _, gray = preprocess_image(image)

    if not check_image_quality(gray):
        return {
            "status": "bad_quality",
            "message": "Зображення занадто розмите або непридатне для аналізу. Зробіть фото повторно.",
            "predicted_class": None,
            "confidence": 0.0
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
        top_predictions.append({
            "class": class_name,
            "confidence": round(float(probabilities[idx]), 4)
        })

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "status": "uncertain",
            "message": "Система не має достатньої впевненості для точного визначення хвороби.",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "top_predictions": top_predictions,
            "recommendation": "Рекомендовано зробити якісніше фото або звернутися до фахівця."
        }

    recommendation_data = get_recommendation(predicted_class)

    return {
        "status": "success",
        "predicted_class": predicted_class,
        "display_name": recommendation_data["name"],
        "confidence": round(confidence, 4),
        "top_predictions": top_predictions,
        "recommendation": recommendation_data["recommendation"]
    }
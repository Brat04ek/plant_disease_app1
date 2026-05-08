import os
import cv2
import joblib
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from image_processing import extract_orb_descriptors, descriptors_to_bovw


DATASET_PATH = "dataset/PlantVillage"
MODELS_PATH = "models"

N_CLUSTERS = 100
MAX_IMAGES_PER_CLASS = 500


def load_dataset(dataset_path: str):
    image_paths = []
    labels = []

    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_folder):
            continue

        count = 0

        for file_name in os.listdir(class_folder):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_paths.append(os.path.join(class_folder, file_name))
            labels.append(class_name)

            count += 1
            if count >= MAX_IMAGES_PER_CLASS:
                break

    return image_paths, labels


def collect_descriptors(image_paths):
    all_descriptors = []

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        if image is None:
            continue

        _, descriptors, _ = extract_orb_descriptors(image)

        if descriptors is not None:
            all_descriptors.append(descriptors)

        if idx % 100 == 0:
            print(f"Processed descriptors: {idx}/{len(image_paths)}")

    if len(all_descriptors) == 0:
        raise ValueError("Не знайдено дескрипторів. Перевірте датасет.")

    return np.vstack(all_descriptors)


def build_feature_matrix(image_paths, kmeans_model):
    features = []

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        if image is None:
            features.append(np.zeros(kmeans_model.n_clusters))
            continue

        _, descriptors, _ = extract_orb_descriptors(image)
        feature_vector = descriptors_to_bovw(descriptors, kmeans_model)

        features.append(feature_vector.flatten())

        if idx % 100 == 0:
            print(f"Built features: {idx}/{len(image_paths)}")

    return np.array(features)


def main():
    os.makedirs(MODELS_PATH, exist_ok=True)

    print("Loading dataset...")
    image_paths, labels = load_dataset(DATASET_PATH)

    print(f"Images found: {len(image_paths)}")
    print(f"Classes found: {len(set(labels))}")

    print("Collecting ORB descriptors...")
    descriptors = collect_descriptors(image_paths)

    print("Training Bag of Visual Words model...")
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        batch_size=2048
    )
    kmeans.fit(descriptors)

    print("Building feature vectors...")
    X = build_feature_matrix(image_paths, kmeans)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training classifier...")
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(kmeans, os.path.join(MODELS_PATH, "bovw_kmeans.joblib"))
    joblib.dump(classifier, os.path.join(MODELS_PATH, "classifier.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_PATH, "label_encoder.joblib"))

    print("Models saved successfully.")


if __name__ == "__main__":
    main()
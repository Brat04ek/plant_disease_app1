import os
import cv2
import joblib
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from image_processing import (
    extract_orb_descriptors,
    extract_full_feature_vector
)


DATASET_PATH = "dataset/PlantVillage"
MODELS_PATH = "models"

N_CLUSTERS = 150
MAX_IMAGES_PER_CLASS = 500
TEST_SIZE = 0.2
RANDOM_STATE = 7


def load_dataset(dataset_path: str):
    """
    Завантаження шляхів до зображень та їх класів.
    Очікується структура:
    dataset/PlantVillage/class_name/image.jpg
    """

    image_paths = []
    labels = []

    for class_name in sorted(os.listdir(dataset_path)):
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


def collect_orb_descriptors(image_paths):
    """
    Збір ORB-дескрипторів з навчальних зображень
    для побудови словника візуальних слів.
    """

    all_descriptors = []

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        if image is None:
            continue

        _, descriptors, _, _ = extract_orb_descriptors(image)

        if descriptors is not None and len(descriptors) > 0:
            all_descriptors.append(descriptors)

        if idx % 100 == 0:
            print(f"Collected descriptors from {idx}/{len(image_paths)} images")

    if len(all_descriptors) == 0:
        raise ValueError("Не знайдено ORB-дескрипторів. Перевірте датасет.")

    return np.vstack(all_descriptors)


def build_feature_matrix(image_paths, kmeans_model):
    """
    Формування матриці комбінованих ознак для всіх зображень.
    """

    features = []

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: cannot read image {image_path}")
            continue

        feature_vector = extract_full_feature_vector(image, kmeans_model)
        features.append(feature_vector)

        if idx % 100 == 0:
            print(f"Built features for {idx}/{len(image_paths)} images")

    return np.array(features, dtype=np.float32)


def main():
    os.makedirs(MODELS_PATH, exist_ok=True)

    print("Loading dataset...")
    image_paths, labels = load_dataset(DATASET_PATH)

    print(f"Images found: {len(image_paths)}")
    print(f"Classes found: {len(set(labels))}")
    print("Classes:")
    for class_name in sorted(set(labels)):
        print(" -", class_name)

    if len(image_paths) == 0:
        raise ValueError("Датасет порожній або шлях неправильний.")

    print("\nSplitting dataset...")
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print(f"Train images: {len(train_paths)}")
    print(f"Test images: {len(test_paths)}")

    print("\nCollecting ORB descriptors for BoVW vocabulary...")
    train_descriptors = collect_orb_descriptors(train_paths)

    print(f"Total descriptors shape: {train_descriptors.shape}")

    print("\nTraining MiniBatchKMeans visual vocabulary...")
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init="auto"
    )

    kmeans.fit(train_descriptors)

    print("\nBuilding train feature matrix...")
    X_train = build_feature_matrix(train_paths, kmeans)

    print("\nBuilding test feature matrix...")
    X_test = build_feature_matrix(test_paths, kmeans)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)

    print(f"\nFeature vector size: {X_train.shape[1]}")

    print("\nTraining classifiers...")

    rf_model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_depth=None
    )

    extra_trees_model = ExtraTreesClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            C=10,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    voting_model = VotingClassifier(
        estimators=[
            ("rf", rf_model),
            ("extra", extra_trees_model),
            ("svm", svm_model)
        ],
        voting="soft",
        n_jobs=-1
    )

    voting_model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = voting_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nSaving models...")
    joblib.dump(kmeans, os.path.join(MODELS_PATH, "hybrid_bovw_kmeans.joblib"))
    joblib.dump(voting_model, os.path.join(MODELS_PATH, "hybrid_classifier.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_PATH, "hybrid_label_encoder.joblib"))

    print("\nModels saved successfully.")


if __name__ == "__main__":
    main()
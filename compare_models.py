import os
import time
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from image_processing import extract_full_feature_vector


DATASET_PATH = "dataset/PlantVillage"
MODELS_PATH = "models"
PHOTO_PATH = "photo"

MAX_IMAGES_PER_CLASS = 500
TEST_SIZE = 0.2
RANDOM_STATE = 7


def load_dataset(dataset_path: str):
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


def build_feature_matrix(image_paths, kmeans_model):
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


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {model_name}...")

    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_pred

    accuracy = accuracy_score(y_test, y_pred)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1_score": f1_score,
        "training_time_seconds": train_time,
        "prediction_time_seconds": prediction_time
    }


def save_model_comparison_plot(df):
    os.makedirs(PHOTO_PATH, exist_ok=True)

    x = range(len(df))

    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], df["accuracy"], width=0.2, label="Accuracy")
    plt.bar(x, df["macro_precision"], width=0.2, label="Macro precision")
    plt.bar([i + 0.2 for i in x], df["macro_f1_score"], width=0.2, label="Macro F1-score")

    plt.title("Model comparison")
    plt.xlabel("Model")
    plt.ylabel("Metric value")
    plt.ylim(0, 1.05)
    plt.xticks(x, df["model"], rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(PHOTO_PATH, "model_comparison.jpg"), dpi=300)
    plt.close()


def main():
    os.makedirs(PHOTO_PATH, exist_ok=True)

    print("Loading dataset...")
    image_paths, labels = load_dataset(DATASET_PATH)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print("Loading KMeans visual vocabulary...")
    kmeans = joblib.load(os.path.join(MODELS_PATH, "hybrid_bovw_kmeans.joblib"))

    print("Building train feature matrix...")
    X_train = build_feature_matrix(train_paths, kmeans)

    print("Building test feature matrix...")
    X_test = build_feature_matrix(test_paths, kmeans)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)

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

    models = [
        ("Random Forest", rf_model),
        ("Extra Trees", extra_trees_model),
        ("SVM RBF", svm_model),
        ("Voting Classifier", voting_model)
    ]

    results = []

    for model_name, model in models:
        result = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test
        )
        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False)

    print("\nModel comparison:")
    print(df.to_string(index=False))

    df.to_csv(os.path.join(PHOTO_PATH, "model_comparison.csv"), index=False)
    save_model_comparison_plot(df)

    print("\nSaved:")
    print(os.path.join(PHOTO_PATH, "model_comparison.csv"))
    print(os.path.join(PHOTO_PATH, "model_comparison.jpg"))


if __name__ == "__main__":
    main()
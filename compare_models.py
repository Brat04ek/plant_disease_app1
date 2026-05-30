import os
import time
import cv2
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from image_processing import extract_full_feature_vector


DATASET_PATH = "dataset/PlantVillage"
MODELS_PATH = "models"
RESULTS_PATH = "results"
OUTPUT_PATH = os.path.join(RESULTS_PATH, "model_comparison")

MAX_IMAGES_PER_CLASS = 500
TEST_SIZE = 0.2
RANDOM_STATE = 7

BEST_N_CLUSTERS = 150
BEST_N_ESTIMATORS = 500
BEST_SVM_C = 10


def load_dataset(dataset_path: str, max_images_per_class: int | None = None):
    image_paths = []
    labels = []

    for class_name in sorted(os.listdir(dataset_path)):
        class_folder = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_folder):
            continue

        count = 0

        for file_name in sorted(os.listdir(class_folder)):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_paths.append(os.path.join(class_folder, file_name))
            labels.append(class_name)

            count += 1

            if max_images_per_class is not None and count >= max_images_per_class:
                break

    return np.array(image_paths), np.array(labels)


def build_feature_matrix(image_paths, labels, kmeans_model):
    features = []
    valid_labels = []
    valid_paths = []

    for idx, (image_path, label) in enumerate(zip(image_paths, labels)):
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: cannot read image {image_path}")
            continue

        feature_vector = extract_full_feature_vector(image, kmeans_model)

        features.append(feature_vector)
        valid_labels.append(label)
        valid_paths.append(image_path)

        if idx % 100 == 0:
            print(f"Built features for {idx}/{len(image_paths)} images")

    return (
        np.array(features, dtype=np.float32),
        np.array(valid_labels),
        np.array(valid_paths)
    )


def create_models():
    rf_model = RandomForestClassifier(
        n_estimators=BEST_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_depth=None
    )

    extra_trees_model = ExtraTreesClassifier(
        n_estimators=BEST_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            C=BEST_SVM_C,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    voting_model = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=BEST_N_ESTIMATORS,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
                max_depth=None
            )),
            ("extra", ExtraTreesClassifier(
                n_estimators=BEST_N_ESTIMATORS,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            )),
            ("svm", Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(
                    C=BEST_SVM_C,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ))
            ]))
        ],
        voting="soft",
        n_jobs=1
    )

    return [
        ("Random Forest", rf_model),
        ("Extra Trees", extra_trees_model),
        ("SVM RBF", svm_model),
        ("Voting Classifier", voting_model)
    ]


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1_score": weighted_f1
    }


def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    safe_name = model_name.lower().replace(" ", "_")

    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.to_csv(os.path.join(OUTPUT_PATH, f"{safe_name}_confusion_matrix.csv"))

    fig, ax = plt.subplots(figsize=(12, 10))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=True
    )

    plt.title(f"Confusion matrix — {model_name}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_PATH, f"{safe_name}_confusion_matrix.jpg"),
        dpi=300
    )

    plt.close()


def save_classification_report(y_true, y_pred, class_names, model_name):
    safe_name = model_name.lower().replace(" ", "_")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        os.path.join(OUTPUT_PATH, f"{safe_name}_classification_report.csv")
    )


def save_model_comparison_plots(results_df):
    df = results_df.sort_values("macro_f1_score", ascending=False).reset_index(drop=True)

    df.to_csv(os.path.join(OUTPUT_PATH, "model_comparison_results.csv"), index=False)

    x = range(len(df))

    # Graph 1: Macro F1-score only
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, df["macro_f1_score"])

    plt.title("Порівняння моделей за macro F1-score")
    plt.xlabel("Модель")
    plt.ylabel("Macro F1-score")

    min_value = max(0.0, df["macro_f1_score"].min() - 0.03)
    max_value = min(1.0, df["macro_f1_score"].max() + 0.03)
    plt.ylim(min_value, max_value)

    plt.xticks(x, df["model"], rotation=15, ha="right")

    for bar, value in zip(bars, df["macro_f1_score"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison_macro_f1.jpg"), dpi=300)
    plt.close()

    # Graph 2: Accuracy, macro recall, macro F1
    plt.figure(figsize=(11, 6))

    plt.bar([i - 0.25 for i in x], df["accuracy"], width=0.25, label="Accuracy")
    plt.bar(x, df["macro_recall"], width=0.25, label="Macro recall")
    plt.bar([i + 0.25 for i in x], df["macro_f1_score"], width=0.25, label="Macro F1-score")

    plt.title("Порівняння моделей за основними метриками")
    plt.xlabel("Модель")
    plt.ylabel("Значення метрики")

    metric_min = min(
        df["accuracy"].min(),
        df["macro_recall"].min(),
        df["macro_f1_score"].min()
    )
    plt.ylim(max(0.0, metric_min - 0.03), 1.0)

    plt.xticks(x, df["model"], rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison_main_metrics.jpg"), dpi=300)
    plt.close()

    # Graph 3: Training and prediction time
    plt.figure(figsize=(10, 6))

    plt.bar(x, df["training_time_seconds"])

    plt.title("Порівняння часу навчання моделей")
    plt.xlabel("Модель")
    plt.ylabel("Час навчання, с")
    plt.xticks(x, df["model"], rotation=15, ha="right")

    for i, value in enumerate(df["training_time_seconds"]):
        plt.text(
            i,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison_training_time.jpg"), dpi=300)
    plt.close()


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("=" * 80)
    print("Model comparison with best validated parameters")
    print("=" * 80)

    print(f"K = {BEST_N_CLUSTERS}")
    print(f"trees = {BEST_N_ESTIMATORS}")
    print(f"C = {BEST_SVM_C}")

    print("\nLoading dataset...")
    image_paths, labels = load_dataset(
        DATASET_PATH,
        max_images_per_class=MAX_IMAGES_PER_CLASS
    )

    print(f"Images found: {len(image_paths)}")
    print(f"Classes found: {len(set(labels))}")

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print(f"Train + validation images: {len(train_val_paths)}")
    print(f"Test images: {len(test_paths)}")

    print("\nLoading final KMeans visual vocabulary...")
    kmeans = joblib.load(os.path.join(MODELS_PATH, "hybrid_bovw_kmeans.joblib"))

    print("\nBuilding train_val feature matrix...")
    X_train_val, valid_train_val_labels, _ = build_feature_matrix(
        train_val_paths,
        train_val_labels,
        kmeans
    )

    print("\nBuilding test feature matrix...")
    X_test, valid_test_labels, _ = build_feature_matrix(
        test_paths,
        test_labels,
        kmeans
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    y_train_val = label_encoder.transform(valid_train_val_labels)
    y_test = label_encoder.transform(valid_test_labels)

    class_names = label_encoder.classes_

    models = create_models()

    results = []

    for model_name, model in models:
        print("\n" + "-" * 80)
        print(f"Training and evaluating: {model_name}")
        print("-" * 80)

        start_train = time.time()
        model.fit(X_train_val, y_train_val)
        training_time = time.time() - start_train

        start_pred = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_pred

        metrics = calculate_metrics(y_test, y_pred)

        row = {
            "model": model_name,
            "n_clusters": BEST_N_CLUSTERS,
            "n_estimators": BEST_N_ESTIMATORS if model_name != "SVM RBF" else None,
            "svm_C": BEST_SVM_C if model_name in ["SVM RBF", "Voting Classifier"] else None,
            "train_val_size": len(y_train_val),
            "test_size": len(y_test),
            "training_time_seconds": training_time,
            "prediction_time_seconds": prediction_time,
            **metrics
        }

        results.append(row)

        save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            model_name=model_name
        )

        save_classification_report(
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            model_name=model_name
        )

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1-score: {metrics['macro_f1_score']:.4f}")
        print(f"Training time: {training_time:.2f} s")
        print(f"Prediction time: {prediction_time:.2f} s")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("macro_f1_score", ascending=False)

    print("\nModel comparison results:")
    print(results_df.to_string(index=False))

    save_model_comparison_plots(results_df)

    print("\nSaved results to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
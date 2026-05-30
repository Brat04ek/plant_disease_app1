import os
import json
import time
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from image_processing import (
    extract_orb_descriptors,
    extract_full_feature_vector
)


# =========================
# Global settings
# =========================

DATASET_PATH = "dataset/PlantVillage"
MODELS_PATH = "models"
RESULTS_PATH = "results"

MAX_IMAGES_PER_CLASS = 500

TEST_SIZE = 0.2
N_SPLITS = 8  # 80% train_val -> 7/8 train ≈ 70%, 1/8 validation ≈ 10%
RANDOM_STATE = 7

KMEANS_BATCH_SIZE = 4096

PARAM_CONFIGS = [
    {
        "config_name": "config_1",
        "n_clusters": 100,
        "n_estimators": 300,
        "svm_C": 1
    },
    {
        "config_name": "config_2",
        "n_clusters": 150,
        "n_estimators": 300,
        "svm_C": 10
    },
    {
        "config_name": "config_3",
        "n_clusters": 150,
        "n_estimators": 500,
        "svm_C": 10
    },
    {
        "config_name": "config_4",
        "n_clusters": 200,
        "n_estimators": 300,
        "svm_C": 10
    },
    {
        "config_name": "config_5",
        "n_clusters": 150,
        "n_estimators": 500,
        "svm_C": 30
    }
]


# =========================
# Dataset loading
# =========================

def load_dataset(dataset_path: str, max_images_per_class: int | None = None):
    """
    Завантаження шляхів до зображень та їх класів.

    Очікувана структура:
    dataset/PlantVillage/class_name/image.jpg
    """

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


def save_dataset_statistics(labels, output_path: str):
    """
    Збереження інформації про розподіл класів у використаному наборі даних.
    """

    class_counts = Counter(labels)

    df = pd.DataFrame({
        "class_name": list(class_counts.keys()),
        "count": list(class_counts.values())
    }).sort_values("class_name")

    df.to_csv(os.path.join(output_path, "dataset_class_distribution.csv"), index=False)

    plt.figure(figsize=(14, 7))
    plt.bar(df["class_name"], df["count"])
    plt.title("Dataset class distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "dataset_class_distribution.jpg"), dpi=300)
    plt.close()

    return df


def save_split_distribution(train_val_labels, test_labels, class_names, output_path: str):
    """
    Збереження таблиці та графіка розподілу класів у train_val та test.
    """

    train_val_counts = Counter(train_val_labels)
    test_counts = Counter(test_labels)

    rows = []

    for class_name in class_names:
        rows.append({
            "class_name": class_name,
            "train_val_count": train_val_counts[class_name],
            "test_count": test_counts[class_name],
            "total_count": train_val_counts[class_name] + test_counts[class_name]
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_path, "train_val_test_distribution.csv"), index=False)

    x = range(len(df))

    plt.figure(figsize=(14, 7))
    plt.bar([i - 0.2 for i in x], df["train_val_count"], width=0.4, label="Train + validation")
    plt.bar([i + 0.2 for i in x], df["test_count"], width=0.4, label="Test")

    plt.title("Train/validation and test distribution by class")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.xticks(x, df["class_name"], rotation=75, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "train_val_test_distribution.jpg"), dpi=300)
    plt.close()

    return df


# =========================
# Feature extraction
# =========================

def collect_orb_descriptors(image_paths):
    """
    Збір ORB-дескрипторів лише з навчальної частини поточного fold.
    Це важливо, щоб не було витоку інформації з validation або test.
    """

    all_descriptors = []

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: cannot read image {image_path}")
            continue

        _, descriptors, _, _ = extract_orb_descriptors(image)

        if descriptors is not None and len(descriptors) > 0:
            all_descriptors.append(descriptors)

        if idx % 100 == 0:
            print(f"Collected ORB descriptors from {idx}/{len(image_paths)} images")

    if len(all_descriptors) == 0:
        raise ValueError("Не знайдено ORB-дескрипторів. Перевірте датасет або image_processing.py.")

    return np.vstack(all_descriptors)


def build_feature_matrix(image_paths, labels, kmeans_model):
    """
    Формування матриці гібридних ознак для списку зображень.

    Повертає:
    - X: матриця ознак;
    - valid_labels: мітки тільки для тих зображень, які вдалося зчитати;
    - valid_paths: шляхи тільки для тих зображень, які вдалося зчитати.
    """

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


# =========================
# Model construction
# =========================

def create_voting_model(n_estimators: int, svm_C: float, random_state: int):
    """
    Створення ансамблевої моделі:
    Random Forest + Extra Trees + SVM RBF у режимі soft voting.
    """

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
        max_depth=None
    )

    extra_trees_model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            C=svm_C,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ))
    ])

    voting_model = VotingClassifier(
        estimators=[
            ("rf", rf_model),
            ("extra", extra_trees_model),
            ("svm", svm_model)
        ],
        voting="soft",
        n_jobs=1
    )

    return voting_model


def evaluate_predictions(y_true, y_pred):
    """
    Обчислення основних метрик.
    Основна метрика для вибору параметрів — macro_f1_score.
    """

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


# =========================
# Cross-validation
# =========================

def run_cross_validation(train_val_paths, train_val_labels, label_encoder, output_path: str):
    """
    Запуск StratifiedKFold для 5 конфігурацій параметрів.
    Підтримує resume: якщо частина fold-ів уже порахована,
    вони пропускаються на основі файлу cv_results_all_folds.csv.
    """

    results_file = os.path.join(output_path, "cv_results_all_folds.csv")

    if os.path.exists(results_file):
        existing_results_df = pd.read_csv(results_file)
        all_fold_results = existing_results_df.to_dict("records")
        print(f"\nResume mode enabled.")
        print(f"Loaded existing CV results: {len(existing_results_df)} rows")
    else:
        existing_results_df = pd.DataFrame()
        all_fold_results = []

    def fold_already_done(config_name: str, fold_idx: int) -> bool:
        if existing_results_df.empty:
            return False

        mask = (
            (existing_results_df["config_name"] == config_name) &
            (existing_results_df["fold"] == fold_idx)
        )

        return bool(mask.any())

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for config in PARAM_CONFIGS:
        config_name = config["config_name"]
        n_clusters = config["n_clusters"]
        n_estimators = config["n_estimators"]
        svm_C = config["svm_C"]

        print("\n" + "=" * 80)
        print(f"Running cross-validation for {config_name}")
        print(f"n_clusters={n_clusters}, n_estimators={n_estimators}, svm_C={svm_C}")
        print("=" * 80)

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(train_val_paths, train_val_labels),
            start=1
        ):
            if fold_already_done(config_name, fold_idx):
                print(f"Skipping {config_name} | Fold {fold_idx}/{N_SPLITS} because it already exists.")
                continue

            print("\n" + "-" * 80)
            print(f"{config_name} | Fold {fold_idx}/{N_SPLITS}")
            print("-" * 80)

            fold_start_time = time.time()

            fold_train_paths = train_val_paths[train_idx]
            fold_val_paths = train_val_paths[val_idx]

            fold_train_labels = train_val_labels[train_idx]
            fold_val_labels = train_val_labels[val_idx]

            print(f"Fold train images: {len(fold_train_paths)}")
            print(f"Fold validation images: {len(fold_val_paths)}")

            print("\nCollecting ORB descriptors for current fold...")
            train_descriptors = collect_orb_descriptors(fold_train_paths)

            print(f"Total descriptors shape: {train_descriptors.shape}")

            print("\nTraining MiniBatchKMeans visual vocabulary...")
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=RANDOM_STATE,
                batch_size=KMEANS_BATCH_SIZE,
                n_init="auto"
            )
            kmeans.fit(train_descriptors)

            print("\nBuilding fold train feature matrix...")
            X_train, valid_train_labels, _ = build_feature_matrix(
                fold_train_paths,
                fold_train_labels,
                kmeans
            )

            print("\nBuilding fold validation feature matrix...")
            X_val, valid_val_labels, _ = build_feature_matrix(
                fold_val_paths,
                fold_val_labels,
                kmeans
            )

            y_train = label_encoder.transform(valid_train_labels)
            y_val = label_encoder.transform(valid_val_labels)

            print(f"\nFeature vector size: {X_train.shape[1]}")

            print("\nTraining VotingClassifier...")
            model = create_voting_model(
                n_estimators=n_estimators,
                svm_C=svm_C,
                random_state=RANDOM_STATE
            )

            model.fit(X_train, y_train)

            print("\nEvaluating on validation fold...")
            y_val_pred = model.predict(X_val)

            metrics = evaluate_predictions(y_val, y_val_pred)

            fold_time = time.time() - fold_start_time

            result_row = {
                "config_name": config_name,
                "fold": fold_idx,
                "n_clusters": n_clusters,
                "n_estimators": n_estimators,
                "svm_C": svm_C,
                "train_size": len(valid_train_labels),
                "validation_size": len(valid_val_labels),
                "feature_vector_size": X_train.shape[1],
                "fold_time_seconds": fold_time,
                **metrics
            }

            all_fold_results.append(result_row)

            current_results_df = pd.DataFrame(all_fold_results)
            current_results_df.to_csv(results_file, index=False)

            print("\nFold metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

            print(f"Fold time: {fold_time:.2f} seconds")
            print(f"Intermediate results saved to: {results_file}")

    cv_results_df = pd.DataFrame(all_fold_results)
    cv_results_df.to_csv(results_file, index=False)

    summary_df = (
        cv_results_df
        .groupby(["config_name", "n_clusters", "n_estimators", "svm_C"], as_index=False)
        .agg({
            "accuracy": ["mean", "std"],
            "macro_precision": ["mean", "std"],
            "macro_recall": ["mean", "std"],
            "macro_f1_score": ["mean", "std"],
            "weighted_f1_score": ["mean", "std"],
            "fold_time_seconds": ["mean", "sum"]
        })
    )

    summary_df.columns = [
        "_".join(col).strip("_") for col in summary_df.columns.values
    ]

    summary_df = summary_df.sort_values("macro_f1_score_mean", ascending=False)

    summary_df.to_csv(os.path.join(output_path, "cv_results_summary.csv"), index=False)

    return cv_results_df, summary_df


# =========================
# Saving plots and reports
# =========================

def save_cv_macro_f1_plot(summary_df, output_path: str):
    """
    Збереження графіка порівняння 5 конфігурацій за macro F1-score.
    """

    df = summary_df.copy()
    df = df.sort_values("macro_f1_score_mean", ascending=False)

    labels = [
        f"{row.config_name}\nK={row.n_clusters}, trees={row.n_estimators}, C={row.svm_C}"
        for row in df.itertuples()
    ]

    x = range(len(df))

    plt.figure(figsize=(12, 7))
    bars = plt.bar(x, df["macro_f1_score_mean"])

    plt.title("Cross-validation comparison by macro F1-score")
    plt.xlabel("Configuration")
    plt.ylabel("Mean macro F1-score")

    min_score = max(0.0, df["macro_f1_score_mean"].min() - 0.03)
    max_score = min(1.0, df["macro_f1_score_mean"].max() + 0.03)
    plt.ylim(min_score, max_score)

    plt.xticks(x, labels, rotation=20, ha="right")

    for bar, value in zip(bars, df["macro_f1_score_mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "cv_macro_f1_comparison.jpg"), dpi=300)
    plt.close()


def save_final_confusion_matrix(y_test, y_pred, class_names, output_path: str):
    """
    Збереження фінальної матриці помилок.
    """

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )
    cm_df.to_csv(os.path.join(output_path, "final_confusion_matrix.csv"))

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

    plt.title("Final test confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "final_confusion_matrix.jpg"), dpi=300)
    plt.close()


def save_final_classification_report(y_test, y_pred, class_names, output_path: str):
    """
    Збереження classification_report у форматі CSV.
    """

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_path, "final_classification_report.csv"))

    class_rows = []

    for class_name in class_names:
        class_rows.append({
            "class_name": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1_score": report[class_name]["f1-score"],
            "support": report[class_name]["support"]
        })

    class_metrics_df = pd.DataFrame(class_rows)
    class_metrics_df.to_csv(
        os.path.join(output_path, "final_class_metrics_by_class.csv"),
        index=False
    )

    x = range(len(class_metrics_df))

    plt.figure(figsize=(14, 7))
    plt.bar([i - 0.25 for i in x], class_metrics_df["precision"], width=0.25, label="Precision")
    plt.bar(x, class_metrics_df["recall"], width=0.25, label="Recall")
    plt.bar([i + 0.25 for i in x], class_metrics_df["f1_score"], width=0.25, label="F1-score")

    plt.title("Final test classification metrics by class")
    plt.xlabel("Class")
    plt.ylabel("Metric value")
    plt.ylim(0, 1.05)
    plt.xticks(x, class_metrics_df["class_name"], rotation=75, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "final_classification_metrics_by_class.jpg"), dpi=300)
    plt.close()

    return report_df, class_metrics_df


# =========================
# Final training and testing
# =========================

def train_final_model_and_evaluate(
    train_val_paths,
    train_val_labels,
    test_paths,
    test_labels,
    label_encoder,
    best_params,
    output_path: str
):
    """
    Навчання фінальної моделі на всіх 80% train_val
    та фінальна оцінка на недоторканих 20% test.
    """

    print("\n" + "=" * 80)
    print("Training final model with best parameters")
    print("=" * 80)

    n_clusters = int(best_params["n_clusters"])
    n_estimators = int(best_params["n_estimators"])
    svm_C = float(best_params["svm_C"])

    print(f"Best n_clusters: {n_clusters}")
    print(f"Best n_estimators: {n_estimators}")
    print(f"Best svm_C: {svm_C}")

    final_start_time = time.time()

    print("\nCollecting ORB descriptors for final BoVW vocabulary...")
    train_val_descriptors = collect_orb_descriptors(train_val_paths)

    print(f"Total descriptors shape: {train_val_descriptors.shape}")

    print("\nTraining final MiniBatchKMeans visual vocabulary...")
    final_kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        batch_size=KMEANS_BATCH_SIZE,
        n_init="auto"
    )

    final_kmeans.fit(train_val_descriptors)

    print("\nBuilding final train_val feature matrix...")
    X_train_val, valid_train_val_labels, valid_train_val_paths = build_feature_matrix(
        train_val_paths,
        train_val_labels,
        final_kmeans
    )

    print("\nBuilding final test feature matrix...")
    X_test, valid_test_labels, valid_test_paths = build_feature_matrix(
        test_paths,
        test_labels,
        final_kmeans
    )

    y_train_val = label_encoder.transform(valid_train_val_labels)
    y_test = label_encoder.transform(valid_test_labels)

    print(f"\nFinal feature vector size: {X_train_val.shape[1]}")

    print("\nTraining final VotingClassifier...")
    final_model = create_voting_model(
        n_estimators=n_estimators,
        svm_C=svm_C,
        random_state=RANDOM_STATE
    )

    final_model.fit(X_train_val, y_train_val)

    print("\nEvaluating final model on untouched test set...")
    y_test_pred = final_model.predict(X_test)

    if hasattr(final_model, "predict_proba"):
        test_probabilities = final_model.predict_proba(X_test)
        test_confidence = np.max(test_probabilities, axis=1)
    else:
        test_confidence = np.full(shape=len(y_test_pred), fill_value=np.nan)

    final_metrics = evaluate_predictions(y_test, y_test_pred)

    final_time = time.time() - final_start_time

    final_metrics_row = {
        "n_clusters": n_clusters,
        "n_estimators": n_estimators,
        "svm_C": svm_C,
        "train_val_size": len(y_train_val),
        "test_size": len(y_test),
        "feature_vector_size": X_train_val.shape[1],
        "final_training_and_test_time_seconds": final_time,
        **final_metrics
    }

    final_metrics_df = pd.DataFrame([final_metrics_row])
    final_metrics_df.to_csv(os.path.join(output_path, "final_test_metrics_summary.csv"), index=False)

    prediction_rows = []

    for image_path, true_label, true_idx, pred_idx, confidence in zip(
        valid_test_paths,
        valid_test_labels,
        y_test,
        y_test_pred,
        test_confidence
    ):
        prediction_rows.append({
            "image_path": image_path,
            "true_class": true_label,
            "predicted_class": label_encoder.inverse_transform([pred_idx])[0],
            "true_label_encoded": int(true_idx),
            "predicted_label_encoded": int(pred_idx),
            "confidence": float(confidence),
            "is_correct": bool(true_idx == pred_idx)
        })

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(os.path.join(output_path, "final_test_predictions.csv"), index=False)

    class_names = label_encoder.classes_

    save_final_confusion_matrix(
        y_test,
        y_test_pred,
        class_names,
        output_path
    )

    save_final_classification_report(
        y_test,
        y_test_pred,
        class_names,
        output_path
    )

    print("\nSaving final models...")
    os.makedirs(MODELS_PATH, exist_ok=True)

    joblib.dump(final_kmeans, os.path.join(MODELS_PATH, "hybrid_bovw_kmeans.joblib"))
    joblib.dump(final_model, os.path.join(MODELS_PATH, "hybrid_classifier.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_PATH, "hybrid_label_encoder.joblib"))

    print("\nFinal test metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"Final training and test time: {final_time:.2f} seconds")

    return final_model, final_kmeans, final_metrics_df


# =========================
# Main
# =========================

def main():
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    full_start_time = time.time()

    print("=" * 80)
    print("Plant disease classification training with StratifiedKFold validation")
    print("=" * 80)

    run_config = {
        "dataset_path": DATASET_PATH,
        "models_path": MODELS_PATH,
        "results_path": RESULTS_PATH,
        "max_images_per_class": MAX_IMAGES_PER_CLASS,
        "test_size": TEST_SIZE,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "kmeans_batch_size": KMEANS_BATCH_SIZE,
        "param_configs": PARAM_CONFIGS,
        "main_selection_metric": "macro_f1_score"
    }

    with open(os.path.join(RESULTS_PATH, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=4)

    print("\nLoading dataset...")
    image_paths, labels = load_dataset(
        DATASET_PATH,
        max_images_per_class=MAX_IMAGES_PER_CLASS
    )

    if len(image_paths) == 0:
        raise ValueError("Датасет порожній або шлях неправильний.")

    print(f"Images found: {len(image_paths)}")
    print(f"Classes found: {len(set(labels))}")

    class_names = sorted(set(labels))

    print("\nClasses:")
    for class_name in class_names:
        print(" -", class_name)

    save_dataset_statistics(labels, RESULTS_PATH)

    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    print("\nCreating untouched test split...")
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print(f"Train + validation images: {len(train_val_paths)}")
    print(f"Test images: {len(test_paths)}")

    save_split_distribution(
        train_val_labels=train_val_labels,
        test_labels=test_labels,
        class_names=label_encoder.classes_,
        output_path=RESULTS_PATH
    )

    print("\nRunning cross-validation on train_val set...")
    cv_results_df, cv_summary_df = run_cross_validation(
        train_val_paths=train_val_paths,
        train_val_labels=train_val_labels,
        label_encoder=label_encoder,
        output_path=RESULTS_PATH
    )

    print("\nCross-validation summary:")
    print(cv_summary_df.to_string(index=False))

    save_cv_macro_f1_plot(cv_summary_df, RESULTS_PATH)

    best_row = cv_summary_df.iloc[0]

    best_params = {
        "config_name": best_row["config_name"],
        "n_clusters": int(best_row["n_clusters"]),
        "n_estimators": int(best_row["n_estimators"]),
        "svm_C": float(best_row["svm_C"]),
        "cv_macro_f1_score_mean": float(best_row["macro_f1_score_mean"]),
        "cv_macro_f1_score_std": float(best_row["macro_f1_score_std"]),
        "cv_accuracy_mean": float(best_row["accuracy_mean"]),
        "cv_accuracy_std": float(best_row["accuracy_std"])
    }

    with open(os.path.join(RESULTS_PATH, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=4)

    print("\nBest parameters:")
    print(json.dumps(best_params, ensure_ascii=False, indent=4))

    train_final_model_and_evaluate(
        train_val_paths=train_val_paths,
        train_val_labels=train_val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        label_encoder=label_encoder,
        best_params=best_params,
        output_path=RESULTS_PATH
    )

    full_time = time.time() - full_start_time

    with open(os.path.join(RESULTS_PATH, "run_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"Total run time seconds: {full_time:.2f}\n")
        f.write(f"Total run time minutes: {full_time / 60:.2f}\n")

    print("\n" + "=" * 80)
    print("Training, validation and final testing completed successfully.")
    print(f"Total run time: {full_time / 60:.2f} minutes")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Models saved to: {MODELS_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
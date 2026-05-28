import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

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

    print("Loading saved models...")
    kmeans = joblib.load(os.path.join(MODELS_PATH, "hybrid_bovw_kmeans.joblib"))
    classifier = joblib.load(os.path.join(MODELS_PATH, "hybrid_classifier.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_PATH, "hybrid_label_encoder.joblib"))

    print("Building test feature matrix...")
    X_test = build_feature_matrix(test_paths, kmeans)

    y_test = label_encoder.transform(test_labels)
    class_names = label_encoder.classes_
    n_classes = len(class_names)

    print("Predicting probabilities...")
    y_score = classifier.predict_proba(X_test)

    y_test_bin = label_binarize(
        y_test,
        classes=np.arange(n_classes)
    )

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_bin.ravel(),
        y_score.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(
        np.concatenate([fpr[i] for i in range(n_classes)])
    )

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 7))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC, AUC = {roc_auc['micro']:.3f}"
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC, AUC = {roc_auc['macro']:.3f}"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.title("Multiclass ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(os.path.join(PHOTO_PATH, "roc_curve_macro_micro.jpg"), dpi=300)
    plt.close()

    auc_rows = []

    for i, class_name in enumerate(class_names):
        auc_rows.append({
            "class_name": class_name,
            "roc_auc": roc_auc[i]
        })

    auc_rows.append({
        "class_name": "micro_average",
        "roc_auc": roc_auc["micro"]
    })

    auc_rows.append({
        "class_name": "macro_average",
        "roc_auc": roc_auc["macro"]
    })

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(os.path.join(PHOTO_PATH, "roc_auc_values.csv"), index=False)

    print("\nSaved:")
    print(os.path.join(PHOTO_PATH, "roc_curve_macro_micro.jpg"))
    print(os.path.join(PHOTO_PATH, "roc_auc_values.csv"))


if __name__ == "__main__":
    main()
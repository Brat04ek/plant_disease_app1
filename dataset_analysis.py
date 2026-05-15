import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_PATH = "dataset/PlantVillage"
OUTPUT_DIR = "photo"

MAX_IMAGES_PER_CLASS = None
TEST_SIZE = 0.2
RANDOM_STATE = 7


def load_dataset(dataset_path: str, max_images_per_class=None):
    """
    Завантажує шляхи до зображень та відповідні назви класів.
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

        for file_name in os.listdir(class_folder):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_paths.append(os.path.join(class_folder, file_name))
            labels.append(class_name)

            count += 1

            if max_images_per_class is not None and count >= max_images_per_class:
                break

    return image_paths, labels


def get_crop_name(class_name: str) -> str:
    """
    Визначає культуру за назвою класу.
    """

    if class_name.startswith("Pepper"):
        return "Pepper"
    if class_name.startswith("Potato"):
        return "Potato"
    if class_name.startswith("Tomato"):
        return "Tomato"

    return "Unknown"


def get_health_status(class_name: str) -> str:
    """
    Визначає стан рослини: healthy або diseased.
    """

    if "healthy" in class_name.lower():
        return "Healthy"

    return "Diseased"


def save_class_distribution(labels):
    """
    Будує стовпчикову діаграму розподілу зображень за класами.
    """

    class_counts = Counter(labels)
    df = pd.DataFrame({
        "class_name": list(class_counts.keys()),
        "count": list(class_counts.values())
    })

    df = df.sort_values("count", ascending=False)

    plt.figure(figsize=(14, 7))
    plt.bar(df["class_name"], df["count"])
    plt.title("Distribution of images by class")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.jpg"), dpi=300)
    plt.close()

    return df


def save_crop_distribution(labels):
    """
    Будує діаграму розподілу зображень за культурами.
    """

    crops = [get_crop_name(label) for label in labels]
    crop_counts = Counter(crops)

    df = pd.DataFrame({
        "crop": list(crop_counts.keys()),
        "count": list(crop_counts.values())
    })

    df = df.sort_values("count", ascending=False)

    plt.figure(figsize=(8, 6))
    plt.bar(df["crop"], df["count"])
    plt.title("Distribution of images by crop")
    plt.xlabel("Crop")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "crop_distribution.jpg"), dpi=300)
    plt.close()

    return df


def save_health_status_distribution(labels):
    """
    Будує діаграму розподілу зображень за станом: здорові / уражені.
    """

    statuses = [get_health_status(label) for label in labels]
    status_counts = Counter(statuses)

    df = pd.DataFrame({
        "status": list(status_counts.keys()),
        "count": list(status_counts.values())
    })

    df = df.sort_values("count", ascending=False)

    plt.figure(figsize=(7, 6))
    plt.bar(df["status"], df["count"])
    plt.title("Distribution of images by plant condition")
    plt.xlabel("Condition")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "health_status_distribution.jpg"), dpi=300)
    plt.close()

    return df


def save_train_test_distribution(image_paths, labels):
    """
    Будує діаграму розподілу класів у навчальній та тестовій вибірках.
    """

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    classes = sorted(set(labels))

    df = pd.DataFrame({
        "class_name": classes,
        "train": [train_counts[class_name] for class_name in classes],
        "test": [test_counts[class_name] for class_name in classes]
    })

    x = range(len(classes))

    plt.figure(figsize=(14, 7))
    plt.bar([i - 0.2 for i in x], df["train"], width=0.4, label="Train")
    plt.bar([i + 0.2 for i in x], df["test"], width=0.4, label="Test")

    plt.title("Train and test distribution by class")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.xticks(x, classes, rotation=75, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "train_test_distribution.jpg"), dpi=300)
    plt.close()

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths, labels = load_dataset(
        DATASET_PATH,
        max_images_per_class=MAX_IMAGES_PER_CLASS
    )

    if len(image_paths) == 0:
        raise ValueError("Датасет порожній або шлях до нього неправильний.")

    print(f"Total images: {len(image_paths)}")
    print(f"Total classes: {len(set(labels))}")

    class_df = save_class_distribution(labels)
    crop_df = save_crop_distribution(labels)
    health_df = save_health_status_distribution(labels)
    train_test_df = save_train_test_distribution(image_paths, labels)

    print("\nClass distribution:")
    print(class_df.to_string(index=False))

    print("\nCrop distribution:")
    print(crop_df.to_string(index=False))

    print("\nHealth status distribution:")
    print(health_df.to_string(index=False))

    print("\nTrain/test distribution:")
    print(train_test_df.to_string(index=False))

    class_df.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)
    crop_df.to_csv(os.path.join(OUTPUT_DIR, "crop_distribution.csv"), index=False)
    health_df.to_csv(os.path.join(OUTPUT_DIR, "health_status_distribution.csv"), index=False)
    train_test_df.to_csv(os.path.join(OUTPUT_DIR, "train_test_distribution.csv"), index=False)

    print("\nCharts and CSV files saved to the photo folder.")


if __name__ == "__main__":
    main()
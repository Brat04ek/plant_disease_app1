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
MAX_IMAGES_PER_CLASS = 500
TEST_SIZE = 0.2
N_SPLITS = 8

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

def save_kfold_split_bar_chart():
    """
    Побудова bar chart для схеми поділу даних:
    80% train_val поділено на 8 fold-ів по 10%,
    20% відкладено як тестову вибірку.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels = [
        "Fold 1", "Fold 2", "Fold 3", "Fold 4",
        "Fold 5", "Fold 6", "Fold 7", "Fold 8",
        "Test"
    ]

    percentages = [10, 10, 10, 10, 10, 10, 10, 10, 20]

    x = range(len(labels))

    plt.figure(figsize=(13, 6))

    bars = plt.bar(x, percentages)

    plt.title("Поділ набору даних для крос-валідації та фінального тестування")
    plt.xlabel("Частини вибірки")
    plt.ylabel("Частка від використаного набору даних, %")

    plt.xticks(x, labels, rotation=0)
    plt.ylim(0, 25)

    for bar, value in zip(bars, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value}%",
            ha="center",
            va="bottom",
            fontsize=11
        )

    # Пояснювальні підписи під групами
    plt.text(
        3.5,
        -4.2,
        "Train_val = 80%: використовується для StratifiedKFold, k = 8",
        ha="center",
        va="center",
        fontsize=11
    )

    plt.text(
        8,
        -4.2,
        "Test = 20%:\nвідкладена вибірка",
        ha="center",
        va="center",
        fontsize=10
    )

    # Лінія-групування для 8 fold-ів
    plt.plot([-.4, 7.4], [-1.5, -1.5], linewidth=1.5)
    plt.plot([-.4, -.4], [-1.2, -1.8], linewidth=1.5)
    plt.plot([7.4, 7.4], [-1.2, -1.8], linewidth=1.5)

    # Лінія-групування для test
    plt.plot([7.65, 8.35], [-1.5, -1.5], linewidth=1.5)
    plt.plot([7.65, 7.65], [-1.2, -1.8], linewidth=1.5)
    plt.plot([8.35, 8.35], [-1.2, -1.8], linewidth=1.5)

    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "kfold_data_split_bar_chart.jpg"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



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

    # Новий рисунок 3.4.4
    save_kfold_split_bar_chart()

    print("\nClass distribution:")
    print(class_df.to_string(index=False))

    print("\nCrop distribution:")
    print(crop_df.to_string(index=False))

    print("\nHealth status distribution:")
    print(health_df.to_string(index=False))

    class_df.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)
    crop_df.to_csv(os.path.join(OUTPUT_DIR, "crop_distribution.csv"), index=False)
    health_df.to_csv(os.path.join(OUTPUT_DIR, "health_status_distribution.csv"), index=False)

    print("\nCharts and CSV files saved to the photo folder.")


if __name__ == "__main__":
    main()
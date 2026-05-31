import os
import json
import time
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "scripts" else CURRENT_FILE.parent

if str(PROJECT_ROOT) not in sys.path:
    # Цей крок є частиною послідовного алгоритму навчання та оцінювання моделі.
    sys.path.insert(0, str(PROJECT_ROOT))

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

from src.plant_diagnosis.image_processing import (
    extract_orb_descriptors,
    extract_full_feature_vector
)


# =========================
# Global settings
# =========================

DATASET_PATH = str(PROJECT_ROOT / "dataset" / "PlantVillage")
MODELS_PATH = str(PROJECT_ROOT / "models")
RESULTS_PATH = str(PROJECT_ROOT / "results" / "train")

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

    # Список image_paths накопичує повні шляхи до зображень датасету.
    image_paths = []
    # Список labels зберігає назви класів у тому самому порядку, що й image_paths.
    labels = []

    # Обхід класів виконується у відсортованому порядку для відтворюваності результатів.
    for class_name in sorted(os.listdir(dataset_path)):
        # Для кожного класу формується шлях до відповідної папки зображень.
        class_folder = os.path.join(dataset_path, class_name)

        # Перевірка відсікає файли або службові об’єкти, які не є папками класів.
        if not os.path.isdir(class_folder):
            # Якщо умова пропуску виконана, поточний елемент не використовується далі.
            continue

        # Лічильник count контролює обмеження максимальної кількості зображень на клас.
        count = 0

        # У межах кожного класу послідовно переглядаються всі файли зображень.
        for file_name in sorted(os.listdir(class_folder)):
            # Перевірка формату залишає тільки файли з розширеннями JPG, JPEG або PNG.
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                # Якщо умова пропуску виконана, поточний елемент не використовується далі.
                continue

            # Коректний файл додається до списку зображень для подальшого навчання.
            image_paths.append(os.path.join(class_folder, file_name))
            # Для кожного зображення додається відповідна мітка класу.
            labels.append(class_name)

            # Після додавання зображення оновлюється кількість використаних файлів поточного класу.
            count += 1

            # Якщо встановлено обмеження, зайві зображення поточного класу не додаються.
            if max_images_per_class is not None and count >= max_images_per_class:
                # Завершуємо обхід поточного класу після досягнення заданого ліміту.
                break

    # Повертаємо шляхи та мітки як NumPy-масиви для зручного індексування.
    return np.array(image_paths), np.array(labels)


def save_dataset_statistics(labels, output_path: str):
    """
    Збереження інформації про розподіл класів у використаному наборі даних.
    """

    # Counter підраховує кількість зображень для кожного класу.
    class_counts = Counter(labels)

    # На основі підрахунку формується таблиця розподілу класів.
    df = pd.DataFrame({
        "class_name": list(class_counts.keys()),
        "count": list(class_counts.values())
    }).sort_values("class_name")

    # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
    df.to_csv(os.path.join(output_path, "dataset_class_distribution.csv"), index=False)

    # Створюється окрема фігура matplotlib для збереження графіка у файл.
    plt.figure(figsize=(14, 7))
    # Стовпчикова діаграма використовується для наочного порівняння значень.
    plt.bar(df["class_name"], df["count"])
    # Заголовок графіка пояснює, які саме результати відображені.
    plt.title("Dataset class distribution")
    # Підпис осі X уточнює, які об’єкти порівнюються на графіку.
    plt.xlabel("Class")
    # Підпис осі Y уточнює вимірюване значення.
    plt.ylabel("Number of images")
    # Поворот підписів робить довгі назви класів читабельними.
    plt.xticks(rotation=75, ha="right")
    # tight_layout запобігає обрізанню підписів під час збереження графіка.
    plt.tight_layout()
    # Графік зберігається у файл із високою роздільною здатністю для вставки в роботу.
    plt.savefig(os.path.join(output_path, "dataset_class_distribution.jpg"), dpi=300)
    # Закриття фігури звільняє пам’ять під час довгих експериментів.
    plt.close()

    # Повернення DataFrame дозволяє надалі вивести або додатково обробити статистику.
    return df


def save_split_distribution(train_val_labels, test_labels, class_names, output_path: str):
    """
    Збереження таблиці та графіка розподілу класів у train_val та test.
    """

    # Окремо підраховується кількість прикладів у навчально-валідаційній частині.
    train_val_counts = Counter(train_val_labels)
    # Окремо підраховується кількість прикладів у відкладеній тестовій частині.
    test_counts = Counter(test_labels)

    # Список rows накопичує рядки майбутньої таблиці розподілу.
    rows = []

    # Розподіл формується для кожного класу, збереженого в LabelEncoder.
    for class_name in class_names:
        # У таблицю додається кількість зображень класу в train_val і test.
        rows.append({
            "class_name": class_name,
            "train_val_count": train_val_counts[class_name],
            "test_count": test_counts[class_name],
            "total_count": train_val_counts[class_name] + test_counts[class_name]
        })

    # Проміжний результат зберігається у змінній для подальшого використання.
    df = pd.DataFrame(rows)
    # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
    df.to_csv(os.path.join(output_path, "train_val_test_distribution.csv"), index=False)

    # Індекси x використовуються як позиції стовпчиків на графіку.
    x = range(len(df))

    # Створюється окрема фігура matplotlib для збереження графіка у файл.
    plt.figure(figsize=(14, 7))
    # Стовпчикова діаграма використовується для наочного порівняння значень.
    plt.bar([i - 0.2 for i in x], df["train_val_count"], width=0.4, label="Train + validation")
    # Стовпчикова діаграма використовується для наочного порівняння значень.
    plt.bar([i + 0.2 for i in x], df["test_count"], width=0.4, label="Test")

    # Заголовок графіка пояснює, які саме результати відображені.
    plt.title("Train/validation and test distribution by class")
    # Підпис осі X уточнює, які об’єкти порівнюються на графіку.
    plt.xlabel("Class")
    # Підпис осі Y уточнює вимірюване значення.
    plt.ylabel("Number of images")
    # Поворот підписів робить довгі назви класів читабельними.
    plt.xticks(x, df["class_name"], rotation=75, ha="right")
    # Легенда пояснює, які групи даних показані на графіку.
    plt.legend()
    # tight_layout запобігає обрізанню підписів під час збереження графіка.
    plt.tight_layout()
    # Графік зберігається у файл із високою роздільною здатністю для вставки в роботу.
    plt.savefig(os.path.join(output_path, "train_val_test_distribution.jpg"), dpi=300)
    # Закриття фігури звільняє пам’ять під час довгих експериментів.
    plt.close()

    # Повернення DataFrame дозволяє надалі вивести або додатково обробити статистику.
    return df


# =========================
# Feature extraction
# =========================

def collect_orb_descriptors(image_paths):
    """
    Збір ORB-дескрипторів лише з навчальної частини поточного fold.
    Це важливо, щоб не було витоку інформації з validation або test.
    """

    # У цей список збираються ORB-дескриптори з усіх навчальних зображень fold.
    all_descriptors = []

    # Зображення обробляються по черзі, щоб поступово формувати ознаки або дескриптори.
    for idx, image_path in enumerate(image_paths):
        # OpenCV зчитує файл зображення у форматі BGR.
        image = cv2.imread(image_path)

        # Перевірка захищає алгоритм від пошкоджених або недоступних файлів.
        if image is None:
            # Попередження допомагає знайти файл, який не вдалося зчитати.
            print(f"Warning: cannot read image {image_path}")
            # Якщо умова пропуску виконана, поточний елемент не використовується далі.
            continue

        # З кожного зображення виділяються ORB-дескриптори для побудови BoVW-словника.
        _, descriptors, _, _ = extract_orb_descriptors(image)

        # До словника додаються тільки непорожні набори локальних дескрипторів.
        if descriptors is not None and len(descriptors) > 0:
            # Дескриптори поточного зображення додаються до загального масиву навчання KMeans.
            all_descriptors.append(descriptors)

        # Періодичний вивід прогресу корисний під час обробки великої кількості зображень.
        if idx % 100 == 0:
            # У консолі фіксується прогрес збору ORB-дескрипторів.
            print(f"Collected ORB descriptors from {idx}/{len(image_paths)} images")

    # Якщо дескриптори не знайдено, подальше навчання BoVW неможливе.
    if len(all_descriptors) == 0:
        # Помилка явно повідомляє про некоректний датасет або проблему виділення ознак.
        raise ValueError("Не знайдено ORB-дескрипторів. Перевірте датасет або image_processing.py.")

    # vstack об’єднує всі локальні дескриптори в одну матрицю для MiniBatchKMeans.
    return np.vstack(all_descriptors)


def build_feature_matrix(image_paths, labels, kmeans_model):
    """
    Формування матриці гібридних ознак для списку зображень.

    Повертає:
    - X: матриця ознак;
    - valid_labels: мітки тільки для тих зображень, які вдалося зчитати;
    - valid_paths: шляхи тільки для тих зображень, які вдалося зчитати.
    """

    # Список features поступово заповнюється векторами ознак для кожного коректного зображення.
    features = []
    # valid_labels потрібний для синхронізації міток лише з успішно зчитаними зображеннями.
    valid_labels = []
    # valid_paths зберігає шляхи тільки до тих файлів, які реально потрапили в обробку.
    valid_paths = []

    # Зображення і мітки проходять синхронний обхід для формування X та y.
    for idx, (image_path, label) in enumerate(zip(image_paths, labels)):
        # OpenCV зчитує файл зображення у форматі BGR.
        image = cv2.imread(image_path)

        # Перевірка захищає алгоритм від пошкоджених або недоступних файлів.
        if image is None:
            # Попередження допомагає знайти файл, який не вдалося зчитати.
            print(f"Warning: cannot read image {image_path}")
            # Якщо умова пропуску виконана, поточний елемент не використовується далі.
            continue

        # Для кожного зображення формується повний гібридний вектор ознак.
        feature_vector = extract_full_feature_vector(image, kmeans_model)

        # Вектор ознак додається до матриці майбутніх вхідних даних моделі.
        features.append(feature_vector)
        # Мітка додається тільки після успішного формування вектора ознак.
        valid_labels.append(label)
        # Шлях зберігається для можливого аналізу прогнозів і помилок.
        valid_paths.append(image_path)

        # Періодичний вивід прогресу корисний під час обробки великої кількості зображень.
        if idx % 100 == 0:
            # У консолі показується прогрес побудови матриці ознак.
            print(f"Built features for {idx}/{len(image_paths)} images")

    # Функція повертає матрицю ознак разом із валідними мітками та шляхами.
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

    # Random Forest використовується як один із базових деревоподібних класифікаторів.
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
        max_depth=None
    )

    # Extra Trees додає до ансамблю більш випадковий деревоподібний класифікатор.
    extra_trees_model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )

    # SVM розміщується в Pipeline разом зі стандартизацією ознак.
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

    # VotingClassifier об’єднує прогнози трьох базових моделей.
    voting_model = VotingClassifier(
        estimators=[
            ("rf", rf_model),
            ("extra", extra_trees_model),
            ("svm", svm_model)
        ],
        voting="soft",
        n_jobs=1
    )

    # Повертається готова ансамблева модель для навчання або тестування.
    return voting_model


def evaluate_predictions(y_true, y_pred):
    """
    Обчислення основних метрик.
    Основна метрика для вибору параметрів — macro_f1_score.
    """

    # Accuracy використовується як додаткова загальна характеристика якості.
    accuracy = accuracy_score(y_true, y_pred)

    # Macro-усереднення дає однакову вагу кожному класу незалежно від його розміру.
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # Weighted-усереднення враховує кількість прикладів кожного класу.
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Метрики повертаються словником для зручного збереження у таблиці результатів.
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

    # Файл cv_results_all_folds.csv зберігає проміжні результати всіх fold-ів.
    results_file = os.path.join(output_path, "cv_results_all_folds.csv")

    # Якщо результати вже існують, запускається режим продовження експерименту.
    if os.path.exists(results_file):
        # Наявна таблиця завантажується, щоб не перераховувати вже завершені fold-и.
        existing_results_df = pd.read_csv(results_file)
        # Попередні результати переводяться у список словників для доповнення новими fold-ами.
        all_fold_results = existing_results_df.to_dict("records")
        # Повідомлення вказує, що повторний запуск не починається з нуля.
        print(f"\nResume mode enabled.")
        # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
        print(f"Loaded existing CV results: {len(existing_results_df)} rows")
    else:
        # Якщо результатів ще немає, створюється порожня таблиця для перевірки.
        existing_results_df = pd.DataFrame()
        # Новий експеримент починається з порожнього списку результатів.
        all_fold_results = []

    # Вкладена функція перевіряє, чи був конкретний fold уже збережений.
    def fold_already_done(config_name: str, fold_idx: int) -> bool:
        # За відсутності попередніх результатів жоден fold не може бути пропущений.
        if existing_results_df.empty:
            # Повернене значення використовується наступними етапами пайплайна.
            return False

        # Маска знаходить рядок із потрібною конфігурацією та номером fold.
        mask = (
            (existing_results_df["config_name"] == config_name) &
            (existing_results_df["fold"] == fold_idx)
        )

        # True означає, що цей fold уже є у файлі результатів.
        return bool(mask.any())

    # StratifiedKFold зберігає приблизні пропорції класів у кожному fold.
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Кожна конфігурація параметрів перевіряється окремо.
    for config in PARAM_CONFIGS:
        # Назва конфігурації використовується для ідентифікації результатів у CSV.
        config_name = config["config_name"]
        # n_clusters визначає розмір словника візуальних слів BoVW.
        n_clusters = config["n_clusters"]
        # n_estimators задає кількість дерев у Random Forest та Extra Trees.
        n_estimators = config["n_estimators"]
        # svm_C контролює регуляризацію SVM з RBF-ядром.
        svm_C = config["svm_C"]

        # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
        print("\n" + "=" * 80)
        # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
        print(f"Running cross-validation for {config_name}")
        # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
        print(f"n_clusters={n_clusters}, n_estimators={n_estimators}, svm_C={svm_C}")
        # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
        print("=" * 80)

        # Для кожного fold формується окрема навчальна та валідаційна частина.
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(train_val_paths, train_val_labels),
            start=1
        ):
            # У режимі resume вже завершені fold-и не перераховуються повторно.
            if fold_already_done(config_name, fold_idx):
                # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
                print(f"Skipping {config_name} | Fold {fold_idx}/{N_SPLITS} because it already exists.")
                # Якщо умова пропуску виконана, поточний елемент не використовується далі.
                continue

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print("\n" + "-" * 80)
            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print(f"{config_name} | Fold {fold_idx}/{N_SPLITS}")
            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print("-" * 80)

            # Час старту потрібний для оцінки тривалості обробки поточного fold.
            fold_start_time = time.time()

            # Індекси StratifiedKFold відбирають зображення навчальної частини fold.
            fold_train_paths = train_val_paths[train_idx]
            # Окремо формується список зображень для валідаційної частини fold.
            fold_val_paths = train_val_paths[val_idx]

            # Мітки навчальної частини відповідають відібраним train-зображенням.
            fold_train_labels = train_val_labels[train_idx]
            # Мітки validation-частини відповідають відібраним валідаційним зображенням.
            fold_val_labels = train_val_labels[val_idx]

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print(f"Fold train images: {len(fold_train_paths)}")
            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print(f"Fold validation images: {len(fold_val_paths)}")

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print("\nCollecting ORB descriptors for current fold...")
            # BoVW-словник навчається тільки на train-частині поточного fold.
            train_descriptors = collect_orb_descriptors(fold_train_paths)

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print(f"Total descriptors shape: {train_descriptors.shape}")

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print("\nTraining MiniBatchKMeans visual vocabulary...")
            # MiniBatchKMeans будує словник візуальних слів із заданою кількістю кластерів.
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=RANDOM_STATE,
                batch_size=KMEANS_BATCH_SIZE,
                n_init="auto"
            )
            # Навчання KMeans виконується на зібраних ORB-дескрипторах.
            kmeans.fit(train_descriptors)

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print("\nBuilding fold train feature matrix...")
            # Для навчальної частини формується матриця гібридних ознак.
            X_train, valid_train_labels, _ = build_feature_matrix(
                fold_train_paths,
                fold_train_labels,
                kmeans
            )

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print("\nBuilding fold validation feature matrix...")
            # Для validation-частини формується окрема матриця ознак.
            X_val, valid_val_labels, _ = build_feature_matrix(
                fold_val_paths,
                fold_val_labels,
                kmeans
            )

            # Текстові мітки train перетворюються у числовий формат sklearn.
            y_train = label_encoder.transform(valid_train_labels)
            # Текстові мітки validation також кодуються тим самим LabelEncoder.
            y_val = label_encoder.transform(valid_val_labels)

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print(f"\nFeature vector size: {X_train.shape[1]}")

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print("\nTraining VotingClassifier...")
            # Для поточної конфігурації створюється новий VotingClassifier.
            model = create_voting_model(
                n_estimators=n_estimators,
                svm_C=svm_C,
                random_state=RANDOM_STATE
            )

            # Модель навчається тільки на train-частині поточного fold.
            model.fit(X_train, y_train)

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print("\nEvaluating on validation fold...")
            # Після навчання модель прогнозує класи для validation-зображень.
            y_val_pred = model.predict(X_val)

            # Для отриманих прогнозів обчислюються основні метрики якості.
            metrics = evaluate_predictions(y_val, y_val_pred)

            # Тривалість fold використовується для аналізу обчислювальної складності.
            fold_time = time.time() - fold_start_time

            # У result_row збираються параметри, розміри вибірок, час і метрики fold.
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

            # Результат поточного fold додається до загального списку експерименту.
            all_fold_results.append(result_row)

            # Поточний список результатів одразу перетворюється у таблицю.
            current_results_df = pd.DataFrame(all_fold_results)
            # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
            current_results_df.to_csv(results_file, index=False)

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print("\nFold metrics:")
            # Усі метрики поточного fold виводяться у консоль для контролю.
            for key, value in metrics.items():
                # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
                print(f"{key}: {value:.4f}")

            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print(f"Fold time: {fold_time:.2f} seconds")
            # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
            print(f"Intermediate results saved to: {results_file}")

    # Після завершення крос-валідації формується повна таблиця fold-ів.
    cv_results_df = pd.DataFrame(all_fold_results)
    # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
    cv_results_df.to_csv(results_file, index=False)

    # Зведена таблиця агрегує результати за конфігураціями параметрів.
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

    # Після groupby назви багаторівневих колонок спрощуються до одного рівня.
    summary_df.columns = [
        "_".join(col).strip("_") for col in summary_df.columns.values
    ]

    # Конфігурації сортуються за основною метрикою macro F1-score.
    summary_df = summary_df.sort_values("macro_f1_score_mean", ascending=False)

    # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
    summary_df.to_csv(os.path.join(output_path, "cv_results_summary.csv"), index=False)

    # Функція повертає детальні та узагальнені результати крос-валідації.
    return cv_results_df, summary_df


# =========================
# Saving plots and reports
# =========================

def save_cv_macro_f1_plot(summary_df, output_path: str):
    """
    Збереження графіка порівняння 5 конфігурацій за macro F1-score.
    """

    # Копія таблиці захищає вихідний DataFrame від випадкових змін.
    df = summary_df.copy()
    # Проміжний результат зберігається у змінній для подальшого використання.
    df = df.sort_values("macro_f1_score_mean", ascending=False)

    # Підписи графіка містять назву конфігурації та її головні параметри.
    labels = [
        f"{row.config_name}\nK={row.n_clusters}, trees={row.n_estimators}, C={row.svm_C}"
        for row in df.itertuples()
    ]

    # Індекси x використовуються як позиції стовпчиків на графіку.
    x = range(len(df))

    # Створюється окрема фігура matplotlib для збереження графіка у файл.
    plt.figure(figsize=(12, 7))
    # Збереження об’єктів bars потрібне для підписування значень над стовпчиками.
    bars = plt.bar(x, df["macro_f1_score_mean"])

    # Заголовок графіка пояснює, які саме результати відображені.
    plt.title("Cross-validation comparison by macro F1-score")
    # Підпис осі X уточнює, які об’єкти порівнюються на графіку.
    plt.xlabel("Configuration")
    # Підпис осі Y уточнює вимірюване значення.
    plt.ylabel("Mean macro F1-score")

    # Нижня межа осі Y підбирається так, щоб краще показати різницю між моделями.
    min_score = max(0.0, df["macro_f1_score_mean"].min() - 0.03)
    # Верхня межа осі Y обмежується одиницею, оскільки F1-score не може бути більшим за 1.
    max_score = min(1.0, df["macro_f1_score_mean"].max() + 0.03)
    # Межі осі Y роблять порівняння конфігурацій більш наочним.
    plt.ylim(min_score, max_score)

    # Поворот підписів робить довгі назви класів читабельними.
    plt.xticks(x, labels, rotation=20, ha="right")

    # Кожен стовпчик підписується точним числовим значенням метрики.
    for bar, value in zip(bars, df["macro_f1_score_mean"]):
        # Числовий підпис допомагає читати графік без звернення до CSV.
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # tight_layout запобігає обрізанню підписів під час збереження графіка.
    plt.tight_layout()
    # Графік зберігається у файл із високою роздільною здатністю для вставки в роботу.
    plt.savefig(os.path.join(output_path, "cv_macro_f1_comparison.jpg"), dpi=300)
    # Закриття фігури звільняє пам’ять під час довгих експериментів.
    plt.close()


def save_final_confusion_matrix(y_test, y_pred, class_names, output_path: str):
    """
    Збереження фінальної матриці помилок.
    """

    # Матриця помилок показує, між якими класами виникає плутанина.
    cm = confusion_matrix(y_test, y_pred)

    # Матриця переводиться у DataFrame, щоб зберегти назви класів у рядках і стовпцях.
    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )
    # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
    cm_df.to_csv(os.path.join(output_path, "final_confusion_matrix.csv"))

    # Фігура та вісь створюються явно для коректного відображення ConfusionMatrixDisplay.
    fig, ax = plt.subplots(figsize=(12, 10))

    # ConfusionMatrixDisplay будує стандартне графічне представлення матриці помилок.
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    # Матриця відображається з підписами класів і кольоровою шкалою.
    display.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=True
    )

    # Заголовок графіка пояснює, які саме результати відображені.
    plt.title("Final test confusion matrix")
    # tight_layout запобігає обрізанню підписів під час збереження графіка.
    plt.tight_layout()
    # Графік зберігається у файл із високою роздільною здатністю для вставки в роботу.
    plt.savefig(os.path.join(output_path, "final_confusion_matrix.jpg"), dpi=300)
    # Закриття фігури звільняє пам’ять під час довгих експериментів.
    plt.close()


def save_final_classification_report(y_test, y_pred, class_names, output_path: str):
    """
    Збереження classification_report у форматі CSV.
    """

    # classification_report формує precision, recall і F1-score для кожного класу.
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Повний звіт перетворюється у таблицю для збереження.
    report_df = pd.DataFrame(report).transpose()
    # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
    report_df.to_csv(os.path.join(output_path, "final_classification_report.csv"))

    # Окремий список class_rows зберігає тільки покласові метрики.
    class_rows = []

    # Розподіл формується для кожного класу, збереженого в LabelEncoder.
    for class_name in class_names:
        # До таблиці додаються precision, recall, F1-score і support поточного класу.
        class_rows.append({
            "class_name": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1_score": report[class_name]["f1-score"],
            "support": report[class_name]["support"]
        })

    # Покласові метрики перетворюються у DataFrame для графіків і дипломного тексту.
    class_metrics_df = pd.DataFrame(class_rows)
    # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
    class_metrics_df.to_csv(
        os.path.join(output_path, "final_class_metrics_by_class.csv"),
        index=False
    )

    # Індекси x використовуються як позиції стовпчиків на графіку.
    x = range(len(class_metrics_df))

    # Створюється окрема фігура matplotlib для збереження графіка у файл.
    plt.figure(figsize=(14, 7))
    # Стовпчикова діаграма використовується для наочного порівняння значень.
    plt.bar([i - 0.25 for i in x], class_metrics_df["precision"], width=0.25, label="Precision")
    # Стовпчикова діаграма використовується для наочного порівняння значень.
    plt.bar(x, class_metrics_df["recall"], width=0.25, label="Recall")
    # Стовпчикова діаграма використовується для наочного порівняння значень.
    plt.bar([i + 0.25 for i in x], class_metrics_df["f1_score"], width=0.25, label="F1-score")

    # Заголовок графіка пояснює, які саме результати відображені.
    plt.title("Final test classification metrics by class")
    # Підпис осі X уточнює, які об’єкти порівнюються на графіку.
    plt.xlabel("Class")
    # Підпис осі Y уточнює вимірюване значення.
    plt.ylabel("Metric value")
    # Межі осі Y роблять порівняння конфігурацій більш наочним.
    plt.ylim(0, 1.05)
    # Поворот підписів робить довгі назви класів читабельними.
    plt.xticks(x, class_metrics_df["class_name"], rotation=75, ha="right")
    # Легенда пояснює, які групи даних показані на графіку.
    plt.legend()
    # tight_layout запобігає обрізанню підписів під час збереження графіка.
    plt.tight_layout()

    # Графік зберігається у файл із високою роздільною здатністю для вставки в роботу.
    plt.savefig(os.path.join(output_path, "final_classification_metrics_by_class.jpg"), dpi=300)
    # Закриття фігури звільняє пам’ять під час довгих експериментів.
    plt.close()

    # Функція повертає повний звіт і скорочену таблицю покласових метрик.
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

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\n" + "=" * 80)
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("Training final model with best parameters")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("=" * 80)

    # Найкраще значення K береться з результатів крос-валідації.
    n_clusters = int(best_params["n_clusters"])
    # Кількість дерев фінальної моделі також береться з найкращої конфігурації.
    n_estimators = int(best_params["n_estimators"])
    # Параметр C для SVM відновлюється з обраної конфігурації.
    svm_C = float(best_params["svm_C"])

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Best n_clusters: {n_clusters}")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Best n_estimators: {n_estimators}")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Best svm_C: {svm_C}")

    # Час старту потрібний для вимірювання повного фінального навчання і тестування.
    final_start_time = time.time()

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nCollecting ORB descriptors for final BoVW vocabulary...")
    # Фінальний BoVW-словник будується на всій train_val частині.
    train_val_descriptors = collect_orb_descriptors(train_val_paths)

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Total descriptors shape: {train_val_descriptors.shape}")

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nTraining final MiniBatchKMeans visual vocabulary...")
    # Фінальний MiniBatchKMeans використовує найкращу кількість кластерів.
    final_kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        batch_size=KMEANS_BATCH_SIZE,
        n_init="auto"
    )

    # Словник візуальних слів остаточно навчається на train_val-дескрипторах.
    final_kmeans.fit(train_val_descriptors)

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nBuilding final train_val feature matrix...")
    # Для всіх train_val-зображень формується фінальна матриця ознак.
    X_train_val, valid_train_val_labels, valid_train_val_paths = build_feature_matrix(
        train_val_paths,
        train_val_labels,
        final_kmeans
    )

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nBuilding final test feature matrix...")
    # Для відкладеної test-вибірки формується незалежна матриця ознак.
    X_test, valid_test_labels, valid_test_paths = build_feature_matrix(
        test_paths,
        test_labels,
        final_kmeans
    )

    # Мітки train_val кодуються для навчання фінального класифікатора.
    y_train_val = label_encoder.transform(valid_train_val_labels)
    # Мітки test кодуються для фінального оцінювання моделі.
    y_test = label_encoder.transform(valid_test_labels)

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"\nFinal feature vector size: {X_train_val.shape[1]}")

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nTraining final VotingClassifier...")
    # Фінальний ансамбль створюється з параметрами найкращої конфігурації.
    final_model = create_voting_model(
        n_estimators=n_estimators,
        svm_C=svm_C,
        random_state=RANDOM_STATE
    )

    # Фінальна модель навчається на всій навчально-валідаційній частині.
    final_model.fit(X_train_val, y_train_val)

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nEvaluating final model on untouched test set...")
    # Після навчання модель прогнозує класи для недоторканої test-вибірки.
    y_test_pred = final_model.predict(X_test)

    # Ймовірності потрібні для аналізу впевненості кожного тестового прогнозу.
    if hasattr(final_model, "predict_proba"):
        # predict_proba повертає розподіл імовірностей за всіма класами.
        test_probabilities = final_model.predict_proba(X_test)
        # Максимальна ймовірність інтерпретується як впевненість прогнозу.
        test_confidence = np.max(test_probabilities, axis=1)
    else:
        # Якщо ймовірності недоступні, впевненість заповнюється NaN.
        test_confidence = np.full(shape=len(y_test_pred), fill_value=np.nan)

    # Фінальні метрики обчислюються тільки на відкладеній тестовій вибірці.
    final_metrics = evaluate_predictions(y_test, y_test_pred)

    # Проміжний результат зберігається у змінній для подальшого використання.
    final_time = time.time() - final_start_time

    # У final_metrics_row збираються параметри моделі, розміри вибірок і метрики.
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

    # Підсумкові метрики фінального тестування зберігаються одним рядком.
    final_metrics_df = pd.DataFrame([final_metrics_row])
    # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
    final_metrics_df.to_csv(os.path.join(output_path, "final_test_metrics_summary.csv"), index=False)

    # Список prediction_rows потрібний для збереження кожного тестового прогнозу.
    prediction_rows = []

    # Цикл формує детальну таблицю прогнозів для всіх тестових зображень.
    for image_path, true_label, true_idx, pred_idx, confidence in zip(
        valid_test_paths,
        valid_test_labels,
        y_test,
        y_test_pred,
        test_confidence
    ):
        # Для кожного зображення зберігаються справжній клас, прогноз і впевненість.
        prediction_rows.append({
            "image_path": image_path,
            "true_class": true_label,
            "predicted_class": label_encoder.inverse_transform([pred_idx])[0],
            "true_label_encoded": int(true_idx),
            "predicted_label_encoded": int(pred_idx),
            "confidence": float(confidence),
            "is_correct": bool(true_idx == pred_idx)
        })

    # Детальні прогнози перетворюються у таблицю для аналізу помилок.
    predictions_df = pd.DataFrame(prediction_rows)
    # CSV-файл зберігає числові результати, щоб їх можна було використати у звіті та графіках.
    predictions_df.to_csv(os.path.join(output_path, "final_test_predictions.csv"), index=False)

    # Назви класів беруться з LabelEncoder у тому самому порядку, що й числові мітки.
    class_names = label_encoder.classes_

    # Після фінального тестування зберігається матриця помилок.
    save_final_confusion_matrix(
        y_test,
        y_test_pred,
        class_names,
        output_path
    )

    # Також зберігається classification report з покласовими метриками.
    save_final_classification_report(
        y_test,
        y_test_pred,
        class_names,
        output_path
    )

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nSaving final models...")
    # Каталог models створюється перед збереженням фінальних компонентів.
    os.makedirs(MODELS_PATH, exist_ok=True)

    # Навчений BoVW-словник зберігається для подальшого використання в app.py.
    joblib.dump(final_kmeans, os.path.join(MODELS_PATH, "hybrid_bovw_kmeans.joblib"))
    # Фінальний VotingClassifier зберігається як основна модель класифікації.
    joblib.dump(final_model, os.path.join(MODELS_PATH, "hybrid_classifier.joblib"))
    # LabelEncoder зберігається для відновлення назв класів під час прогнозування.
    joblib.dump(label_encoder, os.path.join(MODELS_PATH, "hybrid_label_encoder.joblib"))

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nFinal test metrics:")
    # Цикл послідовно обробляє елементи поточної структури даних.
    for key, value in final_metrics.items():
        # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
        print(f"{key}: {value:.4f}")

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Final training and test time: {final_time:.2f} seconds")

    # Повернення фінальних об’єктів дозволяє за потреби використати їх без повторного читання з диска.
    return final_model, final_kmeans, final_metrics_df


# =========================
# Main
# =========================

def main():
    # Каталог models створюється перед збереженням фінальних компонентів.
    os.makedirs(MODELS_PATH, exist_ok=True)
    # Перед запуском створюється папка для таблиць і графіків результатів.
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Фіксується час початку всього сценарію навчання.
    full_start_time = time.time()

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("=" * 80)
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("Plant disease classification training with StratifiedKFold validation")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("=" * 80)

    # Конфігурація запуску зберігає всі ключові параметри експерименту.
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

    # run_config.json дозволяє відтворити умови проведеного експерименту.
    with open(os.path.join(RESULTS_PATH, "run_config.json"), "w", encoding="utf-8") as f:
        # JSON зберігається у читабельному форматі з відступами.
        json.dump(run_config, f, ensure_ascii=False, indent=4)

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nLoading dataset...")
    # Завантажуються шляхи до зображень і відповідні назви класів.
    image_paths, labels = load_dataset(
        DATASET_PATH,
        max_images_per_class=MAX_IMAGES_PER_CLASS
    )

    # Порожній датасет зупиняє виконання, щоб не отримати некоректні результати.
    if len(image_paths) == 0:
        # Помилка явно повідомляє про некоректний датасет або проблему виділення ознак.
        raise ValueError("Датасет порожній або шлях неправильний.")

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Images found: {len(image_paths)}")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Classes found: {len(set(labels))}")

    # Список класів використовується для виводу та контролю складу датасету.
    class_names = sorted(set(labels))

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nClasses:")
    # Розподіл формується для кожного класу, збереженого в LabelEncoder.
    for class_name in class_names:
        # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
        print(" -", class_name)

    # Перед навчанням зберігається статистика розподілу класів.
    save_dataset_statistics(labels, RESULTS_PATH)

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nEncoding labels...")
    # LabelEncoder перетворює текстові назви класів у числові мітки.
    label_encoder = LabelEncoder()
    # Кодувальник навчається на всіх доступних назвах класів.
    label_encoder.fit(labels)

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nCreating untouched test split...")
    # Стратифікований train/test split відкладає 20% даних для фінального тестування.
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Train + validation images: {len(train_val_paths)}")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Test images: {len(test_paths)}")

    # Після поділу зберігається таблиця розподілу класів у train_val та test.
    save_split_distribution(
        train_val_labels=train_val_labels,
        test_labels=test_labels,
        class_names=label_encoder.classes_,
        output_path=RESULTS_PATH
    )

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nRunning cross-validation on train_val set...")
    # Крос-валідація повертає детальні fold-результати та зведену таблицю.
    cv_results_df, cv_summary_df = run_cross_validation(
        train_val_paths=train_val_paths,
        train_val_labels=train_val_labels,
        label_encoder=label_encoder,
        output_path=RESULTS_PATH
    )

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nCross-validation summary:")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(cv_summary_df.to_string(index=False))

    # Додатковий графік показує порівняння конфігурацій за головною метрикою.
    save_cv_macro_f1_plot(cv_summary_df, RESULTS_PATH)

    # Перший рядок після сортування відповідає найкращій конфігурації.
    best_row = cv_summary_df.iloc[0]

    # Параметри найкращої конфігурації формуються окремим словником.
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

    # best_params.json зберігає параметри, використані у фінальному навчанні.
    with open(os.path.join(RESULTS_PATH, "best_params.json"), "w", encoding="utf-8") as f:
        # Найкращі параметри записуються для прозорості та відтворюваності.
        json.dump(best_params, f, ensure_ascii=False, indent=4)

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\nBest parameters:")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(json.dumps(best_params, ensure_ascii=False, indent=4))

    # Після вибору параметрів запускається фінальне навчання і тестування.
    train_final_model_and_evaluate(
        train_val_paths=train_val_paths,
        train_val_labels=train_val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        label_encoder=label_encoder,
        best_params=best_params,
        output_path=RESULTS_PATH
    )

    # Загальна тривалість виконання обчислюється після завершення всіх етапів.
    full_time = time.time() - full_start_time

    # Файл run_time.txt зберігає сумарний час запуску.
    with open(os.path.join(RESULTS_PATH, "run_time.txt"), "w", encoding="utf-8") as f:
        # У файл записується час виконання у секундах і хвилинах.
        f.write(f"Total run time seconds: {full_time:.2f}\n")
        # У файл записується час виконання у секундах і хвилинах.
        f.write(f"Total run time minutes: {full_time / 60:.2f}\n")

    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("\n" + "=" * 80)
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("Training, validation and final testing completed successfully.")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Total run time: {full_time / 60:.2f} minutes")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Results saved to: {RESULTS_PATH}")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print(f"Models saved to: {MODELS_PATH}")
    # Повідомлення в консолі допомагає відстежувати виконання довгого процесу.
    print("=" * 80)


if __name__ == "__main__":
    # Запуск main виконується тільки при прямому запуску цього файлу.
    main()

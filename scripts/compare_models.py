# Файл порівнює базові класифікатори у найкращій знайденій конфігурації.
# Він не підбирає параметри заново, а перевіряє якість окремих моделей на test-вибірці.
# Результати зберігаються у вигляді CSV-таблиць, матриць помилок і графіків.
# Коментарі нижче пояснюють ключові етапи алгоритму без зміни функціоналу коду.
# Модуль os використовується для перевірки папок і формування шляхів до файлів.
import os
# Модуль time потрібний для вимірювання часу навчання та прогнозування.
import time
# OpenCV використовується для зчитування зображень із диска.
import cv2

# sys і pathlib допомагають коректно підключити корінь проєкту до sys.path.
import sys
# Path робить роботу зі шляхами незалежнішою від операційної системи.
from pathlib import Path

# Визначаємо абсолютний шлях до поточного файлу.
CURRENT_FILE = Path(__file__).resolve()
# Якщо файл лежить у scripts, коренем проєкту є батьківська папка вище.
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "scripts" else CURRENT_FILE.parent

# Додаємо корінь проєкту до sys.path, щоб імпорти з src працювали при різних способах запуску.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# joblib потрібний для завантаження збереженого KMeans-словника.
import joblib
# NumPy використовується для формування масивів ознак і міток.
import numpy as np
# pandas використовується для збереження таблиць результатів у CSV.
import pandas as pd

# Matplotlib відповідає за побудову графіків порівняння моделей.
import matplotlib
# Backend Agg дозволяє зберігати графіки без відкриття віконного інтерфейсу.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Імпортуємо деревоподібні моделі та ансамбль soft voting.
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
# SVC з RBF-ядром використовується як третій базовий класифікатор.
from sklearn.svm import SVC
# Pipeline потрібний для поєднання масштабування ознак та SVM.
from sklearn.pipeline import Pipeline
# LabelEncoder кодує назви класів, а StandardScaler масштабує ознаки перед SVM.
from sklearn.preprocessing import LabelEncoder, StandardScaler
# train_test_split відтворює той самий stratified 80/20 поділ, що й у навчанні.
from sklearn.model_selection import train_test_split
# Метрики sklearn використовуються для оцінювання моделей та побудови звітів.
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Функція формує той самий гібридний вектор ознак, що використовувався при навчанні.
from src.plant_diagnosis.image_processing import extract_full_feature_vector


# Шлях до набору даних PlantVillage у структурі проєкту.
DATASET_PATH = str(PROJECT_ROOT / "dataset" / "PlantVillage")
# У models зберігаються навчені компоненти, зокрема BoVW-словник.
MODELS_PATH = str(PROJECT_ROOT / "models")
# Загальний каталог результатів експериментів.
RESULTS_PATH = str(PROJECT_ROOT / "results")
# Окрема папка для результатів порівняння базових моделей.
OUTPUT_PATH = str(PROJECT_ROOT / "results" / "model_comparison")

# Обмеження кількості зображень на клас відповідає основному експерименту.
MAX_IMAGES_PER_CLASS = 500
# 20% даних відкладається для фінального тестування.
TEST_SIZE = 0.2
# Фіксований random_state забезпечує відтворюваність поділу та моделей.
RANDOM_STATE = 7

# Найкраща кількість візуальних слів, знайдена за крос-валідацією.
BEST_N_CLUSTERS = 150
# Кількість дерев для Random Forest та Extra Trees у фінальній конфігурації.
BEST_N_ESTIMATORS = 500
# Значення параметра C для SVM RBF у найкращій конфігурації.
BEST_SVM_C = 10


# Функція збирає список файлів і відповідних класів із папкової структури датасету.
def load_dataset(dataset_path: str, max_images_per_class: int | None = None):
    image_paths = []
    # У цей список додаються повні шляхи до зображень.
    labels = []
    # У цей список додаються назви класів, отримані з назв папок.

    for class_name in sorted(os.listdir(dataset_path)):
        # Кожна папка верхнього рівня вважається окремим класом.
        class_folder = os.path.join(dataset_path, class_name)
        # Формуємо шлях до папки конкретного класу.

        if not os.path.isdir(class_folder):
            # Пропускаємо файли або службові обʼєкти, які не є папками класів.
            continue

        count = 0
        # Лічильник потрібний для обмеження кількості зображень на клас.

        for file_name in sorted(os.listdir(class_folder)):
            # Усередині класу перебираємо файли у сталому відсортованому порядку.
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                # До датасету включаються лише підтримувані формати зображень.
                continue

            image_paths.append(os.path.join(class_folder, file_name))
            # Зберігаємо шлях до зображення для подальшого зчитування OpenCV.
            labels.append(class_name)
            # Міткою зображення є назва папки, у якій воно розташоване.

            count += 1
            # Оновлюємо кількість використаних зображень поточного класу.

            if max_images_per_class is not None and count >= max_images_per_class:
                # Якщо досягнуто ліміту, решта зображень цього класу не береться.
                break

    # Повертаємо NumPy-масиви, щоб їх було зручно індексувати після train_test_split.
    return np.array(image_paths), np.array(labels)


# Функція перетворює список зображень на матрицю гібридних ознак.
def build_feature_matrix(image_paths, labels, kmeans_model):
    features = []
    # Тут накопичуються вектори ознак для коректно зчитаних зображень.
    valid_labels = []
    # Мітки зберігаються тільки для зображень, які вдалося обробити.
    valid_paths = []
    # Шляхи valid_paths допомагають відстежити, які саме файли увійшли до матриці.

    for idx, (image_path, label) in enumerate(zip(image_paths, labels)):
        # Обробляємо зображення разом із його реальною міткою класу.
        image = cv2.imread(image_path)
        # OpenCV зчитує зображення у форматі BGR.

        if image is None:
            # Якщо файл пошкоджений або недоступний, він не включається до навчальної матриці.
            print(f"Warning: cannot read image {image_path}")
            continue

        feature_vector = extract_full_feature_vector(image, kmeans_model)
        # Формується повний вектор: ORB/BoVW, HSV, статистики кольору, LBP та ознаки ураження.

        features.append(feature_vector)
        # Додаємо ознаки поточного зображення до загального списку.
        valid_labels.append(label)
        # Зберігаємо відповідну мітку, щоб X та y мали однакову довжину.
        valid_paths.append(image_path)
        # Зберігаємо шлях для можливого аналізу помилок.

        if idx % 100 == 0:
            # Періодичний вивід показує прогрес під час довгої обробки датасету.
            print(f"Built features for {idx}/{len(image_paths)} images")

    # Повертаємо матрицю ознак і валідні метадані після фільтрації незчитаних файлів.
    return (
        np.array(features, dtype=np.float32),
        np.array(valid_labels),
        np.array(valid_paths)
    )


# Функція створює всі моделі, які порівнюються в однакових умовах.
def create_models():
    rf_model = RandomForestClassifier(
        # Random Forest використовується як базова ансамблева модель дерев рішень.
        n_estimators=BEST_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        # class_weight="balanced" частково компенсує нерівномірність кількості прикладів у класах.
        n_jobs=-1,
        max_depth=None
    )

    extra_trees_model = ExtraTreesClassifier(
        # Extra Trees додає більшу випадковість під час побудови дерев.
        n_estimators=BEST_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        # Для Extra Trees також враховується дисбаланс класів.
        n_jobs=-1
    )

    svm_model = Pipeline([
        # Для SVM створюється Pipeline, тому масштабування виконується всередині моделі.
        ("scaler", StandardScaler()),
        # StandardScaler потрібний, бо SVM чутлива до масштабу ознак.
        ("svc", SVC(
            # SVC з RBF-ядром дозволяє будувати нелінійні межі між класами.
            C=BEST_SVM_C,
            kernel="rbf",
            gamma="scale",
            probability=True,
            # probability=True потрібний для soft voting і порівняння ймовірностей.
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    voting_model = VotingClassifier(
        # VotingClassifier обʼєднує прогнози трьох різних класифікаторів.
        estimators=[
            # Кожна модель отримує власну назву, за якою ансамбль звертається до неї.
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
        # Soft voting усереднює ймовірності класів, а не лише голоси за мітку.
        n_jobs=1
        # Для VotingClassifier залишено n_jobs=1, щоб уникнути конфліктів паралельності з вкладеними моделями.
    )

    # Повертаємо список пар: назва моделі та обʼєкт моделі для циклу оцінювання.
    return [
        ("Random Forest", rf_model),
        ("Extra Trees", extra_trees_model),
        ("SVM RBF", svm_model),
        ("Voting Classifier", voting_model)
    ]


# Функція обчислює однаковий набір метрик для кожної моделі.
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    # Accuracy показує загальну частку правильних прогнозів.

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        # Macro-усереднення дає однакову вагу кожному класу незалежно від його розміру.
        y_true,
        y_pred,
        average="macro",
        # Саме macro F1-score використовується як головна метрика порівняння.
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        # Weighted-усереднення додатково враховує кількість прикладів у кожному класі.
        y_true,
        y_pred,
        average="weighted",
        # Ця метрика корисна для загальної оцінки на незбалансованій вибірці.
        zero_division=0
    )

    # Повертаємо метрики у словнику, щоб легко додати їх до рядка результатів.
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1_score": weighted_f1
    }


# Функція зберігає матрицю помилок для окремої моделі.
def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    safe_name = model_name.lower().replace(" ", "_")
    # safe_name використовується у назвах файлів без пробілів.

    cm = confusion_matrix(y_true, y_pred)
    # Матриця помилок показує, які класи модель плутає між собою.

    cm_df = pd.DataFrame(
        # DataFrame додає підписи класів до рядків і стовпців матриці.
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.to_csv(os.path.join(OUTPUT_PATH, f"{safe_name}_confusion_matrix.csv"))
    # CSV-версія потрібна для подальшого аналізу або вставки в роботу.

    fig, ax = plt.subplots(figsize=(12, 10))
    # Створюємо окрему область побудови для збереження графічної матриці.

    display = ConfusionMatrixDisplay(
        # ConfusionMatrixDisplay формує стандартну візуалізацію sklearn.
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        # Підписи класів повертаються вертикально, щоб довгі назви не перекривались.
        ax=ax,
        xticks_rotation=90,
        colorbar=True
    )

    plt.title(f"Confusion matrix — {model_name}")
    plt.tight_layout()
    # tight_layout зменшує ризик обрізання підписів класів.

    plt.savefig(
        # Зберігаємо матрицю як зображення для дипломної роботи.
        os.path.join(OUTPUT_PATH, f"{safe_name}_confusion_matrix.jpg"),
        dpi=300
    )

    plt.close()
    # Закриваємо фігуру, щоб не накопичувати обʼєкти matplotlib у памʼяті.


# Функція зберігає повний classification report для окремої моделі.
def save_classification_report(y_true, y_pred, class_names, model_name):
    safe_name = model_name.lower().replace(" ", "_")
    # Назва моделі нормалізується для безпечного імені CSV-файлу.

    report = classification_report(
        # classification_report містить precision, recall, F1-score та support для класів.
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        # output_dict=True дозволяє перетворити звіт на pandas DataFrame.
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    # transpose робить метрики стовпцями, а класи рядками.

    report_df.to_csv(
        # Зберігаємо звіт для перевірки покласових результатів кожної моделі.
        os.path.join(OUTPUT_PATH, f"{safe_name}_classification_report.csv")
    )


# Функція будує підсумкові графіки порівняння моделей.
def save_model_comparison_plots(results_df):
    df = results_df.sort_values("macro_f1_score", ascending=False).reset_index(drop=True)
    # Моделі сортуються за macro F1-score, бо це головний критерій якості.

    df.to_csv(os.path.join(OUTPUT_PATH, "model_comparison_results.csv"), index=False)
    # Загальна таблиця результатів зберігається окремо для дипломних графіків.

    x = range(len(df))
    # Індекси x використовуються для позицій стовпчиків на графіках.

    # Graph 1: Macro F1-score only
    # Перший графік спеціально фокусується на основній метриці macro F1-score.
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, df["macro_f1_score"])
    # Кожен стовпчик відповідає одній моделі з порівняння.

    plt.title("Порівняння моделей за macro F1-score")
    plt.xlabel("Модель")
    plt.ylabel("Macro F1-score")

    min_value = max(0.0, df["macro_f1_score"].min() - 0.03)
    # Нижня межа осі Y підбирається так, щоб різниця між моделями була помітною.
    max_value = min(1.0, df["macro_f1_score"].max() + 0.03)
    plt.ylim(min_value, max_value)

    plt.xticks(x, df["model"], rotation=15, ha="right")

    for bar, value in zip(bars, df["macro_f1_score"]):
        # Над кожним стовпчиком підписується точне значення метрики.
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
    # Графік зберігається з високою роздільною здатністю для вставки в роботу.
    plt.close()

    # Graph 2: Accuracy, macro recall, macro F1
    # Другий графік порівнює одразу кілька основних метрик.
    plt.figure(figsize=(11, 6))

    plt.bar([i - 0.25 for i in x], df["accuracy"], width=0.25, label="Accuracy")
    # Accuracy показує загальну частку правильних відповідей.
    plt.bar(x, df["macro_recall"], width=0.25, label="Macro recall")
    # Macro recall показує здатність знаходити обʼєкти кожного класу.
    plt.bar([i + 0.25 for i in x], df["macro_f1_score"], width=0.25, label="Macro F1-score")
    # Macro F1-score поєднує precision і recall для всіх класів.

    plt.title("Порівняння моделей за основними метриками")
    plt.xlabel("Модель")
    plt.ylabel("Значення метрики")

    metric_min = min(
        # Масштаб осі Y визначається за найменшим значенням серед показаних метрик.
        df["accuracy"].min(),
        df["macro_recall"].min(),
        df["macro_f1_score"].min()
    )
    plt.ylim(max(0.0, metric_min - 0.03), 1.0)

    plt.xticks(x, df["model"], rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_PATH, "model_comparison_main_metrics.jpg"), dpi=300)
    # Цей графік використовується для порівняння базових моделей у дипломі.
    plt.close()

    # Graph 3: Training and prediction time
    # Третій графік показує обчислювальну складність моделей за часом навчання.
    plt.figure(figsize=(10, 6))

    plt.bar(x, df["training_time_seconds"])
    # Висота стовпчика відповідає часу навчання конкретної моделі.

    plt.title("Порівняння часу навчання моделей")
    plt.xlabel("Модель")
    plt.ylabel("Час навчання, с")
    plt.xticks(x, df["model"], rotation=15, ha="right")

    for i, value in enumerate(df["training_time_seconds"]):
        # Підписуємо точні значення часу, щоб графік був зрозумілим без таблиці.
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
    # Графік часу допомагає оцінити компроміс між якістю і швидкістю.
    plt.close()


# Головна функція запускає повний сценарій порівняння моделей.
def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # Створюємо папку результатів, якщо її ще немає.

    print("=" * 80)
    # Службовий вивід відокремлює запуск експерименту в консолі.
    print("Model comparison with best validated parameters")
    print("=" * 80)

    print(f"K = {BEST_N_CLUSTERS}")
    # У консолі фіксуються параметри, з якими виконується порівняння.
    print(f"trees = {BEST_N_ESTIMATORS}")
    print(f"C = {BEST_SVM_C}")

    print("\nLoading dataset...")
    image_paths, labels = load_dataset(
        # Використовується той самий ліміт зображень на клас, що й у фінальному навчанні.
        DATASET_PATH,
        max_images_per_class=MAX_IMAGES_PER_CLASS
    )

    print(f"Images found: {len(image_paths)}")
    print(f"Classes found: {len(set(labels))}")

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        # Відтворюємо стратифікований поділ 80/20 для чесного порівняння на test.
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
        # stratify зберігає пропорції класів у train_val і test.
    )

    print(f"Train + validation images: {len(train_val_paths)}")
    print(f"Test images: {len(test_paths)}")

    print("\nLoading final KMeans visual vocabulary...")
    kmeans = joblib.load(os.path.join(MODELS_PATH, "hybrid_bovw_kmeans.joblib"))
    # Завантажений KMeans перетворює ORB-дескриптори у BoVW-гістограму.

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
    # LabelEncoder перетворює текстові назви класів у числові мітки.
    label_encoder.fit(labels)

    y_train_val = label_encoder.transform(valid_train_val_labels)
    # Кодуємо мітки train_val після відкидання незчитаних зображень.
    y_test = label_encoder.transform(valid_test_labels)
    # Кодуємо test-мітки у тому самому просторі класів.

    class_names = label_encoder.classes_
    # Назви класів потрібні для матриць помилок і classification report.

    models = create_models()
    # Отримуємо список моделей, які будуть навчатися та оцінюватися.

    results = []
    # У цьому списку накопичуються метрики всіх моделей.

    for model_name, model in models:
        # Кожна модель проходить однаковий цикл: навчання, прогнозування, оцінювання.
        print("\n" + "-" * 80)
        print(f"Training and evaluating: {model_name}")
        print("-" * 80)

        start_train = time.time()
        # Починаємо вимірювати час навчання поточної моделі.
        model.fit(X_train_val, y_train_val)
        # Навчання виконується на всій train_val частині з однаковими ознаками.
        training_time = time.time() - start_train
        # Фіксуємо тривалість навчання для подальшого порівняння.

        start_pred = time.time()
        # Окремо вимірюємо час прогнозування на test-вибірці.
        y_pred = model.predict(X_test)
        # Прогноз виконується на відкладених тестових даних.
        prediction_time = time.time() - start_pred
        # Час прогнозування показує швидкість застосування моделі.

        metrics = calculate_metrics(y_test, y_pred)
        # Для кожної моделі обчислюється однаковий набір метрик.

        row = {
            # Формуємо один рядок підсумкової таблиці результатів.
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
        # Додаємо результати поточної моделі до загального списку.

        save_confusion_matrix(
            # Матриця помилок зберігається окремо для кожного класифікатора.
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            model_name=model_name
        )

        save_classification_report(
            # Classification report дозволяє переглянути покласові метрики.
            y_true=y_test,
            y_pred=y_pred,
            class_names=class_names,
            model_name=model_name
        )

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        # У консоль виводяться ключові метрики для швидкої перевірки запуску.
        print(f"Macro F1-score: {metrics['macro_f1_score']:.4f}")
        print(f"Training time: {training_time:.2f} s")
        print(f"Prediction time: {prediction_time:.2f} s")

    results_df = pd.DataFrame(results)
    # Після циклу всі результати перетворюються у таблицю pandas.
    results_df = results_df.sort_values("macro_f1_score", ascending=False)
    # Підсумкова таблиця сортується за головною метрикою macro F1-score.

    print("\nModel comparison results:")
    print(results_df.to_string(index=False))

    save_model_comparison_plots(results_df)
    # На основі таблиці будуються графіки для розділу порівняння моделей.

    print("\nSaved results to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    # Запуск main відбувається тільки при прямому виконанні цього файлу.
    main()

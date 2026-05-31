# Модуль os використовується для створення каталогів і перевірки наявності файлів.
import os
# sys потрібний для додавання кореня проєкту до шляху імпорту.
import sys
# Path забезпечує зручну та незалежну від ОС роботу зі шляхами.
from pathlib import Path

# OpenCV використовується для зчитування зображень під час ROC-аналізу.
import cv2
# joblib завантажує збережені компоненти моделі: KMeans, класифікатор і LabelEncoder.
import joblib
# NumPy використовується для роботи з масивами ознак, міток і ROC-координат.
import numpy as np
# pandas потрібний для зчитування CSV-результатів і формування таблиць AUC.
import pandas as pd

# matplotlib використовується для побудови всіх графіків валідації.
import matplotlib
# Backend Agg дозволяє зберігати графіки у файл без відкриття графічного вікна.
matplotlib.use("Agg")
# pyplot надає функції побудови стовпчикових графіків, boxplot, ROC-кривих і heatmap.
import matplotlib.pyplot as plt

# train_test_split відтворює той самий test-поділ, що й під час навчання.
from sklearn.model_selection import train_test_split
# label_binarize перетворює багатокласові мітки у формат one-vs-rest для ROC.
from sklearn.preprocessing import label_binarize
# roc_curve та auc використовуються для обчислення ROC-кривих і площ під ними.
from sklearn.metrics import roc_curve, auc

# Дозволяє запускати файл і з кореня проєкту, і як scripts/plot_results.py
# Визначаємо поточний файл, щоб коректно знайти корінь проєкту.
CURRENT_FILE = Path(__file__).resolve()
# Якщо файл лежить у scripts, коренем проєкту вважається батьківська папка.
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "scripts" else CURRENT_FILE.parent

# Додаємо корінь проєкту до sys.path для коректних імпортів із src.
if str(PROJECT_ROOT) not in sys.path:
# Вставляємо шлях на початок, щоб використовувались локальні модулі проєкту.
    sys.path.insert(0, str(PROJECT_ROOT))

# Імпортуємо функцію, яка формує той самий гібридний вектор ознак, що й при навчанні.
from src.plant_diagnosis.image_processing import extract_full_feature_vector


# Шлях до набору даних потрібний для повторного формування тестових ознак у ROC-аналізі.
DATASET_PATH = PROJECT_ROOT / "dataset" / "PlantVillage"
# У цій папці зберігаються навчені компоненти фінальної моделі.
MODELS_PATH = PROJECT_ROOT / "models"

# Загальна папка для всіх результатів експериментів.
RESULTS_PATH = PROJECT_ROOT / "results"
# CSV-файли навчання та фінального тестування зберігаються у results/train.
TRAIN_RESULTS_PATH = RESULTS_PATH / "train"
# У цю папку записуються графіки для вставки в дипломну роботу.
VALIDATION_OUTPUT_PATH = RESULTS_PATH / "validation_plots"

# Обмеження кількості зображень на клас має збігатися з налаштуваннями train_model.py.
MAX_IMAGES_PER_CLASS = 500
# Частка test-вибірки має бути такою самою, як під час фінального навчання.
TEST_SIZE = 0.2
# Фіксований random_state забезпечує відтворюваність train/test-поділу.
RANDOM_STATE = 7

# Для сумісності з уже написаними функціями
# Рядковий шлях залишено для сумісності з функціями matplotlib та os.path.
OUTPUT_PATH = str(VALIDATION_OUTPUT_PATH)

def ensure_output_dir():
    # Створюємо папку для графіків, якщо вона ще не існує.
    # exist_ok=True дозволяє повторно запускати скрипт без помилки.
    os.makedirs(VALIDATION_OUTPUT_PATH, exist_ok=True)
    


def load_required_csv(file_name: str):
    """
    Завантажує CSV-файли, які створює train_model.py.
    У новій структурі вони знаходяться в results/train.
    """
    # Формуємо повний шлях до потрібного CSV-файлу у папці train.
    file_path = TRAIN_RESULTS_PATH / file_name

    # Перед зчитуванням перевіряємо, чи файл був створений train_model.py.
    if not file_path.exists():
        # Якщо файл відсутній, одразу повідомляємо точний шлях до проблемного файлу.
        raise FileNotFoundError(f"File not found: {file_path}")

    # Повертаємо DataFrame, який далі використовується для побудови графіків.
    return pd.read_csv(file_path)


def add_config_label(df: pd.DataFrame):
    # Працюємо з копією, щоб не змінювати початкову таблицю результатів.
    df = df.copy()

    # Додаємо текстову мітку конфігурації для підписів на графіках.
    df["config_label"] = df.apply(
        # Мітка збирається з назви конфігурації та ключових параметрів моделі.
        lambda row: (
            f"{row['config_name']}\n"
            f"K={int(row['n_clusters'])}, "
            f"trees={int(row['n_estimators'])}, "
            f"C={row['svm_C']}"
        ),
        # axis=1 означає, що функція застосовується до кожного рядка таблиці.
        axis=1
    )

    # Повертаємо таблицю з новим стовпцем config_label.
    return df


def plot_cv_macro_f1_summary(summary_df: pd.DataFrame):
    # Додаємо зрозумілі підписи конфігурацій для осі X.
    # Для summary_df формуємо підписи з K, кількістю дерев і параметром C.
    df = add_config_label(summary_df)
    # Сортуємо конфігурації від найкращої до найгіршої за головною метрикою.
    df = df.sort_values("macro_f1_score_mean", ascending=False).reset_index(drop=True)

    # Індекси x використовуються як позиції стовпчиків на графіку.
    x = range(len(df))

    # Задаємо розмір графіка, достатній для підписів конфігурацій.
    plt.figure(figsize=(12, 7))

    # Будуємо стовпчики середнього macro F1-score для кожної конфігурації.
    bars = plt.bar(x, df["macro_f1_score_mean"])

    # Якщо є стандартне відхилення, додаємо його як похибку на графіку.
    if "macro_f1_score_std" in df.columns:
        # Errorbar показує стабільність результату між fold-ами.
        plt.errorbar(
            x,
            df["macro_f1_score_mean"],
            yerr=df["macro_f1_score_std"],
            fmt="none",
            capsize=5,
            linewidth=1.2
        )

    plt.title("Порівняння конфігурацій за середнім macro F1-score")
    plt.xlabel("Конфігурація моделі")
    plt.ylabel("Середнє значення macro F1-score")

    # Нижню межу осі Y наближаємо до значень метрик, щоб краще бачити різницю.
    min_value = max(0.0, df["macro_f1_score_mean"].min() - 0.03)
    # Верхня межа не перевищує 1.0, оскільки F1-score належить діапазону [0, 1].
    max_value = min(1.0, df["macro_f1_score_mean"].max() + 0.03)
    plt.ylim(min_value, max_value)

    plt.xticks(x, df["config_label"], rotation=20, ha="right")

    # Підписуємо кожен стовпчик точним числовим значенням.
    for bar, value in zip(bars, df["macro_f1_score_mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # Автоматично коригуємо відступи, щоб підписи не обрізалися.
    plt.tight_layout()
    # Зберігаємо ключовий графік вибору конфігурації у високій якості.
    plt.savefig(os.path.join(OUTPUT_PATH, "01_cv_macro_f1_summary.jpg"), dpi=300)
    # Закриваємо фігуру, щоб не накопичувати графіки в пам'яті.
    plt.close()


def plot_cv_main_metrics_summary(summary_df: pd.DataFrame):
    # Додаємо зрозумілі підписи конфігурацій для осі X.
    df = add_config_label(summary_df)
    # Зберігаємо той самий порядок конфігурацій, що і на графіку macro F1-score.
    df = df.sort_values("macro_f1_score_mean", ascending=False).reset_index(drop=True)

    # Індекси x використовуються як позиції стовпчиків на графіку.
    x = range(len(df))

    # Задаємо ширший формат, оскільки на графіку декілька груп метрик.
    plt.figure(figsize=(13, 7))

    # Перший стовпчик у групі показує загальну accuracy.
    plt.bar([i - 0.30 for i in x], df["accuracy_mean"], width=0.20, label="Accuracy")
    # Другий стовпчик показує macro precision для порівняння точності прогнозів.
    plt.bar([i - 0.10 for i in x], df["macro_precision_mean"], width=0.20, label="Macro precision")
    # Третій стовпчик показує macro recall для оцінки повноти розпізнавання класів.
    plt.bar([i + 0.10 for i in x], df["macro_recall_mean"], width=0.20, label="Macro recall")
    # Четвертий стовпчик показує головну метрику вибору параметрів.
    plt.bar([i + 0.30 for i in x], df["macro_f1_score_mean"], width=0.20, label="Macro F1-score")

    plt.title("Порівняння конфігурацій за основними метриками крос-валідації")
    plt.xlabel("Конфігурація моделі")
    plt.ylabel("Середнє значення метрики")

    # Визначаємо мінімальне значення серед усіх метрик для масштабування осі Y.
    min_metric = min(
        df["accuracy_mean"].min(),
        df["macro_precision_mean"].min(),
        df["macro_recall_mean"].min(),
        df["macro_f1_score_mean"].min()
    )

    plt.ylim(max(0.0, min_metric - 0.03), 1.0)

    plt.xticks(x, df["config_label"], rotation=20, ha="right")
    # Легенда пояснює, які стовпчики відповідають окремим метрикам.
    plt.legend()
    # Автоматично коригуємо відступи, щоб підписи не обрізалися.
    plt.tight_layout()

    # Зберігаємо порівняння основних метрик крос-валідації.
    plt.savefig(os.path.join(OUTPUT_PATH, "02_cv_main_metrics_summary.jpg"), dpi=300)
    # Закриваємо фігуру, щоб не накопичувати графіки в пам'яті.
    plt.close()


def plot_cv_fold_stability(cv_df: pd.DataFrame):
    # Копія таблиці fold-результатів використовується для побудови boxplot.
    df = cv_df.copy()

    # Визначаємо порядок конфігурацій за середнім macro F1-score.
    config_order = (
        df.groupby("config_name")["macro_f1_score"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # Задаємо розмір графіка, достатній для підписів конфігурацій.
    plt.figure(figsize=(12, 7))

    # positions зберігає координати boxplot для кожної конфігурації.
    positions = []
    # labels містить підписи конфігурацій із параметрами.
    labels = []
    # data містить значення macro F1-score за fold-ами.
    data = []

    # Для кожної конфігурації збираємо значення метрики на всіх fold-ах.
    for idx, config_name in enumerate(config_order, start=1):
        # Відбираємо рядки лише для поточної конфігурації.
        subset = df[df["config_name"] == config_name]
        # Додаємо всі fold-значення macro F1-score до boxplot.
        data.append(subset["macro_f1_score"].values)
        positions.append(idx)

        # Перший рядок використовується для отримання параметрів конфігурації.
        first_row = subset.iloc[0]
        labels.append(
            f"{config_name}\n"
            f"K={int(first_row['n_clusters'])}, "
            f"trees={int(first_row['n_estimators'])}, "
            f"C={first_row['svm_C']}"
        )

    # Boxplot показує розкид значень і середнє значення для кожної конфігурації.
    plt.boxplot(data, positions=positions, showmeans=True)

    plt.title("Стабільність macro F1-score за fold крос-валідації")
    plt.xlabel("Конфігурація моделі")
    plt.ylabel("Macro F1-score")

    # Беремо всі значення metric для адаптивного масштабу осі Y.
    all_values = df["macro_f1_score"]
    plt.ylim(max(0.0, all_values.min() - 0.03), min(1.0, all_values.max() + 0.03))

    plt.xticks(positions, labels, rotation=20, ha="right")
    # Автоматично коригуємо відступи, щоб підписи не обрізалися.
    plt.tight_layout()

    # Зберігаємо графік стабільності результатів між fold-ами.
    plt.savefig(os.path.join(OUTPUT_PATH, "03_cv_macro_f1_fold_stability.jpg"), dpi=300)
    # Закриваємо фігуру, щоб не накопичувати графіки в пам'яті.
    plt.close()


def plot_cv_fold_lines(cv_df: pd.DataFrame):
    # Копія таблиці використовується для побудови лінійних графіків по fold-ах.
    df = cv_df.copy()

    # Задаємо розмір графіка, достатній для підписів конфігурацій.
    plt.figure(figsize=(12, 7))

    # Будуємо окрему лінію для кожної конфігурації.
    for config_name, subset in df.groupby("config_name"):
        # Сортування за номером fold забезпечує правильну послідовність точок.
        subset = subset.sort_values("fold")

        # Параметри конфігурації беруться з першого рядка відповідної групи.
        first_row = subset.iloc[0]
        # Формуємо підпис лінії з назвою конфігурації та її параметрами.
        label = (
            f"{config_name} "
            f"(K={int(first_row['n_clusters'])}, "
            f"trees={int(first_row['n_estimators'])}, "
            f"C={first_row['svm_C']})"
        )

        # Лінійний графік показує, як змінювався macro F1-score між fold-ами.
        plt.plot(
            subset["fold"],
            subset["macro_f1_score"],
            marker="o",
            label=label
        )

    plt.title("Значення macro F1-score на окремих fold")
    plt.xlabel("Номер fold")
    plt.ylabel("Macro F1-score")

    # Нижня межа осі Y підбирається так, щоб показати відмінності між значеннями.
    min_value = max(0.0, df["macro_f1_score"].min() - 0.03)
    # Верхня межа обмежується 1.0, бо це максимальне можливе значення F1-score.
    max_value = min(1.0, df["macro_f1_score"].max() + 0.03)
    plt.ylim(min_value, max_value)

    # На осі X відображаємо всі номери fold-ів.
    plt.xticks(sorted(df["fold"].unique()))
    plt.legend(fontsize=8)
    # Автоматично коригуємо відступи, щоб підписи не обрізалися.
    plt.tight_layout()

    # Зберігаємо деталізований графік значень за fold-ами.
    plt.savefig(os.path.join(OUTPUT_PATH, "04_cv_macro_f1_by_fold.jpg"), dpi=300)
    # Закриваємо фігуру, щоб не накопичувати графіки в пам'яті.
    plt.close()


def plot_cv_time_comparison(summary_df: pd.DataFrame):
    # Додаємо зрозумілі підписи конфігурацій для осі X.
    # Формуємо підписи, щоб час можна було співвіднести з параметрами конфігурацій.
    df = add_config_label(summary_df)
    # Сортуємо конфігурації за сумарним часом виконання крос-валідації.
    df = df.sort_values("fold_time_seconds_sum", ascending=True).reset_index(drop=True)

    # Індекси x використовуються як позиції стовпчиків на графіку.
    x = range(len(df))

    # Задаємо розмір графіка, достатній для підписів конфігурацій.
    plt.figure(figsize=(12, 7))

    # Час переводиться з секунд у хвилини для зручнішого сприйняття.
    bars = plt.bar(x, df["fold_time_seconds_sum"] / 60)

    plt.title("Порівняння сумарного часу крос-валідації")
    plt.xlabel("Конфігурація моделі")
    plt.ylabel("Сумарний час, хв")

    plt.xticks(x, df["config_label"], rotation=20, ha="right")

    # Підписуємо кожен стовпчик числовим значенням часу.
    for bar, value in zip(bars, df["fold_time_seconds_sum"] / 60):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # Автоматично коригуємо відступи, щоб підписи не обрізалися.
    plt.tight_layout()

    # Зберігаємо графік для аналізу обчислювальної складності.
    plt.savefig(os.path.join(OUTPUT_PATH, "05_cv_time_comparison.jpg"), dpi=300)
    # Закриваємо фігуру, щоб не накопичувати графіки в пам'яті.
    plt.close()


def plot_final_metrics(final_metrics_df: pd.DataFrame):
    # У таблиці фінального тестування очікується один рядок із підсумковими метриками.
    row = final_metrics_df.iloc[0]

    # Формуємо словник метрик, які треба показати на підсумковому графіку.
    metrics = {
        "Accuracy": row["accuracy"],
        "Macro precision": row["macro_precision"],
        "Macro recall": row["macro_recall"],
        "Macro F1-score": row["macro_f1_score"],
        "Weighted F1-score": row["weighted_f1_score"]
    }

    # Назви метрик використовуються як підписи осі X.
    labels = list(metrics.keys())
    # Значення метрик використовуються як висоти стовпчиків.
    values = list(metrics.values())

    x = range(len(labels))

    # Задаємо компактний формат для одного підсумкового графіка.
    plt.figure(figsize=(10, 6))

    # Будуємо стовпчикову діаграму фінальних показників моделі.
    bars = plt.bar(x, values)

    plt.title("Фінальні метрики на відкладеній тестовій вибірці")
    plt.xlabel("Метрика")
    plt.ylabel("Значення")

    # Масштабуємо вісь Y, щоб різниця між близькими метриками була видимою.
    min_value = max(0.0, min(values) - 0.03)
    max_value = min(1.0, max(values) + 0.03)
    plt.ylim(min_value, max_value)

    plt.xticks(x, labels, rotation=15, ha="right")

    # Додаємо точні значення над стовпчиками.
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # Автоматично коригуємо відступи, щоб підписи не обрізалися.
    plt.tight_layout()

    # Зберігаємо графік фінальних метрик на test-вибірці.
    plt.savefig(os.path.join(OUTPUT_PATH, "06_final_test_metrics.jpg"), dpi=300)
    # Закриваємо фігуру, щоб не накопичувати графіки в пам'яті.
    plt.close()


def plot_class_f1_scores(class_metrics_df: pd.DataFrame):
    # Створюємо копію покласових метрик для безпечного сортування.
    df = class_metrics_df.copy()
    # Сортування за зростанням дозволяє одразу побачити найскладніші класи.
    df = df.sort_values("f1_score", ascending=True).reset_index(drop=True)

    # Індекси y використовуються як позиції горизонтальних стовпчиків.
    y = range(len(df))

    # Задаємо вертикальний формат для читабельного списку класів.
    plt.figure(figsize=(11, 8))

    # Горизонтальна діаграма зручніша для довгих назв класів.
    bars = plt.barh(y, df["f1_score"])

    plt.title("F1-score за класами на тестовій вибірці")
    plt.xlabel("F1-score")
    plt.ylabel("Клас")

    # Масштаб осі X фокусується на фактичному діапазоні F1-score.
    plt.xlim(max(0.0, df["f1_score"].min() - 0.05), 1.0)
    plt.yticks(y, df["class_name"], fontsize=8)

    # Біля кожного класу додається точне значення F1-score.
    for bar, value in zip(bars, df["f1_score"]):
        plt.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=8
        )

    # Автоматично коригуємо відступи, щоб підписи не обрізалися.
    plt.tight_layout()

    # Зберігаємо покласовий графік F1-score для аналізу помилок.
    plt.savefig(os.path.join(OUTPUT_PATH, "07_final_f1_score_by_class.jpg"), dpi=300)
    # Закриваємо фігуру, щоб не накопичувати графіки в пам'яті.
    plt.close()


def plot_class_precision_recall_f1(class_metrics_df: pd.DataFrame):
    # Створюємо копію покласових метрик для безпечного сортування.
    df = class_metrics_df.copy()
    # Класи впорядковуються від найкращих до найгірших за F1-score.
    df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)

    # Індекси x використовуються як позиції стовпчиків на графіку.
    x = range(len(df))

    # Задаємо широкий формат, щоб умістити всі назви класів.
    plt.figure(figsize=(15, 7))

    # Precision показує, наскільки точними були прогнози певного класу.
    plt.bar([i - 0.25 for i in x], df["precision"], width=0.25, label="Precision")
    # Recall показує, яку частину справжніх об'єктів класу модель знайшла.
    plt.bar(x, df["recall"], width=0.25, label="Recall")
    # F1-score поєднує precision і recall в одну збалансовану метрику.
    plt.bar([i + 0.25 for i in x], df["f1_score"], width=0.25, label="F1-score")

    plt.title("Precision, recall та F1-score за класами")
    plt.xlabel("Клас")
    plt.ylabel("Значення метрики")

    # Визначаємо мінімальне значення серед трьох метрик для налаштування осі Y.
    min_metric = min(
        df["precision"].min(),
        df["recall"].min(),
        df["f1_score"].min()
    )

    plt.ylim(max(0.0, min_metric - 0.05), 1.0)

    plt.xticks(x, df["class_name"], rotation=75, ha="right", fontsize=8)
    # Легенда дозволяє відрізнити precision, recall та F1-score.
    plt.legend()
    # Автоматично коригуємо відступи, щоб підписи не обрізалися.
    plt.tight_layout()

    # Зберігаємо графік порівняння трьох покласових метрик.
    plt.savefig(os.path.join(OUTPUT_PATH, "08_final_precision_recall_f1_by_class.jpg"), dpi=300)
    # Закриваємо фігуру, щоб не накопичувати графіки в пам'яті.
    plt.close()


def plot_lowest_classes(class_metrics_df: pd.DataFrame, n: int = 5):
    # Створюємо копію покласових метрик для безпечного сортування.
    df = class_metrics_df.copy()
    # Вибираємо n класів із найнижчим F1-score для окремого аналізу.
    df = df.sort_values("f1_score", ascending=True).head(n).reset_index(drop=True)

    # Індекси x використовуються як позиції стовпчиків на графіку.
    x = range(len(df))

    # Задаємо компактний формат для одного підсумкового графіка.
    plt.figure(figsize=(10, 6))

    # Для найскладніших класів окремо показуємо precision.
    plt.bar([i - 0.25 for i in x], df["precision"], width=0.25, label="Precision")
    # Recall допомагає зрозуміти, чи модель пропускає справжні приклади класу.
    plt.bar(x, df["recall"], width=0.25, label="Recall")
    # F1-score показує загальну якість для кожного зі складних класів.
    plt.bar([i + 0.25 for i in x], df["f1_score"], width=0.25, label="F1-score")

    plt.title(f"Класи з найнижчим значенням F1-score")
    plt.xlabel("Клас")
    plt.ylabel("Значення метрики")

    # Масштаб графіка адаптується до найменших значень трьох метрик.
    min_metric = min(
        df["precision"].min(),
        df["recall"].min(),
        df["f1_score"].min()
    )

    plt.ylim(max(0.0, min_metric - 0.05), 1.0)

    plt.xticks(x, df["class_name"], rotation=25, ha="right", fontsize=9)
    plt.legend()
    # Автоматично коригуємо відступи, щоб підписи не обрізалися.
    plt.tight_layout()

    # Зберігаємо графік класів, які потребують найбільшої уваги.
    plt.savefig(os.path.join(OUTPUT_PATH, "09_lowest_f1_classes.jpg"), dpi=300)
    # Закриваємо фігуру, щоб не накопичувати графіки в пам'яті.
    plt.close()


def plot_confusion_matrix_from_csv():
    # Матриця помилок читається з CSV, створеного під час фінального тестування.
    cm_path = TRAIN_RESULTS_PATH / "final_confusion_matrix.csv"

    # Якщо матриця ще не створена, графік пропускається без аварійного завершення.
    if not os.path.exists(cm_path):
        # У консолі виводиться пояснення, чому heatmap не була побудована.
        print("final_confusion_matrix.csv not found, skipping confusion matrix heatmap.")
        return

    # Перший стовпець CSV використовується як назви справжніх класів.
    cm_df = pd.read_csv(cm_path, index_col=0)

    # Великий розмір потрібний через 15 класів і довгі назви.
    plt.figure(figsize=(12, 10))

    # imshow візуалізує кількість правильних і помилкових прогнозів як теплову карту.
    plt.imshow(cm_df.values, aspect="auto")
    plt.title("Матриця помилок на тестовій вибірці")
    plt.xlabel("Прогнозований клас")
    plt.ylabel("Справжній клас")

    # Підписи осі X відповідають прогнозованим класам.
    plt.xticks(
        ticks=np.arange(len(cm_df.columns)),
        labels=cm_df.columns,
        rotation=90,
        fontsize=7
    )

    # Підписи осі Y відповідають справжнім класам.
    plt.yticks(
        ticks=np.arange(len(cm_df.index)),
        labels=cm_df.index,
        fontsize=7
    )

    # Colorbar показує, яке значення відповідає інтенсивності кольору.
    plt.colorbar(label="Кількість зображень")
    plt.tight_layout()

    # Зберігаємо матрицю помилок у форматі, придатному для звіту.
    plt.savefig(os.path.join(OUTPUT_PATH, "10_final_confusion_matrix_heatmap.jpg"), dpi=300)
    plt.close()




# =========================
# ROC analysis
# =========================

def load_dataset_for_roc(dataset_path: Path, max_images_per_class: int | None = None):
    """
    Завантаження шляхів до зображень і міток класів для відтворення
    того самого стратифікованого train/test поділу, що й у train_model.py.
    """
    # Список шляхів формується заново, щоб відтворити test-вибірку для ROC.
    image_paths = []
    # Паралельно зберігаємо назви класів для кожного зображення.
    labels = []

    # Класи обходяться у відсортованому порядку для відтворюваності.
    for class_name in sorted(os.listdir(dataset_path)):
        # Для кожного класу визначається відповідна папка зображень.
        class_folder = dataset_path / class_name

        # Пропускаємо елементи, які не є папками класів.
        if not class_folder.is_dir():
            continue

        count = 0

        # Файли в межах класу також сортуються для стабільного порядку.
        for file_name in sorted(os.listdir(class_folder)):
            # Обробляємо лише графічні формати, які підтримуються OpenCV.
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Додаємо повний шлях до зображення у список.
            image_paths.append(str(class_folder / file_name))
            # Для кожного шляху додається відповідна мітка класу.
            labels.append(class_name)

            count += 1

            # Обмеження кількості зображень має збігатися з навчальним скриптом.
            if max_images_per_class is not None and count >= max_images_per_class:
                break

    # Повертаємо масиви, сумісні з train_test_split.
    return np.array(image_paths), np.array(labels)


def build_feature_matrix_for_roc(image_paths, labels, kmeans_model):
    """
    Формує матрицю ознак для ROC-аналізу.
    Повертає також valid_labels, щоб уникнути розсинхронізації,
    якщо окреме зображення не вдалося зчитати.
    """
    # Тут накопичуються гібридні вектори ознак для тестових зображень.
    features = []
    # valid_labels зберігає мітки тільки успішно зчитаних зображень.
    valid_labels = []

    # Обходимо шляхи й мітки синхронно, щоб не втратити відповідність класів.
    for idx, (image_path, label) in enumerate(zip(image_paths, labels)):
        # Зчитуємо зображення у форматі BGR, який використовує OpenCV.
        image = cv2.imread(image_path)

        # Якщо файл пошкоджений або недоступний, його треба пропустити.
        if image is None:
            print(f"Warning: cannot read image {image_path}")
            continue

        # Формуємо той самий вектор ознак, що використовувався під час навчання.
        feature_vector = extract_full_feature_vector(image, kmeans_model)
        # Додаємо вектор ознак до майбутньої матриці X_test.
        features.append(feature_vector)
        # Зберігаємо мітку лише для успішно обробленого зображення.
        valid_labels.append(label)

        # Періодичний прогрес допомагає контролювати тривалу побудову ознак.
        if idx % 100 == 0:
            print(f"Built ROC features for {idx}/{len(image_paths)} images")

    # Повертаємо матрицю ознак і синхронізовані з нею мітки.
    return np.array(features, dtype=np.float32), np.array(valid_labels)


def plot_multiclass_roc():
    """
    Побудова ROC-кривої для багатокласової задачі у форматі one-vs-rest.
    Зберігає micro-average, macro-average ROC та таблицю AUC.
    """
    # Повідомлення в консолі відокремлює етап ROC-аналізу від інших графіків.
    print("\nBuilding ROC curve...")

    # Завантажуємо ті самі шляхи та мітки, що використовувалися у навчальному пайплайні.
    image_paths, labels = load_dataset_for_roc(
        DATASET_PATH,
        max_images_per_class=MAX_IMAGES_PER_CLASS
    )

    # Відтворюємо відкладену test-вибірку з тим самим random_state і stratify.
    _, test_paths, _, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # Завантажуємо фінальний BoVW-словник для побудови ознак тестових зображень.
    kmeans = joblib.load(MODELS_PATH / "hybrid_bovw_kmeans.joblib")
    # Завантажуємо фінальний класифікатор для отримання ймовірностей класів.
    classifier = joblib.load(MODELS_PATH / "hybrid_classifier.joblib")
    # Завантажуємо LabelEncoder, щоб правильно кодувати назви класів.
    label_encoder = joblib.load(MODELS_PATH / "hybrid_label_encoder.joblib")

    # Формуємо матрицю ознак лише для тестових зображень.
    X_test, valid_test_labels = build_feature_matrix_for_roc(
        test_paths,
        test_labels,
        kmeans
    )

    # Текстові назви класів перетворюються у числові індекси.
    y_test = label_encoder.transform(valid_test_labels)
    # Отримуємо список класів у тому порядку, який використовує модель.
    class_names = label_encoder.classes_
    # Кількість класів потрібна для циклу one-vs-rest ROC.
    n_classes = len(class_names)

    # predict_proba повертає ймовірності, необхідні для ROC-кривих.
    y_score = classifier.predict_proba(X_test)

    # Бінаризація переводить багатокласову задачу у набір one-vs-rest порівнянь.
    y_test_bin = label_binarize(
        y_test,
        classes=np.arange(n_classes)
    )

    # Словник fpr зберігає False Positive Rate для кожного класу.
    fpr = {}
    # Словник tpr зберігає True Positive Rate для кожного класу.
    tpr = {}
    # roc_auc зберігає площу під ROC-кривою для класів і середніх оцінок.
    roc_auc = {}

    # Для кожного класу будується окрема one-vs-rest ROC-крива.
    # Інтерполюємо TPR кожного класу на спільній сітці FPR.
    for i in range(n_classes):
        # ROC для класу i порівнює цей клас проти всіх інших.
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        # AUC стисло характеризує якість відокремлення класу i.
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average враховує всі бінарні рішення разом.
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_bin.ravel(),
        y_score.ravel()
    )
    # Обчислюємо AUC для micro-average ROC.
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Для macro-average формуємо спільну сітку значень FPR.
    all_fpr = np.unique(
        np.concatenate([fpr[i] for i in range(n_classes)])
    )

    # Тут накопичується середній TPR для macro-average кривої.
    mean_tpr = np.zeros_like(all_fpr)

    # Для кожного класу будується окрема one-vs-rest ROC-крива.
    # Інтерполюємо TPR кожного класу на спільній сітці FPR.
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Ділимо суму TPR на кількість класів, отримуючи macro-average.
    mean_tpr /= n_classes

    # Зберігаємо FPR macro-average ROC у спільному словнику.
    fpr["macro"] = all_fpr
    # Зберігаємо усереднений TPR macro-average ROC.
    tpr["macro"] = mean_tpr
    # Обчислюємо AUC для macro-average ROC.
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Квадратний формат добре підходить для ROC-графіка.
    plt.figure(figsize=(8, 7))

    # На графік наносяться micro-average та macro-average криві.
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC, AUC = {roc_auc['micro']:.3f}"
    )

    # На графік наносяться micro-average та macro-average криві.
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC, AUC = {roc_auc['macro']:.3f}"
    )

    # Діагональ показує рівень випадкового класифікатора.
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.title("Багатокласова ROC-крива")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Один файл у папці ROC і копія у validation_plots для зручної вставки в диплом.
    # Ім'я файлу має номер, узгоджений з іншими validation_plots.
    roc_plot_name = "11_roc_curve_macro_micro.jpg"

    
    # Зберігаємо ROC-криву у папку графіків валідації.
    plt.savefig(VALIDATION_OUTPUT_PATH / roc_plot_name, dpi=300)
    plt.close()

    # У цьому списку збираються AUC-значення для таблиці CSV.
    auc_rows = []

    # Додаємо AUC окремо для кожного класу.
    for i, class_name in enumerate(class_names):
        # Кожен рядок містить назву класу та відповідне ROC AUC.
        auc_rows.append({
            "class_name": class_name,
            "roc_auc": roc_auc[i]
        })

    # Кожен рядок містить назву класу та відповідне ROC AUC.
    auc_rows.append({
        "class_name": "micro_average",
        "roc_auc": roc_auc["micro"]
    })

    # Кожен рядок містить назву класу та відповідне ROC AUC.
    auc_rows.append({
        "class_name": "macro_average",
        "roc_auc": roc_auc["macro"]
    })

    # Перетворюємо список словників у таблицю pandas.
    auc_df = pd.DataFrame(auc_rows)
    
    # Зберігаємо таблицю AUC для додаткового аналізу у звіті.
    auc_df.to_csv(VALIDATION_OUTPUT_PATH / "11_roc_auc_values.csv", index=False)


def main():
    # Спочатку гарантуємо наявність папки для всіх вихідних графіків.
    ensure_output_dir()

    print("=" * 80)
    print("Building validation plots and ROC analysis from saved results")
    print("=" * 80)

    # Завантажуємо повні результати по всіх fold-ах крос-валідації.
    cv_df = load_required_csv("cv_results_all_folds.csv")
    # Завантажуємо агреговану таблицю середніх метрик за конфігураціями.
    summary_df = load_required_csv("cv_results_summary.csv")
    # Завантажуємо фінальні метрики на відкладеній test-вибірці.
    final_metrics_df = load_required_csv("final_test_metrics_summary.csv")
    # Завантажуємо покласові precision, recall та F1-score.
    class_metrics_df = load_required_csv("final_class_metrics_by_class.csv")

    # Будуємо головний графік вибору конфігурації за macro F1-score.
    plot_cv_macro_f1_summary(summary_df)
    # Будуємо порівняння конфігурацій за кількома метриками.
    plot_cv_main_metrics_summary(summary_df)
    # Будуємо boxplot для оцінки стабільності результатів між fold-ами.
    plot_cv_fold_stability(cv_df)
    # Будуємо лінійний графік macro F1-score для кожного fold.
    plot_cv_fold_lines(cv_df)
    # Будуємо графік часу, щоб оцінити обчислювальну складність.
    plot_cv_time_comparison(summary_df)

    # Будуємо графік фінальних метрик на test-вибірці.
    plot_final_metrics(final_metrics_df)
    # Будуємо графік F1-score окремо для кожного класу.
    plot_class_f1_scores(class_metrics_df)
    # Будуємо детальний графік precision, recall і F1-score за класами.
    plot_class_precision_recall_f1(class_metrics_df)
    # Окремо показуємо п'ять найскладніших класів.
    plot_lowest_classes(class_metrics_df, n=5)
    # Будуємо heatmap матриці помилок із CSV-файлу.
    plot_confusion_matrix_from_csv()

    # Додаємо ROC-аналіз як додаткову оцінку моделі.
    plot_multiclass_roc()

    print("\nValidation plots saved to:")
    # Виводимо шлях, де збережено всі побудовані графіки.
    print(VALIDATION_OUTPUT_PATH)



if __name__ == "__main__":
    main()

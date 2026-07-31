"""
Аналіз набору даних PlantVillage для дипломної роботи.

Файл відповідає за:
- завантаження шляхів до зображень і міток класів;
- аналіз розподілу зображень за класами, культурами та станом рослини;
- побудову графіків EDA для вставки у звіт;
- створення схеми поділу даних на train_val, fold-и крос-валідації та test.
"""

# os використовується для обходу папок датасету та збереження результатів.
# Модуль не змінює датасет, а лише читає його структуру.
import os

# sys потрібний для додавання кореня проєкту до шляхів імпорту.
import sys

# Path застосовується для коректної побудови шляхів незалежно від ОС.
from pathlib import Path

# Визначаємо фактичне розташування цього файлу у файловій системі.
CURRENT_FILE = Path(__file__).resolve()

# Якщо файл лежить у scripts, коренем проєкту вважається батьківська папка.
PROJECT_ROOT = CURRENT_FILE.parent.parent if CURRENT_FILE.parent.name == "scripts" else CURRENT_FILE.parent

# Додаємо корінь проєкту до sys.path, щоб файл можна було запускати з різних місць.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Counter використовується для швидкого підрахунку кількості елементів у класах.
from collections import Counter

# Matplotlib переводиться у режим Agg, щоб графіки можна було зберігати без відкриття вікна.
import matplotlib
matplotlib.use("Agg")

# pyplot використовується для побудови стовпчикових діаграм.
import matplotlib.pyplot as plt

# pandas потрібний для формування таблиць і збереження CSV-файлів.
# Табличні результати зберігаються поруч із графіками для відтворюваності аналізу.
import pandas as pd

# train_test_split імпортований для узгодженості з логікою поділу даних у проєкті.
from sklearn.model_selection import train_test_split


# =========================
# Global settings
# =========================

# Шлях до набору даних із підпапками класів.
# Відносний шлях дозволяє запускати скрипт із кореня проєкту без додаткових параметрів.
DATASET_PATH = "dataset/PlantVillage"

# Усі графіки та таблиці аналізу датасету зберігаються в окрему папку results.
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "dataset_analysis")

# Початкові значення залишені для сумісності зі старою версією файлу.
MAX_IMAGES_PER_CLASS = None
TEST_SIZE = 0.2
RANDOM_STATE = 7

# У фінальній реалізації використовується обмеження до 500 зображень на клас.
MAX_IMAGES_PER_CLASS = 500

# Тестова частина становить 20% використаного набору даних.
TEST_SIZE = 0.2

# Кількість fold-ів у StratifiedKFold, яка використовується у train_model.py.
N_SPLITS = 8


# Функція не зчитує самі пікселі, а тільки формує список файлів для аналізу.
def load_dataset(dataset_path: str, max_images_per_class=None):
    """
    Завантажує шляхи до зображень та відповідні назви класів.
    Очікувана структура:
    dataset/PlantVillage/class_name/image.jpg
    """

    # Список шляхів до зображень формується окремо від списку міток.
    image_paths = []
    labels = []

    # Кожна папка всередині dataset_path відповідає окремому класу захворювання.
    # sorted() забезпечує стабільний порядок класів між різними запусками.
    for class_name in sorted(os.listdir(dataset_path)):
        class_folder = os.path.join(dataset_path, class_name)

        # Якщо елемент не є папкою класу, він пропускається.
        if not os.path.isdir(class_folder):
            continue

        # Лічильник потрібний для обмеження кількості зображень на клас.
        count = 0

        # Обходимо файли всередині поточної папки класу.
        # Порядок файлів тут не впливає на статистику, бо рахується лише кількість.
        for file_name in os.listdir(class_folder):
            # До аналізу включаються лише поширені формати зображень.
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Зберігаємо повний шлях до зображення та відповідну мітку класу.
            image_paths.append(os.path.join(class_folder, file_name))
            labels.append(class_name)

            count += 1

            # Обмеження використовується для часткового балансування та прискорення експериментів.
            if max_images_per_class is not None and count >= max_images_per_class:
                break

    # Повертаємо паралельні списки: шлях до файлу і назва класу.
    return image_paths, labels


# Допоміжна функція потрібна для агрегування 15 класів у 3 культури.
def get_crop_name(class_name: str) -> str:
    """
    Визначає культуру за назвою класу.
    """

    # Класи PlantVillage мають префікс із назвою культури.
    if class_name.startswith("Pepper"):
        return "Pepper"

    # Класи картоплі починаються з Potato.
    if class_name.startswith("Potato"):
        return "Potato"

    # Класи томату починаються з Tomato.
    if class_name.startswith("Tomato"):
        return "Tomato"

    # Резервне значення на випадок появи нового або неочікуваного класу.
    return "Unknown"


# Допоміжна функція спрощує аналіз співвідношення здорових і хворих рослин.
def get_health_status(class_name: str) -> str:
    """
    Визначає стан рослини: healthy або diseased.
    """

    # Якщо в назві класу є healthy, зображення належить до здорових рослин.
    if "healthy" in class_name.lower():
        return "Healthy"

    # Усі інші класи в цьому наборі даних вважаються ураженими.
    return "Diseased"


# Цей графік показує незбалансованість класів у використаній підмножині.
def save_class_distribution(labels):
    """
    Будує стовпчикову діаграму розподілу зображень за класами.
    """

    # Підраховуємо кількість зображень для кожної назви класу.
    class_counts = Counter(labels)

    # Перетворюємо словник частот у DataFrame для зручного сортування та збереження.
    df = pd.DataFrame({
        "class_name": list(class_counts.keys()),
        "count": list(class_counts.values())
    })

    # Сортування за спаданням дозволяє швидко побачити найчисленніші класи.
    df = df.sort_values("count", ascending=False)

    # Створюємо графік розподілу класів для EDA-частини роботи.
    # Розмір фігури обраний так, щоб довгі назви класів залишалися читабельними.
    plt.figure(figsize=(14, 7))
    plt.bar(df["class_name"], df["count"])
    plt.title("Distribution of images by class")
    plt.xlabel("Class")
    plt.ylabel("Number of images")

    # Підписи класів повертаються під кутом, щоб довгі назви не накладалися одна на одну.
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    # Графік зберігається у форматі JPG з достатньою роздільною здатністю для звіту.
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.jpg"), dpi=300)
    plt.close()

    # DataFrame повертається, щоб далі зберегти його у CSV у main().
    return df


# Цей графік показує, які культури переважають у наборі даних.
def save_crop_distribution(labels):
    """
    Будує діаграму розподілу зображень за культурами.
    """

    # Перетворюємо повні назви класів у назви культур: Pepper, Potato, Tomato.
    crops = [get_crop_name(label) for label in labels]

    # Підраховуємо кількість зображень для кожної культури.
    crop_counts = Counter(crops)

    # Формуємо таблицю для побудови графіка та подальшого збереження.
    df = pd.DataFrame({
        "crop": list(crop_counts.keys()),
        "count": list(crop_counts.values())
    })

    # Сортування спрощує візуальне порівняння культур між собою.
    df = df.sort_values("count", ascending=False)

    # Побудова компактної діаграми розподілу за культурами.
    plt.figure(figsize=(8, 6))
    plt.bar(df["crop"], df["count"])
    plt.title("Distribution of images by crop")
    plt.xlabel("Crop")
    plt.ylabel("Number of images")
    plt.tight_layout()

    # Зберігаємо графік у папку результатів аналізу датасету.
    plt.savefig(os.path.join(OUTPUT_DIR, "crop_distribution.jpg"), dpi=300)
    plt.close()

    return df


# Цей графік узагальнює датасет до двох станів: здорові та уражені.
def save_health_status_distribution(labels):
    """
    Будує діаграму розподілу зображень за станом: здорові / уражені.
    """

    # Для кожного класу визначається узагальнений стан рослини.
    statuses = [get_health_status(label) for label in labels]

    # Підраховуємо кількість здорових і уражених прикладів.
    status_counts = Counter(statuses)

    # Таблиця використовується і для побудови графіка, і для CSV-звіту.
    df = pd.DataFrame({
        "status": list(status_counts.keys()),
        "count": list(status_counts.values())
    })

    # Сортування робить порядок стовпчиків стабільним і зрозумілим.
    df = df.sort_values("count", ascending=False)

    # Будуємо окремий графік для порівняння здорових та уражених рослин.
    plt.figure(figsize=(7, 6))
    plt.bar(df["status"], df["count"])
    plt.title("Distribution of images by plant condition")
    plt.xlabel("Condition")
    plt.ylabel("Number of images")
    plt.tight_layout()

    # Збереження графіка виконується без відображення вікна завдяки backend Agg.
    plt.savefig(os.path.join(OUTPUT_DIR, "health_status_distribution.jpg"), dpi=300)
    plt.close()

    return df


# Схема поділу потрібна для пояснення методики валідації у дипломі.
def save_kfold_split_bar_chart():
    """
    Побудова bar chart для схеми поділу даних:
    80% train_val поділено на 8 fold-ів по 10%,
    20% відкладено як тестову вибірку.
    """

    # Створюємо папку результатів, якщо вона ще не існує.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Вісім fold-ів відповідають 80% train_val, а Test — окремим 20%.
    labels = [
        "Розбиття 1", "Розбиття 2", "Розбиття 3", "Розбиття 4",
        "Розбиття 5", "Розбиття 6", "Розбиття 7", "Розбиття 8",
        "Тестова вибірка"
    ]

    # Кожен fold становить приблизно 10% від усього використаного набору.
    # Сума часток дорівнює 100%, що показує повний розподіл використаних даних.
    percentages = [10, 10, 10, 10, 10, 10, 10, 10, 20]

    # Позиції стовпчиків на осі X.
    x = range(len(labels))

    # Широкий формат потрібний, щоб усі fold-и були добре видимі.
    plt.figure(figsize=(13, 6))

    # Будуємо базову стовпчикову діаграму поділу даних.
    # Кожен стовпчик відповідає окремій частині вибірки.
    bars = plt.bar(x, percentages)

    plt.title("Поділ набору даних для крос-валідації та фінального тестування")
    plt.xlabel("Частини вибірки")
    plt.ylabel("Частка від використаного набору даних, %")

    plt.xticks(x, labels, rotation=0)
    plt.ylim(0, 25)

    # Над кожним стовпчиком виводиться його частка у відсотках.
    for bar, value in zip(bars, percentages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value}%",
            ha="center",
            va="bottom",
            fontsize=11
        )

    # Підпис пояснює, що перші 8 стовпчиків використовуються у StratifiedKFold.
    plt.text(
        3.5,
        -4.2,
        "Train_val = 80%: використовується для StratifiedKFold, k = 8",
        ha="center",
        va="center",
        fontsize=11
    )

    # Окремо позначається test-вибірка, яка не бере участі у виборі параметрів.
    plt.text(
        8,
        -4.2,
        "Test = 20%:\nвідкладена вибірка",
        ha="center",
        va="center",
        fontsize=10
    )

    # Нижня лінія-групування показує, що Fold 1–Fold 8 належать до train_val.
    plt.plot([-.4, 7.4], [-1.5, -1.5], linewidth=1.5)
    plt.plot([-.4, -.4], [-1.2, -1.8], linewidth=1.5)
    plt.plot([7.4, 7.4], [-1.2, -1.8], linewidth=1.5)

    # Окрема коротка лінія-групування виділяє тестову частину.
    plt.plot([7.65, 8.35], [-1.5, -1.5], linewidth=1.5)
    plt.plot([7.65, 7.65], [-1.2, -1.8], linewidth=1.5)
    plt.plot([8.35, 8.35], [-1.2, -1.8], linewidth=1.5)

    plt.tight_layout()

    # bbox_inches="tight" потрібний, щоб нижні пояснювальні підписи не обрізалися.
    plt.savefig(
        os.path.join(OUTPUT_DIR, "kfold_data_split_bar_chart.jpg"),
        dpi=300,
        bbox_inches="tight"
    )

    # Закриваємо фігуру, щоб не накопичувати об'єкти matplotlib у пам'яті.
    plt.close()



# main() об’єднує всі кроки аналізу датасету в один відтворюваний запуск.
def main():
    # Головна функція створює папку результатів і запускає всі етапи EDA.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Завантажуємо список зображень і класів з урахуванням обмеження на клас.
    image_paths, labels = load_dataset(
        DATASET_PATH,
        max_images_per_class=MAX_IMAGES_PER_CLASS
    )

    # Порожній список означає помилковий шлях або неправильну структуру датасету.
    if len(image_paths) == 0:
        raise ValueError("Датасет порожній або шлях до нього неправильний.")

    # Виводимо загальну кількість використаних зображень і класів.
    print(f"Total images: {len(image_paths)}")
    print(f"Total classes: {len(set(labels))}")

    # Послідовно будуємо три основні EDA-графіки.
    # class_df використовується і для графіка, і для CSV-таблиці.
    class_df = save_class_distribution(labels)
    # crop_df агрегує класи за культурою, що спрощує опис структури датасету.
    crop_df = save_crop_distribution(labels)
    # health_df показує загальне співвідношення healthy/diseased.
    health_df = save_health_status_distribution(labels)

    # Додатково будуємо рисунок 3.4.4 зі схемою 80/20 та k=8.
    save_kfold_split_bar_chart()

    # Виводимо таблиці у консоль для швидкої перевірки результатів.
    print("\nClass distribution:")
    print(class_df.to_string(index=False))

    print("\nCrop distribution:")
    print(crop_df.to_string(index=False))

    print("\nHealth status distribution:")
    print(health_df.to_string(index=False))

    # Зберігаємо всі таблиці, щоб їх можна було вставити у звіт або перевірити окремо.
    class_df.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)
    crop_df.to_csv(os.path.join(OUTPUT_DIR, "crop_distribution.csv"), index=False)
    health_df.to_csv(os.path.join(OUTPUT_DIR, "health_status_distribution.csv"), index=False)

    # Повідомлення показує, куди були збережені графіки й CSV-файли.
    # Повідомлення залишено як індикатор успішного завершення скрипта.
    print("\nCharts and CSV files saved to the photo folder.")


# Запуск main() виконується тільки при прямому запуску цього файлу.
if __name__ == "__main__":
    main()

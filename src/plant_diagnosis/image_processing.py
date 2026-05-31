"""
Модуль попередньої обробки зображень та формування гібридного вектора ознак
для системи автоматизованого діагностування хвороб рослинних культур.

Файл містить функції для:
- стандартизації розміру зображення;
- перевірки якості фото;
- виділення ORB-дескрипторів;
- перетворення локальних дескрипторів у BoVW-гістограму;
- формування колірних, текстурних та статистичних ознак;
- об'єднання всіх груп ознак у фінальний вектор для класифікатора.
"""

# OpenCV використовується для зміни розміру, перетворення кольорів,
# пошуку ORB-дескрипторів і побудови простих колірних масок.
import cv2

# NumPy потрібний для роботи з векторами, гістограмами та матрицями ознак.
import numpy as np

# local_binary_pattern використовується для побудови текстурних LBP-ознак.
from skimage.feature import local_binary_pattern


# Єдиний стандартний розмір зображень для всіх етапів обробки.
# Це забезпечує однакову довжину ознак і стабільність роботи алгоритмів.
IMAGE_SIZE = (256, 256)


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Попередня обробка зображення.

    Повертає:
    - resized: кольорове зображення BGR після зміни розміру;
    - gray: зображення у відтінках сірого для ORB/LBP.
    """

    # Зображення приводиться до фіксованого розміру 256 x 256.
    # Це потрібно, щоб усі наступні ознаки будувалися в однакових умовах.
    resized = cv2.resize(image, IMAGE_SIZE)

    # Для ORB, LBP і перевірки різкості достатньо grayscale-представлення.
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Гаусове згладжування зменшує дрібний шум, який може заважати ORB і LBP.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Вирівнювання гістограми підвищує контрастність grayscale-зображення.
    # Це допомагає краще виділяти локальні структури листка.
    gray = cv2.equalizeHist(gray)

    # Повертаємо і кольорову, і grayscale-версію, бо вони потрібні різним ознакам.
    return resized, gray


def check_image_quality(gray_image: np.ndarray, blur_threshold: float = 35.0) -> bool:
    """
    Перевірка розмитості зображення через дисперсію лапласіана.
    Чим менше значення, тим більш розмите зображення.
    """

    # Лапласіан реагує на різкі переходи яскравості, тобто на краї та деталі.
    # Якщо зображення розмите, таких переходів мало, і дисперсія буде низькою.
    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    # Порогове значення використовується як просте бізнес-правило якості фото.
    # True означає, що фото достатньо чітке для подальшої класифікації.
    return blur_score >= blur_threshold


def extract_orb_descriptors(image: np.ndarray) -> tuple[list, np.ndarray | None, np.ndarray, np.ndarray]:
    """
    Виділення ключових точок та дескрипторів ORB.

    ORB використовує FAST для пошуку ключових точок
    та BRIEF-подібний дескриптор для опису їх околу.
    """

    # Спочатку виконується стандартна попередня обробка зображення.
    # ORB працює з grayscale-зображенням, але resized потрібний для інших ознак.
    resized, gray = preprocess_image(image)

    # ORB налаштований на пошук до 1000 ключових точок на одному зображенні.
    # Це дає достатньо локальної інформації, але не робить обчислення надто важкими.
    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        fastThreshold=15
    )

    # detectAndCompute одночасно знаходить ключові точки та формує дескриптори.
    # Якщо інформативних точок немає, descriptors може бути None.
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Повертаємо не лише дескриптори, а й оброблені версії зображення.
    # Це дозволяє не виконувати preprocess_image повторно у фінальній функції.
    return keypoints, descriptors, resized, gray


def descriptors_to_bovw(descriptors: np.ndarray | None, kmeans_model) -> np.ndarray:
    """
    Перетворення ORB-дескрипторів у вектор фіксованої довжини
    методом Bag of Visual Words.
    """

    # Кількість кластерів K відповідає кількості візуальних слів у словнику.
    number_of_clusters = kmeans_model.n_clusters

    # Гістограма має фіксовану довжину K незалежно від кількості ключових точок.
    histogram = np.zeros(number_of_clusters, dtype=np.float32)

    # Якщо ORB не знайшов дескрипторів, повертається нульова BoVW-гістограма.
    # Це дозволяє не пропускати зображення через відсутність локальних точок.
    if descriptors is None or len(descriptors) == 0:
        return histogram

    # Кожен ORB-дескриптор відноситься до найближчого кластера MiniBatchKMeans.
    # Номер кластера інтерпретується як номер візуального слова.
    words = kmeans_model.predict(descriptors)

    # Підраховується, скільки разів кожне візуальне слово зустрілося на зображенні.
    for word in words:
        histogram[word] += 1

    # L2-нормалізація зменшує залежність вектора від кількості знайдених точок.
    # 1e-7 запобігає діленню на нуль для порожніх або дуже малих гістограм.
    histogram = histogram / (np.linalg.norm(histogram) + 1e-7)

    return histogram


def extract_hsv_histogram(image: np.ndarray) -> np.ndarray:
    """
    Формування HSV-гістограми.

    HSV дозволяє краще описувати колірні симптоми:
    пожовтіння, коричневі плями, темні уражені ділянки.
    """

    # Зображення знову приводиться до стандартного розміру для стабільності ознак.
    resized = cv2.resize(image, IMAGE_SIZE)

    # HSV-простір зручніший для аналізу кольору, ніж BGR/RGB,
    # бо відтінок, насиченість і яскравість описуються окремо.
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Для кожного HSV-каналу формується окрема гістограма.
    # Канал H має діапазон 0-180 в OpenCV, а S та V — 0-256.
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    # Три гістограми об'єднуються в один колірний вектор.
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()

    # Перетворення до float32 робить тип сумісним із іншими ознаками та sklearn.
    hist = hist.astype(np.float32)

    # Нормалізація потрібна, щоб масштаб гістограми не домінував над іншими ознаками.
    hist = hist / (np.linalg.norm(hist) + 1e-7)

    return hist


def extract_color_statistics(image: np.ndarray) -> np.ndarray:
    """
    Статистичні колірні ознаки в HSV:
    середнє та стандартне відхилення каналів H, S, V.
    """

    # Використовується той самий стандартний розмір, що й для інших ознак.
    resized = cv2.resize(image, IMAGE_SIZE)

    # Перехід у HSV дозволяє окремо аналізувати відтінок і яскравість листка.
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Середні значення каналів описують загальний колірний стан зображення.
    means = hsv.mean(axis=(0, 1))

    # Стандартні відхилення показують неоднорідність кольорів на листку.
    stds = hsv.std(axis=(0, 1))

    # Вектор містить 6 чисел: mean(H,S,V) та std(H,S,V).
    features = np.concatenate([means, stds]).astype(np.float32)

    return features


def extract_lbp_features(gray_image: np.ndarray) -> np.ndarray:
    """
    Формування LBP-ознак для опису текстури поверхні листка.

    LBP корисний для аналізу плям, нальоту, некрозу
    та неоднорідних текстур.
    """

    # Радіус 1 означає аналіз локального околу навколо кожного пікселя.
    radius = 1

    # Для radius=1 використовується 8 сусідніх точок навколо центрального пікселя.
    points = 8 * radius

    # Uniform LBP зменшує кількість шаблонів і робить гістограму компактнішою.
    lbp = local_binary_pattern(
        gray_image,
        P=points,
        R=radius,
        method="uniform"
    )

    # Для uniform LBP кількість бінів дорівнює P + 2.
    n_bins = points + 2

    # Гістограма рахує частоту появи різних локальних текстурних шаблонів.
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_bins + 1),
        range=(0, n_bins)
    )

    # Перетворення типу потрібне для подальшого об'єднання з іншими ознаками.
    hist = hist.astype(np.float32)

    # Нормалізація за сумою перетворює гістограму на розподіл частот.
    hist = hist / (hist.sum() + 1e-7)

    return hist


def extract_disease_area_features(image: np.ndarray) -> np.ndarray:
    """
    Приблизне виділення потенційно уражених ділянок за кольором.

    Ознака не є точним діагнозом, але допомагає оцінити,
    яку частку зображення займають жовто-коричневі або темні області.
    """

    # Зображення стандартизується за розміром перед побудовою масок.
    resized = cv2.resize(image, IMAGE_SIZE)

    # Маски будуються в HSV, бо в цьому просторі простіше задавати колірні діапазони.
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Жовто-коричневі ділянки, які часто відповідають плямам/ураженню.
    # Межі підібрані евристично для грубої оцінки симптомів на листку.
    lower_yellow_brown = np.array([10, 35, 30])
    upper_yellow_brown = np.array([45, 255, 255])
    mask_yellow_brown = cv2.inRange(hsv, lower_yellow_brown, upper_yellow_brown)

    # Темні ділянки, які можуть відповідати некрозу.
    # Ця маска допомагає врахувати чорні або дуже темні пошкоджені області.
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    # Об'єднання двох масок формує приблизну область потенційного ураження.
    disease_mask = cv2.bitwise_or(mask_yellow_brown, mask_dark)

    # Частка ураженої області показує, яку частину зображення займають такі пікселі.
    disease_area_ratio = np.sum(disease_mask > 0) / disease_mask.size

    # Якщо потенційно уражені пікселі знайдено, рахуємо їхні HSV-статистики.
    # Це додає не лише площу, а й колірний опис уражених областей.
    if np.sum(disease_mask > 0) > 0:
        affected_pixels = hsv[disease_mask > 0]
        affected_mean = affected_pixels.mean(axis=0)
        affected_std = affected_pixels.std(axis=0)
    else:
        # Якщо маска порожня, використовуються нульові значення,
        # щоб довжина вектора ознак залишалася незмінною.
        affected_mean = np.zeros(3)
        affected_std = np.zeros(3)

    # Фінальний блок ознак містить площу ураження,
    # середні значення HSV та стандартні відхилення HSV в уражених областях.
    features = np.concatenate([
        np.array([disease_area_ratio]),
        affected_mean,
        affected_std
    ]).astype(np.float32)

    return features


def extract_full_feature_vector(image: np.ndarray, kmeans_model) -> np.ndarray:
    """
    Формування повного комбінованого вектора ознак:

    ORB/BoVW + HSV histogram + HSV statistics + LBP + disease area features.
    """

    # ORB-функція одночасно повертає дескриптори, resized-зображення та gray-версію.
    _, descriptors, resized, gray = extract_orb_descriptors(image)

    # ORB-дескриптори перетворюються в BoVW-гістограму фіксованої довжини.
    bovw_features = descriptors_to_bovw(descriptors, kmeans_model)

    # HSV-гістограма описує загальний розподіл кольорів на листку.
    hsv_histogram = extract_hsv_histogram(resized)

    # Статистичні HSV-ознаки описують середній колір і варіативність кольорів.
    color_statistics = extract_color_statistics(resized)

    # LBP-ознаки описують текстуру листка та локальні неоднорідності.
    lbp_features = extract_lbp_features(gray)

    # Ознаки уражених ділянок додають евристичний опис плям і темних областей.
    disease_area_features = extract_disease_area_features(resized)

    # Усі групи ознак об'єднуються в один гібридний вектор для класифікатора.
    full_vector = np.concatenate([
        bovw_features,
        hsv_histogram,
        color_statistics,
        lbp_features,
        disease_area_features
    ])

    # Повертаємо float32, щоб зменшити обсяг пам'яті і забезпечити сумісність зі sklearn.
    return full_vector.astype(np.float32)

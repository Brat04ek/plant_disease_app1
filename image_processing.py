import cv2
import numpy as np
from skimage.feature import local_binary_pattern


IMAGE_SIZE = (256, 256)


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Попередня обробка зображення.

    Повертає:
    - resized: кольорове зображення BGR після зміни розміру;
    - gray: зображення у відтінках сірого для ORB/LBP.
    """

    resized = cv2.resize(image, IMAGE_SIZE)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    return resized, gray


def check_image_quality(gray_image: np.ndarray, blur_threshold: float = 35.0) -> bool:
    """
    Перевірка розмитості зображення через дисперсію лапласіана.
    Чим менше значення, тим більш розмите зображення.
    """

    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return blur_score >= blur_threshold


def extract_orb_descriptors(image: np.ndarray) -> tuple[list, np.ndarray | None, np.ndarray, np.ndarray]:
    """
    Виділення ключових точок та дескрипторів ORB.

    ORB використовує FAST для пошуку ключових точок
    та BRIEF-подібний дескриптор для опису їх околу.
    """

    resized, gray = preprocess_image(image)

    orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        fastThreshold=15
    )

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors, resized, gray


def descriptors_to_bovw(descriptors: np.ndarray | None, kmeans_model) -> np.ndarray:
    """
    Перетворення ORB-дескрипторів у вектор фіксованої довжини
    методом Bag of Visual Words.
    """

    number_of_clusters = kmeans_model.n_clusters
    histogram = np.zeros(number_of_clusters, dtype=np.float32)

    if descriptors is None or len(descriptors) == 0:
        return histogram

    words = kmeans_model.predict(descriptors)

    for word in words:
        histogram[word] += 1

    histogram = histogram / (np.linalg.norm(histogram) + 1e-7)

    return histogram


def extract_hsv_histogram(image: np.ndarray) -> np.ndarray:
    """
    Формування HSV-гістограми.

    HSV дозволяє краще описувати колірні симптоми:
    пожовтіння, коричневі плями, темні уражені ділянки.
    """

    resized = cv2.resize(image, IMAGE_SIZE)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = hist.astype(np.float32)
    hist = hist / (np.linalg.norm(hist) + 1e-7)

    return hist


def extract_color_statistics(image: np.ndarray) -> np.ndarray:
    """
    Статистичні колірні ознаки в HSV:
    середнє та стандартне відхилення каналів H, S, V.
    """

    resized = cv2.resize(image, IMAGE_SIZE)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    means = hsv.mean(axis=(0, 1))
    stds = hsv.std(axis=(0, 1))

    features = np.concatenate([means, stds]).astype(np.float32)

    return features


def extract_lbp_features(gray_image: np.ndarray) -> np.ndarray:
    """
    Формування LBP-ознак для опису текстури поверхні листка.

    LBP корисний для аналізу плям, нальоту, некрозу
    та неоднорідних текстур.
    """

    radius = 1
    points = 8 * radius

    lbp = local_binary_pattern(
        gray_image,
        P=points,
        R=radius,
        method="uniform"
    )

    n_bins = points + 2

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_bins + 1),
        range=(0, n_bins)
    )

    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-7)

    return hist


def extract_disease_area_features(image: np.ndarray) -> np.ndarray:
    """
    Приблизне виділення потенційно уражених ділянок за кольором.

    Ознака не є точним діагнозом, але допомагає оцінити,
    яку частку зображення займають жовто-коричневі або темні області.
    """

    resized = cv2.resize(image, IMAGE_SIZE)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Жовто-коричневі ділянки, які часто відповідають плямам/ураженню
    lower_yellow_brown = np.array([10, 35, 30])
    upper_yellow_brown = np.array([45, 255, 255])
    mask_yellow_brown = cv2.inRange(hsv, lower_yellow_brown, upper_yellow_brown)

    # Темні ділянки, які можуть відповідати некрозу
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    disease_mask = cv2.bitwise_or(mask_yellow_brown, mask_dark)

    disease_area_ratio = np.sum(disease_mask > 0) / disease_mask.size

    # Додатково рахуємо середні HSV-значення в потенційно ураженій області
    if np.sum(disease_mask > 0) > 0:
        affected_pixels = hsv[disease_mask > 0]
        affected_mean = affected_pixels.mean(axis=0)
        affected_std = affected_pixels.std(axis=0)
    else:
        affected_mean = np.zeros(3)
        affected_std = np.zeros(3)

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

    _, descriptors, resized, gray = extract_orb_descriptors(image)

    bovw_features = descriptors_to_bovw(descriptors, kmeans_model)
    hsv_histogram = extract_hsv_histogram(resized)
    color_statistics = extract_color_statistics(resized)
    lbp_features = extract_lbp_features(gray)
    disease_area_features = extract_disease_area_features(resized)

    full_vector = np.concatenate([
        bovw_features,
        hsv_histogram,
        color_statistics,
        lbp_features,
        disease_area_features
    ])

    return full_vector.astype(np.float32)
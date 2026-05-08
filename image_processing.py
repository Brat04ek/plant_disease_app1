import cv2
import numpy as np

IMAGE_SIZE = (256, 256)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Попередня обробка зображення:
    - зміна розміру;
    - перетворення у відтінки сірого;
    - фільтрація шуму;
    - вирівнювання контрасту.
    """

    image = cv2.resize(image, IMAGE_SIZE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gray = cv2.equalizeHist(gray)

    return gray


def check_image_quality(gray_image: np.ndarray) -> bool:
    """
    Перевірка якості зображення.
    Використовується оцінка розмитості через дисперсію лапласіана.
    """

    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    if blur_score < 40:
        return False

    return True


def extract_orb_descriptors(image: np.ndarray):
    """
    Виділення ключових точок та дескрипторів.
    ORB використовує FAST для пошуку ключових точок
    і BRIEF-подібний дескриптор для опису локальної області.
    """

    gray = preprocess_image(image)

    orb = cv2.ORB_create(
        nfeatures=700,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        fastThreshold=20
    )

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors, gray


def descriptors_to_bovw(descriptors: np.ndarray, kmeans_model) -> np.ndarray:
    """
    Перетворення набору дескрипторів у вектор фіксованої довжини
    за допомогою Bag of Visual Words.
    """

    number_of_clusters = kmeans_model.n_clusters
    histogram = np.zeros(number_of_clusters)

    if descriptors is None or len(descriptors) == 0:
        return histogram.reshape(1, -1)

    words = kmeans_model.predict(descriptors)

    for word in words:
        histogram[word] += 1

    histogram = histogram / (np.linalg.norm(histogram) + 1e-7)

    return histogram.reshape(1, -1)
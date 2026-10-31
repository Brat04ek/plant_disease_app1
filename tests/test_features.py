"""Check shared numerical feature contracts without trained model files."""
import numpy as np
from src.plant_diagnosis.image_processing import extract_full_feature_vector


class Vocabulary:
    n_clusters = 3

    def predict(self, descriptors):
        return np.zeros(len(descriptors), dtype=int)


def test_feature_vector_shape_and_finiteness():
    image = np.random.default_rng(7).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    vector = extract_full_feature_vector(image, Vocabulary())
    assert vector.shape == (3 + 119,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_no_keypoints_still_produces_features():
    vector = extract_full_feature_vector(np.zeros((256, 256, 3), dtype=np.uint8), Vocabulary())
    assert vector.shape == (122,)
    assert np.array_equal(vector[:3], np.zeros(3))
    assert np.isfinite(vector).all()

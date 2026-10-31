# Plant Disease Diagnosis

A Python computer vision prototype for classifying leaf images, with a FastAPI web interface and a classical machine learning training pipeline. The interface and recommendation text are in Ukrainian.

## Project status

The repository contains application and training source code. **Trained artifacts, the dataset, and experiment reports are not included.** A fresh clone cannot serve predictions until compatible artifacts are supplied or trained. No accuracy or F1 result is independently verifiable from the checked-in files.

This is an educational prototype for initial assessment, not a validated agricultural diagnostic service. Predictions do not replace an agronomist's advice.

## What the code implements

- Image resizing to 256 × 256, grayscale preprocessing, and a Laplacian-based blur check.
- A hybrid feature vector: ORB descriptors encoded with a MiniBatchKMeans Bag of Visual Words vocabulary, HSV histograms, HSV mean/standard deviation, uniform LBP, and heuristic color-mask features.
- A soft-voting ensemble of Random Forest, Extra Trees, and an RBF SVM with feature scaling inside its pipeline.
- Stratified holdout evaluation and cross-validation for configuration selection.
- Image upload, top-three predictions, a low-confidence status below 0.55, and recommendation lookup.

The feature vector has `K + 119` values, where `K` is the vocabulary size. Color-mask features describe regions of the whole image; they are not validated leaf segmentation or a measurement of disease severity. Model probabilities are not established as calibrated diagnostic confidence.

## Repository layout

```text
app.py                              FastAPI routes and inference
src/plant_diagnosis/
  image_processing.py               Shared preprocessing and features
  recommendations.py                Ukrainian class names and recommendations
scripts/
  train_model.py                    Cross-validation, final training, evaluation
  compare_models.py                 Model comparison on the holdout
  dataset_analysis.py               Dataset distribution plots
  plot_results.py                   Experiment plots and ROC curves
  fast_scheme.py                    Diagram generation
  feature_extraction_pipeline.py    Illustrative diagram generation
static/                             Browser JavaScript and CSS
templates/index.html                Web interface
tests/                              API tests with substituted model artifacts
```

## Local setup

Python 3.10 or later is required by the source syntax. Dependency versions are currently unpinned; compatibility with pre-existing serialized models must be verified in their original training environment.

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

### Required model artifacts

Place these three mutually compatible files in `models/`:

```text
models/hybrid_bovw_kmeans.joblib
models/hybrid_classifier.joblib
models/hybrid_label_encoder.joblib
```

The application loads them at import time and fails with an explanatory error if they are absent. Only load trusted artifacts: joblib deserialization can execute code. There is currently no published artifact download or checksum manifest in this repository.

### Train from a local dataset

The training script expects class directories:

```text
dataset/PlantVillage/<class_name>/<image.jpg or image.jpeg or image.png>
```

The exact dataset source, version, license, and file manifest are not documented in the original repository. Obtain and verify those before reproducing an experiment; a similarly named dataset is not sufficient to establish identical results. Recommendation lookup contains 15 class keys for pepper, potato, and tomato; training uses the folders actually present, rather than enforcing those 15 classes.

```bash
python scripts/train_model.py
```

The script selects at most the first 500 sorted image filenames per class, uses a stratified 20% test holdout with seed 7, and evaluates five configurations with eight-fold stratified cross-validation on the remaining 80%. Each fold fits its own visual vocabulary on that fold's training images. The configuration with the highest mean validation macro F1 is retrained on the train/validation partition and evaluated on the holdout.

Training writes the three artifacts to `models/` and reports to `results/train/`, including `run_config.json`, `best_params.json`, `cv_results_summary.csv`, `final_test_metrics_summary.csv`, and `final_test_predictions.csv`. These directories are ignored by Git. Training can be computationally expensive; no measured runtime or hardware requirement is available. Sufficient valid images per class and enough ORB descriptors are required.

### Start the application

After installing dependencies and providing the three artifacts:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. API documentation is available at `/docs`.

| Route | Purpose |
| --- | --- |
| `GET /` | Upload interface |
| `GET /health` | Process health; not an inference or model-quality check |
| `POST /predict` | Multipart upload with field `file` |

```bash
curl -F "file=@/path/to/leaf.jpg" http://127.0.0.1:8000/predict
```

Prediction statuses are `success`, `uncertain`, and `bad_quality`. Invalid or empty image content returns HTTP 400; uploads over 10 MiB return HTTP 413. The route bounds its byte read, but this is not a transport-level or decoded-image memory limit. Public deployment needs additional resource controls.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

API checks substitute model loading. They cover page/static serving, health, invalid and empty uploads, the upload-size limit, blur rejection, missing-artifact diagnostics, success/uncertain response contracts with substituted inference, and finite feature-vector shape checks. They do not validate model accuracy or successful real-artifact inference.

## Evaluation and reproducibility limits

The code provides an evaluation procedure, not verified published results. Reproducing an experiment also requires a dataset manifest, saved split identities, artifact hashes, dependency versions, hardware details, and the resulting reports. A fixed seed alone does not establish reproducibility.

Near-duplicate images or related leaves across splits have not been ruled out. Performance on a curated dataset does not establish performance on field photographs. The 0.55 confidence threshold and blur threshold are heuristics without supplied validation. `compare_models.py` hard-codes configuration values; they may differ from a new training run's selected configuration. Diagram scripts contain illustrative elements and should not be treated as measured results or an exact feature specification.

See [the technical review and prioritized fixes](docs/TECHNICAL_REVIEW_UK.md).

## License

No project license was present in the reviewed source. Dataset and third-party artifact rights must be checked separately before redistribution.

### Review validation

The local review ran on Python 3.14.8: 10 tests passed and `pip check` reported no broken requirements. The installed package snapshot is in [validation-environment.txt](docs/validation-environment.txt); it is not the original model training environment or a cross-platform lockfile. Starlette emitted a TestClient deprecation warning concerning httpx. Real-artifact inference and full training were not run because the required model files and data are absent.

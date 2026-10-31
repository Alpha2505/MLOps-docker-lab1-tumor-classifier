# Docker Lab 1 — Containerized Tumor Classifier with Model Selection

This project packages a small machine-learning training job in a Docker container. The container trains three candidate models on the **Breast Cancer Wisconsin (Diagnostic)** dataset, selects the best one with stratified cross-validation, evaluates it on a held-out test set, and writes the trained model and a JSON metrics report to a mounted volume.

## What I changed from the course lab

The course's Docker Lab 1 trains a single Random Forest on the Iris dataset and saves it inside the container. This version keeps the same idea (train a model inside a container) but changes the data, the logic, and the container setup:

| Area | Course lab | This project |
|---|---|---|
| Dataset | Iris (150 rows, 3 classes) | Breast Cancer Wisconsin (569 rows, 30 features, binary) |
| Modeling | One fixed Random Forest | Logistic Regression, RBF SVM and Gradient Boosting compared with stratified k-fold CV; best one is selected automatically |
| Preprocessing | None | `StandardScaler` inside a scikit-learn `Pipeline`, so scaling is fit only on training folds and saved with the model |
| Evaluation | None (only a "success" message) | Accuracy, precision, recall, F1, ROC-AUC, malignant-class recall and confusion matrix on a held-out set |
| Output | `iris_model.pkl` left inside the container | `tumor_classifier.joblib` and `metrics.json` written to a volume you can mount on the host |
| Configuration | Hard-coded | Environment variables (`TEST_SIZE`, `RANDOM_SEED`, `CV_FOLDS`, `SELECTION_METRIC`, `OUTPUT_DIR`) with validation |
| Inference | None | `predict.py` loads the saved model and classifies sample rows |
| Code structure | One script | Separate modules for config, data, models, training and prediction |
| Tests | None | pytest suite covering config validation, data splitting, model selection, a quality bar and the saved model |
| Dockerfile | `python:3.10`, root user, unpinned deps | `python:3.10-slim`, non-root user, pinned dependencies, layer caching, `.dockerignore`, volume for artifacts |

I report **malignant recall** separately because in this dataset label 0 means malignant, and a missed malignant tumor is the most costly mistake. The standard `recall` score only describes the benign class.

## Project structure

```
.
├── Dockerfile
├── .dockerignore
├── requirements.txt        # runtime dependencies (pinned)
├── requirements-dev.txt    # adds pytest
├── src/
│   ├── config.py           # env-var configuration + validation
│   ├── data.py             # load + stratified split
│   ├── models.py           # candidate pipelines
│   ├── train.py            # CV model selection, evaluation, saving
│   └── predict.py          # load saved model and predict
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_data.py
    └── test_train.py
```

## Running with Docker

Build the image:

```bash
docker build -t lab1-tumor-classifier:v1 .
```

Run training and copy the model and metrics back to your machine:

```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" lab1-tumor-classifier:v1
```

After it finishes, `artifacts/` contains `tumor_classifier.joblib` and `metrics.json`.

Override settings with environment variables, for example 10-fold CV selected on F1:

```bash
docker run --rm -e CV_FOLDS=10 -e SELECTION_METRIC=f1 \
  -v "$(pwd)/artifacts:/app/artifacts" lab1-tumor-classifier:v1
```

Run the prediction script against the saved model:

```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" lab1-tumor-classifier:v1 python predict.py 5
```

Save the image as a tarball:

```bash
docker save lab1-tumor-classifier:v1 > lab1-tumor-classifier.tar
```

## Configuration

| Variable | Default | Allowed values |
|---|---|---|
| `TEST_SIZE` | `0.2` | 0.05 – 0.5 |
| `RANDOM_SEED` | `42` | any integer |
| `CV_FOLDS` | `5` | ≥ 2 |
| `SELECTION_METRIC` | `roc_auc` | `roc_auc`, `f1`, `accuracy`, `recall`, `precision` |
| `OUTPUT_DIR` | `/app/artifacts` in Docker, `artifacts` locally | any path |

Invalid values stop the run with a clear error message.

## Running locally and running the tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
python src/train.py
```

The tests check that:

- invalid configuration values are rejected and environment overrides are read;
- the dataset loads with the expected shape and the split is stratified and reproducible;
- model selection picks the candidate with the highest CV score;
- training writes both artifacts and evaluates all three candidates;
- the selected model reaches ROC-AUC ≥ 0.95 and malignant recall ≥ 0.90 on the held-out set;
- the saved model loads and returns valid labels with confidence scores.

## Sample results (default settings)

```
Cross-validation (5-fold, metric = roc_auc):
  logistic_regression    0.9959 ± 0.0050
  svm_rbf                0.9956 ± 0.0048
  gradient_boosting      0.9920 ± 0.0052

Selected model: logistic_regression
Held-out test metrics:
  accuracy          0.9825
  precision         0.9861
  recall            0.9861
  f1                0.9861
  roc_auc           0.9954
  malignant_recall  0.9762
  confusion         TN=41 FP=1 FN=1 TP=71
```

Logistic regression edges out the other two models on this dataset, which suggests the classes are close to linearly separable once the features are standardized.

## Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset ships with scikit-learn (`sklearn.datasets.load_breast_cancer`), so no download is needed at build or run time. It has 569 samples with 30 numeric features computed from images of fine-needle aspirates of breast masses, labeled malignant (212) or benign (357).

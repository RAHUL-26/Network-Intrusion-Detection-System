"""Model training, comparison, and artifact saving."""

import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from src.config import MODELS, MODEL_DIR, RANDOM_STATE, BEST_MODEL_NAME


def _build_model(name: str, params: dict):
    """Instantiate a sklearn model by name."""
    constructors = {
        "Random Forest": lambda p: RandomForestClassifier(
            max_depth=p.get("max_depth", None), random_state=RANDOM_STATE
        ),
        "Decision Tree": lambda p: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": lambda p: KNeighborsClassifier(),
        "SVM": lambda p: SVC(probability=True, random_state=RANDOM_STATE),
        "AdaBoost": lambda p: AdaBoostClassifier(
            n_estimators=p.get("n_estimators", 50),
            learning_rate=p.get("learning_rate", 1),
            random_state=RANDOM_STATE,
        ),
    }
    return constructors[name](params)


def train_single_model(name, params, X_train, y_train, X_test, y_test, cv=5):
    """Train one model, evaluate, and return results dict."""
    print(f"\n{'─' * 50}")
    print(f"Training: {name}")
    print(f"{'─' * 50}")

    model = _build_model(name, params)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=cv)

    # Metrics
    train_acc = metrics.accuracy_score(y_train, y_pred_train)
    test_acc = metrics.accuracy_score(y_test, y_pred_test)
    cm = metrics.confusion_matrix(y_test, y_pred_test)
    report = metrics.classification_report(y_test, y_pred_test, output_dict=True)

    print(f"  Train Accuracy:  {train_acc:.4f}")
    print(f"  Test Accuracy:   {test_acc:.4f}")
    print(f"  CV Mean Score:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Training Time:   {train_time:.2f}s")

    return {
        "name": name,
        "model": model,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "confusion_matrix": cm,
        "classification_report": report,
        "train_time": train_time,
        "y_pred_test": y_pred_test,
    }


def train_all_models(X_train, y_train, X_test, y_test):
    """Train all configured models and return results."""
    results = {}
    for name, params in MODELS.items():
        results[name] = train_single_model(
            name, params, X_train, y_train, X_test, y_test
        )
    return results


def save_best_model(results: dict, feature_names: list, class_names: list,
                    scaler=None, imputer=None):
    """Save the best model + preprocessing artifacts to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    best = results.get(BEST_MODEL_NAME, max(results.values(), key=lambda r: r["test_accuracy"]))
    model = best["model"]

    artifacts = {
        "model": model,
        "feature_names": feature_names,
        "class_names": class_names,
        "model_name": best["name"],
        "test_accuracy": best["test_accuracy"],
    }
    if scaler is not None:
        artifacts["scaler"] = scaler
    if imputer is not None:
        artifacts["imputer"] = imputer

    path = os.path.join(MODEL_DIR, "nids_model.joblib")
    joblib.dump(artifacts, path)
    print(f"\nSaved best model ({best['name']}, acc={best['test_accuracy']:.4f}) to {path}")
    return path


def print_comparison_table(results: dict):
    """Print a formatted comparison of all models."""
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Train Acc':>10} {'Test Acc':>10} {'CV Mean':>10} {'Time (s)':>10}")
    print("=" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1]["test_accuracy"], reverse=True):
        print(f"{name:<20} {r['train_accuracy']:>10.4f} {r['test_accuracy']:>10.4f} "
              f"{r['cv_mean']:>10.4f} {r['train_time']:>10.2f}")
    print("=" * 70)

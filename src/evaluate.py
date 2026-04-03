"""Evaluation metrics and visualization generation."""

import os
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PLOTS_DIR


def plot_confusion_matrix(cm, title, class_names=None, save=True):
    """Plot and optionally save a confusion matrix heatmap."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, f"cm_{title.lower().replace(' ', '_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_accuracy_comparison(results: dict, save=True):
    """Bar chart comparing test accuracy across all models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    names = list(results.keys())
    train_accs = [results[n]["train_accuracy"] for n in names]
    test_accs = [results[n]["test_accuracy"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, train_accs, width, label="Train", color="#2196F3")
    bars2 = ax.bar(x + width / 2, test_accs, width, label="Test", color="#FF9800")

    ax.set_xlabel("Classifier")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(0.5, 1.05)

    for bar in bars1 + bars2:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "accuracy_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_training_time(results: dict, save=True):
    """Bar chart of training times."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    names = list(results.keys())
    times = [results[n]["train_time"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, times, color="#4CAF50")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Model Training Time Comparison")

    for bar in bars:
        ax.annotate(f"{bar.get_height():.1f}s",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "training_time.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_cv_scores(results: dict, save=True):
    """Box-style comparison of cross-validation scores."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    names = list(results.keys())
    means = [results[n]["cv_mean"] for n in names]
    stds = [results[n]["cv_std"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, means, yerr=stds, capsize=5, color="#9C27B0", alpha=0.8)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("CV Score (Mean ± Std)")
    ax.set_title("Cross-Validation Score Comparison")
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "cv_scores.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def generate_all_plots(results: dict, class_names: list = None):
    """Generate all evaluation plots."""
    print("\nGenerating plots...")

    # Confusion matrices for each model
    for name, r in results.items():
        plot_confusion_matrix(r["confusion_matrix"], name, class_names)

    # Comparison charts
    plot_accuracy_comparison(results)
    plot_training_time(results)
    plot_cv_scores(results)

    print(f"\nAll plots saved to {PLOTS_DIR}/")

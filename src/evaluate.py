import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from src.config import (
    CLASSES,
    NUM_CLASSES,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    SVM_MODEL_PATH,
    CNN_MODEL_PATH,
    MOBILENET_MODEL_PATH,
    DATA_SPLIT_DIR,
    DATASET_DIR,
)
from src.data_loader import load_svm_data, load_test_arrays


def plot_dataset_distribution():
    records = []
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            folder = DATA_SPLIT_DIR / split / c
            n = len(list(folder.glob("*.jpg")))
            records.append({"Split": split.capitalize(), "Class": c.capitalize(), "Count": n})
    df = pd.DataFrame(records)

    plt.figure(figsize=(9, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    palette = {"Train": "#3498db", "Val": "#e67e22", "Test": "#2ecc71"}
    ax = sns.barplot(data=df, x="Class", y="Count", hue="Split", palette=palette)
    plt.title("Rock-Paper-Scissors Dataset Split Distribution", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Gesture Class", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Images", fontsize=12, fontweight="bold")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                        textcoords="offset points")

    plt.tight_layout()
    out_path = FIGURES_DIR / "dataset_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f" Saved dataset distribution plot to {out_path}")


def plot_sample_images():
    import cv2
    fig, axes = plt.subplots(3, 6, figsize=(14, 7), dpi=300)
    for row, c in enumerate(CLASSES):
        folder = DATASET_DIR / c
        all_files = sorted(list(folder.glob("*.jpg")))
        step = max(1, len(all_files) // 6)
        files = all_files[::step][:6]
        for col, f in enumerate(files):
            ax = axes[row, col]
            img = cv2.imread(str(f))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_title(f"{c.capitalize()}", fontsize=12, fontweight="bold", loc="left")

    plt.suptitle("Sample Captured Hand Gesture Images (After MediaPipe Crop)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = FIGURES_DIR / "sample_dataset_images.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f" Saved sample images plot to {out_path}")


def plot_learning_curves():
    hist_file = MODELS_DIR / "training_histories.json"
    if not hist_file.exists():
        print(f" {hist_file} not found; skipping learning curves plot.")
        return

    with open(hist_file, "r") as f:
        histories = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

    if "cnn" in histories and histories["cnn"]:
        cnn_h = histories["cnn"]
        epochs = range(1, len(cnn_h.get("accuracy", [])) + 1)
        axes[0, 0].plot(epochs, cnn_h.get("accuracy", []), "o-", label="Train Accuracy", color="#2980b9", lw=2)
        axes[0, 0].plot(epochs, cnn_h.get("val_accuracy", []), "s--", label="Val Accuracy", color="#e74c3c", lw=2)
        axes[0, 0].set_title("Custom CNN: Accuracy vs Epochs", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle="--", alpha=0.6)

        axes[1, 0].plot(epochs, cnn_h.get("loss", []), "o-", label="Train Loss", color="#2980b9", lw=2)
        axes[1, 0].plot(epochs, cnn_h.get("val_loss", []), "s--", label="Val Loss", color="#e74c3c", lw=2)
        axes[1, 0].set_title("Custom CNN: Cross-Entropy Loss vs Epochs", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    if "mobilenet" in histories and histories["mobilenet"]:
        mb_h = histories["mobilenet"]
        epochs_mb = range(1, len(mb_h.get("accuracy", [])) + 1)
        p1_end = mb_h.get("phase1_epochs", 12)

        axes[0, 1].plot(epochs_mb, mb_h.get("accuracy", []), "o-", label="Train Accuracy", color="#27ae60", lw=2)
        axes[0, 1].plot(epochs_mb, mb_h.get("val_accuracy", []), "s--", label="Val Accuracy", color="#d35400", lw=2)
        axes[0, 1].axvline(x=p1_end, color="#7f8c8d", linestyle=":", lw=2, label="Fine-tuning Phase Start")
        axes[0, 1].set_title("MobileNetV2: Accuracy vs Epochs (Two-Phase)", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle="--", alpha=0.6)

        axes[1, 1].plot(epochs_mb, mb_h.get("loss", []), "o-", label="Train Loss", color="#27ae60", lw=2)
        axes[1, 1].plot(epochs_mb, mb_h.get("val_loss", []), "s--", label="Val Loss", color="#d35400", lw=2)
        axes[1, 1].axvline(x=p1_end, color="#7f8c8d", linestyle=":", lw=2, label="Fine-tuning Phase Start")
        axes[1, 1].set_title("MobileNetV2: Cross-Entropy Loss vs Epochs", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = FIGURES_DIR / "learning_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f" Saved learning curves plot to {out_path}")


def evaluate_models():
    plot_dataset_distribution()
    plot_sample_images()
    plot_learning_curves()

    results = {}
    y_preds = {}
    cm_dict = {}

    print("\n--- Evaluating SVM Baseline ---")
    if SVM_MODEL_PATH.exists():
        svm = joblib.load(SVM_MODEL_PATH)
        X_test_svm, y_test_svm = load_svm_data("test")

        t_start = time.time()
        for _ in range(50):
            _ = svm.predict(X_test_svm[:1])
        lat_svm_ms = (time.time() - t_start) / 50.0 * 1000

        pred_svm = svm.predict(X_test_svm)
        acc_svm = accuracy_score(y_test_svm, pred_svm)
        p, r, f1, _ = precision_recall_fscore_support(y_test_svm, pred_svm, average="macro")
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test_svm, pred_svm, average="weighted")
        cm_svm = confusion_matrix(y_test_svm, pred_svm)

        results["SVM Baseline (RBF)"] = {
            "Accuracy": float(acc_svm),
            "Macro Precision": float(p),
            "Macro Recall": float(r),
            "Macro F1": float(f1),
            "Weighted F1": float(f1_w),
            "Latency (ms/frame)": float(lat_svm_ms),
            "FPS": float(1000 / lat_svm_ms) if lat_svm_ms > 0 else 0,
            "Model Size (MB)": round(SVM_MODEL_PATH.stat().st_size / (1024 * 1024), 2),
            "Parameters": "N/A (Kernel)",
        }
        y_preds["SVM"] = (y_test_svm, pred_svm)
        cm_dict["SVM Baseline"] = cm_svm
        print(f"SVM Accuracy: {acc_svm:.4f} | Macro F1: {f1:.4f} | Latency: {lat_svm_ms:.2f}ms")
    else:
        print(f" {SVM_MODEL_PATH} not found.")

    print("\n--- Evaluating Custom CNN ---")
    if CNN_MODEL_PATH.exists():
        cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
        X_test_cnn, y_test_cnn = load_test_arrays(model_type="cnn")

        t_start = time.time()
        for _ in range(50):
            _ = cnn.predict(X_test_cnn[:1], verbose=0)
        lat_cnn_ms = (time.time() - t_start) / 50.0 * 1000

        probs_cnn = cnn.predict(X_test_cnn, verbose=0)
        pred_cnn = np.argmax(probs_cnn, axis=1)
        acc_cnn = accuracy_score(y_test_cnn, pred_cnn)
        p, r, f1, _ = precision_recall_fscore_support(y_test_cnn, pred_cnn, average="macro")
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test_cnn, pred_cnn, average="weighted")
        cm_cnn = confusion_matrix(y_test_cnn, pred_cnn)

        results["Custom Deep CNN"] = {
            "Accuracy": float(acc_cnn),
            "Macro Precision": float(p),
            "Macro Recall": float(r),
            "Macro F1": float(f1),
            "Weighted F1": float(f1_w),
            "Latency (ms/frame)": float(lat_cnn_ms),
            "FPS": float(1000 / lat_cnn_ms) if lat_cnn_ms > 0 else 0,
            "Model Size (MB)": round(CNN_MODEL_PATH.stat().st_size / (1024 * 1024), 2),
            "Parameters": int(cnn.count_params()),
        }
        y_preds["Custom CNN"] = (y_test_cnn, pred_cnn)
        cm_dict["Custom Deep CNN"] = cm_cnn
        print(f"Custom CNN Accuracy: {acc_cnn:.4f} | Macro F1: {f1:.4f} | Latency: {lat_cnn_ms:.2f}ms")
    else:
        print(f" {CNN_MODEL_PATH} not found.")

    print("\n--- Evaluating MobileNetV2 Transfer Learning ---")
    if MOBILENET_MODEL_PATH.exists():
        mobilenet = tf.keras.models.load_model(MOBILENET_MODEL_PATH)
        X_test_mb, y_test_mb = load_test_arrays(model_type="mobilenet")

        t_start = time.time()
        for _ in range(50):
            _ = mobilenet.predict(X_test_mb[:1], verbose=0)
        lat_mb_ms = (time.time() - t_start) / 50.0 * 1000

        probs_mb = mobilenet.predict(X_test_mb, verbose=0)
        pred_mb = np.argmax(probs_mb, axis=1)
        acc_mb = accuracy_score(y_test_mb, pred_mb)
        p, r, f1, _ = precision_recall_fscore_support(y_test_mb, pred_mb, average="macro")
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test_mb, pred_mb, average="weighted")
        cm_mb = confusion_matrix(y_test_mb, pred_mb)

        results["MobileNetV2 (Transfer Learning)"] = {
            "Accuracy": float(acc_mb),
            "Macro Precision": float(p),
            "Macro Recall": float(r),
            "Macro F1": float(f1),
            "Weighted F1": float(f1_w),
            "Latency (ms/frame)": float(lat_mb_ms),
            "FPS": float(1000 / lat_mb_ms) if lat_mb_ms > 0 else 0,
            "Model Size (MB)": round(MOBILENET_MODEL_PATH.stat().st_size / (1024 * 1024), 2),
            "Parameters": int(mobilenet.count_params()),
        }
        y_preds["MobileNetV2"] = (y_test_mb, pred_mb)
        cm_dict["MobileNetV2 (Fine-Tuned)"] = cm_mb
        print(f"MobileNetV2 Accuracy: {acc_mb:.4f} | Macro F1: {f1:.4f} | Latency: {lat_mb_ms:.2f}ms")
    else:
        print(f" {MOBILENET_MODEL_PATH} not found.")

    if len(cm_dict) > 0:
        fig, axes = plt.subplots(1, len(cm_dict), figsize=(5.5 * len(cm_dict), 4.5), dpi=300)
        if len(cm_dict) == 1:
            axes = [axes]

        class_labels = [c.capitalize() for c in CLASSES]
        for ax, (m_title, cm) in zip(axes, cm_dict.items()):
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar=False,
                ax=ax,
                annot_kws={"size": 14, "weight": "bold"},
            )
            ax.set_title(m_title, fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
            ax.set_ylabel("True Label", fontsize=11, fontweight="bold")

        plt.suptitle("Test Set Confusion Matrix Comparison", fontsize=15, fontweight="bold", y=1.03)
        plt.tight_layout()
        cm_all_path = FIGURES_DIR / "confusion_matrices_comparison.png"
        plt.savefig(cm_all_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f" Saved combined confusion matrices to {cm_all_path}")

        for m_title, cm in cm_dict.items():
            plt.figure(figsize=(5, 4), dpi=300)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar=True,
                annot_kws={"size": 12, "weight": "bold"},
            )
            plt.title(f"Confusion Matrix: {m_title}", fontsize=12, fontweight="bold")
            plt.xlabel("Predicted Label", fontsize=10, fontweight="bold")
            plt.ylabel("True Label", fontsize=10, fontweight="bold")
            plt.tight_layout()
            slug = m_title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            plt.savefig(FIGURES_DIR / f"confusion_matrix_{slug}.png", dpi=300)
            plt.close()

    if len(results) > 0:
        df_res = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

        sns.barplot(data=df_res, x="Model", y="Accuracy", hue="Model", ax=axes[0], palette="viridis", legend=False)
        axes[0].set_title("Test Accuracy", fontsize=12, fontweight="bold")
        axes[0].set_ylim(0, 1.05)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                             ha="center", va="bottom", fontsize=11, xytext=(0, 3), textcoords="offset points")

        sns.barplot(data=df_res, x="Model", y="Macro F1", hue="Model", ax=axes[1], palette="magma", legend=False)
        axes[1].set_title("Test Macro F1-Score", fontsize=12, fontweight="bold")
        axes[1].set_ylim(0, 1.05)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                             ha="center", va="bottom", fontsize=11, xytext=(0, 3), textcoords="offset points")

        sns.barplot(data=df_res, x="Model", y="Latency (ms/frame)", hue="Model", ax=axes[2], palette="crest", legend=False)
        axes[2].set_title("Inference Latency (ms / frame, lower is better)", fontsize=12, fontweight="bold")
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():.1f}ms", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                             ha="center", va="bottom", fontsize=11, xytext=(0, 3), textcoords="offset points")

        for ax in axes:
            ax.set_xticks(range(len(df_res)))
            ax.set_xticklabels(df_res["Model"], rotation=15, ha="right", fontsize=9, fontweight="bold")
            ax.set_xlabel("")

        plt.suptitle("Model Benchmark Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        metrics_plot_path = FIGURES_DIR / "metrics_comparison_bar.png"
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        print(f" Saved metrics comparison bar plot to {metrics_plot_path}")

        csv_path = REPORTS_DIR / "metrics_summary.csv"
        df_res.to_csv(csv_path, index=False)
        print(f" Saved metrics CSV to {csv_path}")

        json_path = REPORTS_DIR / "metrics_summary.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f" Saved metrics JSON to {json_path}")

        print(df_res.to_string(index=False))

    return results


if __name__ == "__main__":
    evaluate_models()

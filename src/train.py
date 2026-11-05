import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import joblib
import numpy as np
import tensorflow as tf
from src.config import (
    CLASSES,
    NUM_CLASSES,
    MODELS_DIR,
    SVM_MODEL_PATH,
    CNN_MODEL_PATH,
    MOBILENET_MODEL_PATH,
    CLASSES_MAP_PATH,
    EPOCHS_CNN,
    EPOCHS_MOBILENET_P1,
    EPOCHS_MOBILENET_P2,
)
from src.data_loader import load_svm_data, get_tf_datasets
from src.models import (
    build_svm_model,
    build_custom_cnn,
    build_mobilenet_v2,
    unfreeze_mobilenet_for_finetuning,
)


def save_classes_mapping():
    mapping = {str(i): name for i, name in enumerate(CLASSES)}
    with open(CLASSES_MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f" Saved class mappings to {CLASSES_MAP_PATH}")


def train_svm():
    print("\n" + "=" * 50)
    print(" Training Baseline SVM Model...")
    print("=" * 50)
    t0 = time.time()
    X_train, y_train = load_svm_data("train")
    X_val, y_val = load_svm_data("val")
    print(f"Data loaded: Train {X_train.shape}, Val {X_val.shape}")

    svm = build_svm_model(C=10.0, kernel="rbf", gamma="scale")
    svm.fit(X_train, y_train)

    train_acc = svm.score(X_train, y_train)
    val_acc = svm.score(X_val, y_val)
    elapsed = time.time() - t0

    print(f"SVM Trained in {elapsed:.2f}s | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    joblib.dump(svm, SVM_MODEL_PATH)
    print(f" Saved SVM model to {SVM_MODEL_PATH}")
    return svm


def train_custom_cnn(epochs=15):
    print("\n" + "=" * 50)
    print(" Training Custom CNN Model (from scratch)...")
    print("=" * 50)
    train_ds, val_ds, test_ds, _ = get_tf_datasets(model_type="cnn")

    with tf.device("/CPU:0"):
        model = build_custom_cnn()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(CNN_MODEL_PATH),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        t0 = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )
        elapsed = time.time() - t0
        print(f"Custom CNN training completed in {elapsed:.2f}s")
        model.save(CNN_MODEL_PATH)
        print(f" Saved Custom CNN model to {CNN_MODEL_PATH}")
        return model, history.history


def train_mobilenet(epochs_p1=EPOCHS_MOBILENET_P1, epochs_p2=EPOCHS_MOBILENET_P2):
    print("\n" + "=" * 50)
    print(" Training MobileNetV2 (Transfer Learning & Fine-Tuning)...")
    print("=" * 50)
    train_ds, val_ds, test_ds, _ = get_tf_datasets(model_type="mobilenet")

    model, base_model = build_mobilenet_v2()

    print("\n--- Phase 1: Feature Extraction (Base Frozen) ---")
    cb_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        )
    ]
    t0 = time.time()
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_p1,
        callbacks=cb_p1,
    )

    print("\n--- Phase 2: Fine-Tuning (Top 35 layers unfrozen, LR=1e-5) ---")
    model = unfreeze_mobilenet_for_finetuning(model, base_model, num_unfrozen_layers=35)
    cb_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MOBILENET_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_p2,
        callbacks=cb_p2,
    )
    elapsed = time.time() - t0
    print(f"MobileNetV2 training completed in {elapsed:.2f}s")

    combined_history = {
        "accuracy": h1.history.get("accuracy", []) + h2.history.get("accuracy", []),
        "val_accuracy": h1.history.get("val_accuracy", []) + h2.history.get("val_accuracy", []),
        "loss": h1.history.get("loss", []) + h2.history.get("loss", []),
        "val_loss": h1.history.get("val_loss", []) + h2.history.get("val_loss", []),
        "phase1_epochs": len(h1.history.get("accuracy", [])),
    }

    model.save(MOBILENET_MODEL_PATH)
    print(f" Saved MobileNetV2 model to {MOBILENET_MODEL_PATH}")
    return model, combined_history


def train_all():
    save_classes_mapping()
    train_svm()
    _, cnn_hist = train_custom_cnn()
    _, mobilenet_hist = train_mobilenet()

    histories = {
        "cnn": cnn_hist,
        "mobilenet": mobilenet_hist,
    }
    history_path = MODELS_DIR / "training_histories.json"
    with open(history_path, "w") as f:
        clean_hist = {}
        for m_name, m_h in histories.items():
            clean_hist[m_name] = {
                k: [float(v) for v in vals] if isinstance(vals, list) else vals
                for k, vals in m_h.items()
            }
        json.dump(clean_hist, f, indent=2)
    print(f" Saved training histories to {history_path}")


if __name__ == "__main__":
    train_all()

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tensorflow as tf
from src.config import (
    DATASET_DIR,
    DATA_SPLIT_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    CLASSES,
    IMG_SIZE,
    IMG_SIZE_SVM,
    BATCH_SIZE,
    RANDOM_SEED,
)


def load_svm_data(split="train", size=IMG_SIZE_SVM):
    folder = DATA_SPLIT_DIR / split
    X, y = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = folder / cls
        for img_path in sorted(cls_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size)
            X.append(img.flatten().astype(np.float32) / 255.0)
            y.append(idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def get_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.06, 0.06),
        tf.keras.layers.RandomContrast(0.12),
    ], name="augmentation_pipeline")


def get_tf_datasets(img_size=IMG_SIZE, batch_size=BATCH_SIZE, model_type="mobilenet"):
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED,
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    if model_type == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        norm_fn = lambda x, y: (preprocess_input(x), y)
    else:
        norm_fn = lambda x, y: (x / 255.0, y)

    train_ds = train_ds_raw.map(norm_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds_raw.map(norm_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds_raw.map(norm_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, train_ds_raw.class_names


def load_test_arrays(img_size=IMG_SIZE, model_type="mobilenet"):
    if model_type == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        norm_fn = lambda img: preprocess_input(img.astype(np.float32))
    else:
        norm_fn = lambda img: (img.astype(np.float32) / 255.0)

    X, y = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = TEST_DIR / cls
        for img_path in sorted(cls_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(norm_fn(img))
            y.append(idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

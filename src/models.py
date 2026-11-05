import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from src.data_loader import get_data_augmentation
from src.config import IMG_SIZE, NUM_CLASSES, LR_INITIAL, LR_FINETUNE


def build_svm_model(C=10.0, kernel="rbf", gamma="scale"):
    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        random_state=42,
    )


def build_custom_cnn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES, lr=5e-4):
    augmentation = get_data_augmentation()

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = augmentation(inputs)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="drop1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Custom_CNN_RPS")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_v2(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES, lr=LR_INITIAL):
    augmentation = get_data_augmentation()

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.35, name="drop_head1")(x)
    x = layers.Dense(128, activation="relu", name="fc_head")(x)
    x = layers.Dropout(0.2, name="drop_head2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer_RPS")
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_mobilenet_for_finetuning(model, base_model, num_unfrozen_layers=35, lr=LR_FINETUNE):
    base_model.trainable = True
    for layer in base_model.layers[:-num_unfrozen_layers]:
        layer.trainable = False
    for layer in base_model.layers[-num_unfrozen_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

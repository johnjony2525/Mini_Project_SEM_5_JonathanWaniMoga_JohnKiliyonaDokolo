import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# ==============================
# CONFIGURATION
# ==============================

DATASET_PATH = r"C:\PneumoniaAI\Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

MODEL_DIR = r"C:\PneumoniaAI\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class names:", train_ds.class_names)

# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
])

# ==============================
# CLASS WEIGHTS
# ==============================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=np.concatenate([
        y.numpy().flatten()
        for _, y in train_ds
    ])
)

class_weights = {
    0: class_weights[0],
    1: class_weights[1]
}

print("Class weights:", class_weights)

# ==============================
# BASE MODEL
# ==============================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained layers
base_model.trainable = False

# ==============================
# BUILD OPTIMIZED MODEL
# ==============================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = layers.Rescaling(1./127.5, offset=-1)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(inputs, outputs)

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# ==============================
# CALLBACKS
# ==============================

callbacks = [

    ModelCheckpoint(
        os.path.join(
            MODEL_DIR,
            "optimized_best.keras"
        ),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ==============================
# TRAINING
# ==============================

print("\n")
print("=" * 50)
print("STARTING OPTIMIZED MODEL TRAINING")
print("=" * 50)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==============================
# TEST EVALUATION
# ==============================

print("\n")
print("=" * 50)
print("OPTIMIZED MODEL TEST EVALUATION")
print("=" * 50)

results = model.evaluate(test_ds)

for name, value in zip(
    model.metrics_names,
    results
):
    print(f"{name}: {value:.4f}")

print("\nTraining completed successfully.")
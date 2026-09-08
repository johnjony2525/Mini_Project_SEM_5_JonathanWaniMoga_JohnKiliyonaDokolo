import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

MODEL_PATH = r"C:\PneumoniaAI\models\transfer_best.keras"
DATASET_PATH = r"C:\PneumoniaAI\Dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class names:", test_ds.class_names)

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

print("\nEvaluating model...\n")

results = model.evaluate(test_ds, return_dict=True)

print("\n===== TEST RESULTS =====")

for name, value in results.items():
    print(f"{name}: {value:.4f}")

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().flatten())
    y_pred.extend((predictions.flatten() >= 0.5).astype(int))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_true, y_pred))

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=test_ds.class_names
    )
)
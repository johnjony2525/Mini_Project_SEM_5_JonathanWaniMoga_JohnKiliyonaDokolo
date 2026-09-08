import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# ==========================================
# MODEL PERFORMANCE RESULTS
# ==========================================

accuracy = 0.8798
auc = 0.9518
precision = 0.9008
recall = 0.9077

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = np.array([
    [195, 39],
    [36, 354]
])

# ==========================================
# GRAPH 1: CONFUSION MATRIX
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

disp.plot()

plt.title("PneumoniaAI - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    r"C:\PneumoniaAI\results\confusion_matrix.png",
    dpi=300
)

plt.show()

# ==========================================
# GRAPH 2: PERFORMANCE METRICS
# ==========================================

metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall"
]

values = [
    accuracy,
    auc,
    precision,
    recall
]

plt.figure(figsize=(8, 6))

bars = plt.bar(metrics, values)

plt.ylim(0, 1.0)
plt.ylabel("Score")
plt.title("PneumoniaAI - Model Performance")

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2%}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    r"C:\PneumoniaAI\results\performance_metrics.png",
    dpi=300
)

plt.show()

print("Graphs created successfully.")
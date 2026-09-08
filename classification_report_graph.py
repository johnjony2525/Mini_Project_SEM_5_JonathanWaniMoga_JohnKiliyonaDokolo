import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CLASSIFICATION REPORT RESULTS
# ==========================================

classes = ["NORMAL", "PNEUMONIA"]

precision = [0.84, 0.90]
recall = [0.83, 0.91]
f1_score = [0.84, 0.90]

x = np.arange(len(classes))
width = 0.25

# ==========================================
# CREATE GRAPH
# ==========================================

plt.figure(figsize=(9, 6))

bars1 = plt.bar(
    x - width,
    precision,
    width,
    label="Precision"
)

bars2 = plt.bar(
    x,
    recall,
    width,
    label="Recall"
)

bars3 = plt.bar(
    x + width,
    f1_score,
    width,
    label="F1-Score"
)

plt.xlabel("Class")
plt.ylabel("Score")
plt.title("PneumoniaAI - Classification Report")
plt.xticks(x, classes)
plt.ylim(0, 1.0)
plt.legend()

# Add values above bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center"
        )

plt.tight_layout()

plt.savefig(
    r"C:\PneumoniaAI\results\classification_report.png",
    dpi=300
)

plt.show()

print("Classification report graph created successfully.")
"""
Project 2: Data Classification Using AI
Pipeline: INPUT (load + scale) -> PROCESS (split + KNN) -> OUTPUT (evaluate)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score,
                             ConfusionMatrixDisplay)


# 1. INPUT: Load and understand the dataset

iris = load_iris()
X, y = iris.data, iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(y, iris.target_names)

print("Shape:", df.shape)                      # (150, 5)
print("\nFirst rows:\n", df.head())
print("\nClass balance:\n", df["species"].value_counts())
print("\nMissing values:", df.isnull().sum().sum())
print("\nSummary stats:\n", df.describe())


# 2. PROCESS: Train-test split (80/20, shuffled, stratified)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, stratify=y, random_state=42
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")


# 3. Feature scaling (StandardScaler: mean=0, variance=1)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# 4. Choose the best K (the "elbow") using cross-validation on TRAIN

k_values = range(1, 31)
errors = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    acc = cross_val_score(knn, X_train_s, y_train, cv=5).mean()
    errors.append(1 - acc)

best_k = k_values[int(np.argmin(errors))]
print(f"\nBest K from cross-validation: {best_k}")

plt.figure(figsize=(7, 4))
plt.plot(k_values, errors, marker="o")
plt.axvline(best_k, color="orangered", linestyle="--", label=f"best K = {best_k}")
plt.xlabel("K value"); plt.ylabel("Error rate"); plt.title("Choosing K")
plt.legend(); plt.tight_layout()
plt.savefig("k_selection.png", dpi=150)


# 5. Instantiate -> Fit -> Predict

model = KNeighborsClassifier(n_neighbors=best_k)   # Instantiate
model.fit(X_train_s, y_train)                      # Fit
predictions = model.predict(X_test_s)              # Predict


# 6. OUTPUT: Validation (don't rely on accuracy alone)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("Macro F1 :", round(f1_score(y_test, predictions, average="macro"), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n",
      classification_report(y_test, predictions, target_names=iris.target_names))

ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=iris.target_names, cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)


# 7. Bonus: predict new flowers (type your own values)


while True:
    text = input("\nEnter 4 values separated by spaces (or 'q' to quit): ")
    if text.lower() == "q":
        break
    values = [float(v) for v in text.split()]
    flower = scaler.transform([values])
    print("Prediction:", iris.target_names[model.predict(flower)[0]])
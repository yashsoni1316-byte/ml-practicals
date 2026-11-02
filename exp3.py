
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Create dataset
df = pd.DataFrame({
    "Weather": ["Sunny", "Sunny", "Rainy", "Overcast",
                "Rainy", "Sunny", "Overcast", "Rainy",
                "Sunny", "Overcast", "Rainy", "Sunny"],
    "Wind": ["Weak", "Strong", "Weak", "Weak",
             "Strong", "Weak", "Strong", "Weak",
             "Strong", "Weak", "Strong", "Weak"],
    "Play": ["No", "No", "Yes", "Yes", "No", "Yes",
             "Yes", "Yes", "No", "Yes", "No", "Yes"]
})

print("Dataset:\n", df)

# 2. Encode categorical data
for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 3. Features and target
X = df[["Weather", "Wind"]]
y = df["Play"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Train model
model = CategoricalNB()
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
cm = confusion_matrix(y_test, y_pred)

print("Predictions:", y_pred)
print("Actual:", y_test.to_numpy())
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", cm)

# 8. Plot confusion matrix
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["No", "Yes"])
plt.yticks([0, 1], ["No", "Yes"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes Confusion Matrix")
plt.show()

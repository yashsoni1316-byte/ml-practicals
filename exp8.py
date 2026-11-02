
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Compare feature subsets
for name, features in [
    ("2 Features", X[:, :2]),
    ("4 Features", X)
]:
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(
        criterion="gini", max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(name, "Accuracy:", acc)

    if name == "2 Features":
        scores = [acc]
    else:
        scores.append(acc)

# Plot comparison
plt.bar(["2 Features", "4 Features"], scores)
plt.ylabel("Accuracy")
plt.title("Decision Tree Feature Comparison")
plt.show()


import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score

# 1. Create dataset
X, y = make_classification(
    n_samples=100, n_features=4,
    n_informative=3, n_redundant=0,
    random_state=42
)

# 2. Hide most labels (-1 means unlabelled)
rng = np.random.RandomState(42)
y_train = y.copy()
unlabelled = rng.choice(100, 70, replace=False)
y_train[unlabelled] = -1

print("Initially labelled:", np.sum(y_train != -1))
print("Initially unlabelled:", np.sum(y_train == -1))

# 3. Apply self-training
model = SelfTrainingClassifier(
    estimator=LogisticRegression(max_iter=1000),
    threshold=0.75
)
model.fit(X, y_train)

# 4. Predict and evaluate
pred = model.predict(X)
print("Labels added:", np.sum(model.transduction_[unlabelled] != -1))
print("Predictions:", pred)
print("Accuracy:", accuracy_score(y, pred))

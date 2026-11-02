
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create dataset
df = pd.DataFrame({
    "Age": [20, 22, 21, 25, 24, 23, 26, 27, 22, 28,
            29, 24, 21, 30, 25, 23, 28, 26, 20, 27],
    "StudyHours": [2, 4, 3, 7, 6, 5, 8, 9, 4, 10,
                   11, 6, 3, 12, 7, 5, 10, 8, 2, 9],
    "Attendance": [60, 70, 65, 85, 80, 75, 90, 95, 70, 98,
                   92, 80, 62, 96, 85, 75, 94, 88, 55, 91],
    "Result": ["Fail", "Fail", "Fail", "Pass", "Pass",
               "Pass", "Pass", "Pass", "Fail", "Pass",
               "Pass", "Pass", "Fail", "Pass", "Pass",
               "Pass", "Pass", "Pass", "Fail", "Pass"]
})

# Add missing values
df.loc[0, "Age"] = np.nan
df.loc[3, "StudyHours"] = np.nan

# 2. Handle missing values
df[["Age", "StudyHours", "Attendance"]] = SimpleImputer(
    strategy="median"
).fit_transform(df[["Age", "StudyHours", "Attendance"]])

print("Missing values:\n", df.isnull().sum())

# 3. Encode target
le = LabelEncoder()
y = le.fit_transform(df["Result"])
X = df[["Age", "StudyHours", "Attendance"]]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Standardization
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

# 6. Normalization
mm = MinMaxScaler()
X_train_min = mm.fit_transform(X_train)
X_test_min = mm.transform(X_test)

# 7. Train and compare models
m1 = LogisticRegression(max_iter=1000)
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)

m2 = LogisticRegression(max_iter=1000)
m2.fit(X_train_std, y_train)
p2 = m2.predict(X_test_std)

m3 = LogisticRegression(max_iter=1000)
m3.fit(X_train_min, y_train)
p3 = m3.predict(X_test_min)

a1 = accuracy_score(y_test, p1)
a2 = accuracy_score(y_test, p2)
a3 = accuracy_score(y_test, p3)

print("Original Accuracy:", a1)
print("Standardization Accuracy:", a2)
print("Normalization Accuracy:", a3)

# 8. Log transformation example
df["LogStudyHours"] = np.log1p(df["StudyHours"])
print(df[["StudyHours", "LogStudyHours"]].head())

# 9. Plot comparison
plt.bar(["Original", "Standardized", "Normalized"],
        [a1, a2, a3])
plt.ylabel("Accuracy")
plt.title("Feature Engineering Comparison")
plt.show()

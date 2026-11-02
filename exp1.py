
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.DataFrame({
    "YearsExperience": [1, 2, None, 4, 5, 5],
    "Salary": [30000, 40000, 50000, None, 60000, 60000]
})

# Missing values
print(df.isnull().sum())
df = df.fillna(df.median(numeric_only=True))

# Duplicates
print(df.duplicated().sum())
df = df.drop_duplicates()

# Outliers using IQR
for c in df.columns:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    print(c, df[(df[c] < q1-1.5*iqr) |
                (df[c] > q3+1.5*iqr)])

# Visualization
df.hist()
plt.show()
df.boxplot()
plt.show()

# Model evaluation
X = df[["YearsExperience"]]
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

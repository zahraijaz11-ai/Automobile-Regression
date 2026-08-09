import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Automobile.csv")
print(df.head())

print("\nShape: ", df.shape)
print("Info: ", df.info())
print("types: ", df.dtypes)
print("Description: ", df.describe())
print("Is any null: ", df.isnull().sum())
print("Duplicated: ", df.duplicated().sum())

df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
print("Is any null: ", df.isnull().sum())

df = pd.get_dummies(df, columns=["origin"], drop_first=True)
df.drop("name", axis=1, inplace=True)
print(df.head())
df.boxplot(figsize=(12, 6))
plt.xticks(rotation=45)
plt.show()
# Histograms help you understand how each numerical feature is distributed.
df.hist(figsize=(12, 10), bins=15)  # y-axis=cars fall into each range.
plt.tight_layout()
plt.show()
# helps understand how strongly features are related to each other&to target
print(df.corr(numeric_only=True))  # Positive correlation
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()

# Since we want to predict MPG, it is our target variable.
# Features and Target
X = df.drop("mpg", axis=1)
y = df["mpg"]
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Polynomial Features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("\nBefore Polynomial:", X_train_scaled.shape)
print("After Polynomial:", X_train_poly.shape)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(comparison)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\nR2 Score:", r2)  # model condition
print("Mean Absolute Error:", mae)  # prediction accuracy
print("Mean Squared Error:", mse)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color="red")
plt.xlabel("Predicted MPG")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

new_car = pd.DataFrame({
    "cylinders": [4],
    "displacement": [120],
    "horsepower": [95],
    "weight": [2500],
    "acceleration": [15],
    "model_year": [82],
    "origin_japan": [1],
    "origin_usa": [0]
})
new_car_scaled = scaler.transform(new_car)
new_car_poly = poly.transform(new_car_scaled)
predicted_mpg = model.predict(new_car_poly)
print("\nPredicted MPG:", predicted_mpg[0])

df.to_csv("processed_Automobile.csv", index=False)
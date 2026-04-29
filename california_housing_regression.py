# California housing price regression (simple + multiple)
# Uses sklearn's built-in California housing dataset (20,640 samples)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["median_house_value"] = housing.target * 100000  # target is in $100k units
print(f"Dataset shape: {df.shape}")
print("First 10 rows:")
print(df.head(10))
print()

# Simple linear regression (MedInc only)
X_simple = df[["MedInc"]]
y = df["median_house_value"]

train_x, test_x, train_y, test_y = train_test_split(
    X_simple, y, test_size=0.2, random_state=10
)

simple_model = linear_model.LinearRegression()
simple_model.fit(train_x, train_y)

simple_pred = simple_model.predict(test_x)

print("--- Simple Linear Regression ---")
print("Mean squared error: %.2f" % mean_squared_error(test_y, simple_pred))
print("R² score:           %.2f" % r2_score(test_y, simple_pred))
print("Model score:        ", simple_model.score(test_x, test_y))
print()

# Plot
plt.figure()
plt.scatter(test_x, test_y, color="black", alpha=0.4, label="Actual")
sort_idx = np.argsort(test_x.values.flatten())
plt.plot(test_x.values.flatten()[sort_idx], simple_pred[sort_idx], color="blue", linewidth=3, label="Predicted")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Median Income vs House Value")
plt.legend()
plt.tight_layout()
plt.show()

# Multiple linear regression (all features)
X_multi = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

train_x_m, test_x_m, train_y_m, test_y_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=10
)

multi_model = linear_model.LinearRegression()
multi_model.fit(train_x_m, train_y_m)

multi_pred = multi_model.predict(test_x_m)

print("--- Multiple Linear Regression ---")
print("Mean squared error: %.2f" % mean_squared_error(test_y_m, multi_pred))
print("R² score:           %.2f" % r2_score(test_y_m, multi_pred))
print("Model score:        ", multi_model.score(test_x_m, test_y_m))
print()

# Plot
plt.figure()
plt.scatter(test_y_m, multi_pred, color="red", alpha=0.4, label="Predictions")
line_min, line_max = test_y_m.min(), test_y_m.max()
plt.plot([line_min, line_max], [line_min, line_max], color="black", label="y = y (perfect)")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Predicted vs Actual")
plt.legend()
plt.tight_layout()
plt.show()

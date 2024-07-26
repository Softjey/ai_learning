import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

RANDOM_SEED = 42
dataset_size = 1000
noise_size = 0.05

np.random.seed(RANDOM_SEED)

features = np.random.randn(dataset_size, 5)
features_coefficients = np.array([-2.345, 3.456, -4.567, 5.678, 6.789])
features_b = np.array([1.234, -2.345, 3.456, -4.567, 5.678])
intercept = 3.623

y_true = np.dot(features, features_coefficients) + intercept + np.random.randn(dataset_size) * noise_size

data = train_test_split(features, y_true, test_size=0.5, random_state=RANDOM_SEED)
train_features, test_features, train_y, test_y = data

model = LinearRegression()

model.fit(train_features, train_y)
y_predicted = model.predict(test_features)

print(f"True coefficients: {features_coefficients}")
print(f"Model coefficients: {model.coef_}")

print(f"True intercept: {intercept}")
print(f"Intercept : {model.intercept_}")

print(f"Absolute error: {round(mean_absolute_percentage_error(test_y, y_predicted) * 100, 2)}%")
print(f"Determination coefficient: {model.score(test_features, test_y)}")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from helpers import generate_percentage_noise, generate_polynomial_data

RANDOM_SEED = 193
DATASET_SIZE = 200
NOISE_LEVEL = 0.3
N_FEATURES = 1

np.random.seed(RANDOM_SEED)

X = np.random.randn(DATASET_SIZE, N_FEATURES)
y = generate_percentage_noise(generate_polynomial_data(X, [3.3, 1.5, 12.32, 24]), NOISE_LEVEL)

# X = np.array([2, 6, 8, 42, 186, 2048, 1]).reshape(-1, 1)
# y = np.array([-0.006, 542, 5.16, -196, 458, 245, -421.286]).reshape(-1, 1)

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
train_x, test_x, train_y, test_y = train_test_split(X_poly, y, random_state=RANDOM_SEED, test_size=0.2)

model = LinearRegression()
model.fit(train_x, train_y)
print('R^2:', model.score(test_x, test_y))

# Regression function visualization
coefficients = model.coef_[0]
intercept = model.intercept_[0]

x_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
poly_x_range = poly.fit_transform(x_range)

y_range_pred = np.dot(poly_x_range, coefficients) + intercept

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(x_range, y_range_pred, 'r')
plt.show()

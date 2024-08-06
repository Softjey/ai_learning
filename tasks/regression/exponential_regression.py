import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

DATASET_SIZE = 200
FEATURES_COUNT = 1
NOISE = 0.3

def generate_percentage_noise(Y):
    noise = Y * NOISE * np.random.randn(*Y.shape)

    return Y + noise

X = np.random.rand(DATASET_SIZE, FEATURES_COUNT)
y = generate_percentage_noise(np.exp(3 * X + 14))
y_log = np.log(y)

train_x, test_x, train_y, test_y = train_test_split(X, y_log)

model = LinearRegression()

model.fit(train_x, train_y)

print("Model R^2 score: ", model.score(test_x, test_y))

X_generated = np.linspace(min(X), max(X), 100).reshape(-1, 1)
y_generated = model.predict(X_generated)
y_generated = np.exp(y_generated)

plt.scatter(X, y, alpha=0.5)
plt.plot(X_generated, y_generated, color="orange", linewidth=2)
plt.show()
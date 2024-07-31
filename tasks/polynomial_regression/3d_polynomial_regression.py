import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from helpers import generate_percentage_noise, generate_polynomial_data

RANDOM_SEED = 193
DATASET_SIZE = 200
NOISE_LEVEL = 0.1
N_FEATURES = 2

np.random.seed(RANDOM_SEED) 

features = (np.random.rand(DATASET_SIZE, N_FEATURES) - 0.5) * 10
x = features[:, 0]
z = features[:, 1]

y = generate_percentage_noise(0.07 * z ** 3 + 3 * x ** 2 - 5 * x, NOISE_LEVEL)

poly = PolynomialFeatures(degree=3)

print(f'X shape before poly transform :{features.shape}')
poly_features = poly.fit_transform(features)
print(f'X shape after poly transform :{poly_features.shape}')

x_train, x_test, y_train, y_test = train_test_split(poly_features, y, random_state=RANDOM_SEED)

model = LinearRegression()

model.fit(x_train, y_train)
r2_score = model.score(x_test, y_test)

print("Model R^2 score:", r2_score)

x_generated = np.linspace(min(x), max(x), 100)
z_generated = np.linspace(min(z), max(z), 100)
x_grid, z_grid = np.meshgrid(x_generated, z_generated)
features_generated = poly.transform(np.c_[x_grid.ravel(), z_grid.ravel()])
y_generated = model.predict(features_generated)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(features[:, 0], y, features[:, 1])
ax.plot_surface(x_grid, y_generated.reshape(x_grid.shape), z_grid, color="red", alpha=0.5)
 
plt.show()
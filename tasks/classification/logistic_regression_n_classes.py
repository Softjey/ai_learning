from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 59
GRAPH_DPI = 20

x, y = make_classification(
    n_classes=8,
    n_samples=800,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=2,
    random_state=RANDOM_SEED,
)

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_poly, y, random_state=RANDOM_SEED)

model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

print(f'Accuracy: {model.score(x_test, y_test)}')

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d")

for y_class in np.unique_values(y):
    ax1.scatter(x[y == y_class, 0], x[y == y_class, 1], x[y == y_class, 2])

xs_generated = np.meshgrid(
    *[np.linspace(min(x[:, i]), max(x[:, i]), GRAPH_DPI) for i in range(x.shape[1])]
)
x_generated = np.c_[*[feature.ravel() for feature in xs_generated]]

x_generated = poly.transform(x_generated)
y_generated = model.predict(x_generated)

ax2 = fig.add_subplot(122, projection="3d")

ax2.scatter(*xs_generated, alpha=0.8, c=y_generated, cmap="plasma", s=10)

plt.show()

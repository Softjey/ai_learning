from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# RANDOM_SEED = 4220601
RANDOM_SEED = 2834
GRAPH_DPI = 250

x, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=2,
    class_sep=1.5,
    flip_y=0.01,
    random_state=RANDOM_SEED,
)

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

model = LogisticRegression()

model.fit(x_poly, y)

x1_generated, x2_generated = np.meshgrid(
    np.linspace(min(x[:, 0]), max(x[:, 0]), GRAPH_DPI),
    np.linspace(min(x[:, 1]), max(x[:, 1]), GRAPH_DPI),
)

x_generated = np.c_[(x1_generated.ravel(), x2_generated.ravel())]
x_generated = poly.transform(x_generated)
y_generated_proba = model.predict_proba(x_generated)[:, 1].reshape(x1_generated.shape)
y_generated = model.predict(x_generated).reshape(x1_generated.shape)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121)

ax1.contourf(x1_generated, x2_generated, y_generated_proba)
ax1.contour(x1_generated, x2_generated, y_generated, colors=["red"])

ax1.scatter(x[y == 0, 0], x[y == 0, 1], color="pink", alpha=0.5, marker="s")
ax1.scatter(x[y == 1, 0], x[y == 1, 1], color="blue", alpha=0.5, marker="o")

ax2 = fig.add_subplot(122, projection="3d")

ax2.plot_surface(x1_generated, x2_generated, y_generated_proba, alpha=0.5, cmap='coolwarm')
ax2.plot_wireframe(x1_generated, x2_generated, y_generated_proba, color="black", alpha=0.3)

plt.show()

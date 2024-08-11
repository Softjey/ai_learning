from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# RANDOM_SEED = 4220601
RANDOM_SEED = 2834
GRAPH_DPI = 12

x, y = make_classification(
    n_samples=1000,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=3,
    class_sep=2,
    flip_y=0.01,
    random_state=RANDOM_SEED,
)

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

model = LogisticRegression(max_iter=1000)

model.fit(x_poly, y)

x1_generated, x2_generated, x3_generated = np.meshgrid(
    np.linspace(min(x[:, 0]), max(x[:, 0]), GRAPH_DPI),
    np.linspace(min(x[:, 1]), max(x[:, 1]), GRAPH_DPI),
    np.linspace(min(x[:, 2]), max(x[:, 2]), GRAPH_DPI),
)

x_generated = np.c_[(x1_generated.ravel(), x2_generated.ravel(), x3_generated.ravel())]
x_generated = poly.transform(x_generated)
y_generated = model.predict(x_generated).reshape(x1_generated.shape)
y_generated_proba = model.predict_proba(x_generated)[:, 0].reshape(x1_generated.shape)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d")

scatter = ax1.scatter(
    x1_generated,
    x2_generated,
    x3_generated,
    c=y_generated,
    cmap="viridis",
    alpha=0.6,
    s=30,
)

ax2 = fig.add_subplot(122, projection="3d")

ax2.scatter(
    x1_generated,
    x2_generated,
    x3_generated,
    c=y_generated_proba,
    cmap="viridis",
    alpha=0.6,
    s=30,
)

fig.colorbar(scatter, ax=[ax1, ax2], label='Y')

plt.show()

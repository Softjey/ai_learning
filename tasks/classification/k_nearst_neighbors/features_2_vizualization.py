from sklearn.datasets import make_moons, make_blobs, make_circles, make_gaussian_quantiles
from sklearn.neighbors import KNeighborsClassifier
from helpers import make_hearts, make_checkerboard
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time

RANDOM_SEED = 120
GRAPH_DPI = 400
SAMPLES = 250
MARGINS = 1.1

moons = make_moons(n_samples=SAMPLES, noise=0.1, random_state=RANDOM_SEED)
blobs = make_blobs(n_samples=SAMPLES, centers=2, cluster_std=0.6,  random_state=RANDOM_SEED)
circles = make_circles(n_samples=SAMPLES, noise=0.05, random_state=RANDOM_SEED)
quantiles = make_gaussian_quantiles(n_samples=SAMPLES, n_classes=2, random_state=RANDOM_SEED)
hearts = make_hearts(SAMPLES, SAMPLES, noise_heart=0.1, noise_inner=0.1, random_state=RANDOM_SEED)
checkerboard = make_checkerboard(n_samples=SAMPLES, noise=0.02, random_state=RANDOM_SEED)

fig, axes = plt.subplots(ncols=3, nrows=2)

for (x, y), ax in zip([moons, blobs, circles, quantiles, hearts, checkerboard], axes.ravel()):
  start = time.time()
  model = KNeighborsClassifier()
  model.fit(x, y)

  xx, zz = np.meshgrid(
      np.linspace((min(x[:, 0]) - 0.2) * MARGINS, max(x[:, 0]) * MARGINS, GRAPH_DPI),
      np.linspace((min(x[:, 1]) - 0.2) * MARGINS, max(x[:, 1]) * MARGINS, GRAPH_DPI)
  )
  y_pred = model.predict(np.c_[xx.ravel(), zz.ravel()])
  ax.contourf(xx, zz, y_pred.reshape(xx.shape), alpha=0.5)
  ax.scatter(x[:, 0], x[:, 1], c=y, edgecolor='k')
  print(f"Done for {y.shape[0]} samples in {round(time.time() - start, 2)}s")

plt.show()

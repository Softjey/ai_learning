from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 97375

x, y = make_classification(
  n_samples=200,
  n_features=1,
  n_informative=1,
  n_redundant=0,
  n_classes=2,
  n_clusters_per_class=1,
  class_sep=2,
  flip_y=0.03,
  random_state=RANDOM_SEED
)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=RANDOM_SEED)

model = LogisticRegression()

model.fit(x_train, y_train)

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.scatter(x[y == 0], y[y == 0], color="blue")
ax1.scatter(x[y == 1], y[y == 1], color="red")

x_generated = np.linspace(min(x), max(x), 10)
y_generated = model.predict(x_generated)
y_generated_proba = model.predict_proba(x_generated)

ax1.plot(x_generated, y_generated_proba[:, 1], color="orange")
ax1.plot(x_generated, y_generated, color="green")

plt.show()

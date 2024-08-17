from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# import pickle
import matplotlib.pyplot as plt
import numpy as np

dataset = load_digits()
target = dataset['target']
features = dataset['data']

# print(target.shape)
# print(np.unique_counts(target))
# print(features.shape)
# print(dataset['images'].shape)

pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('model', LogisticRegressionCV(max_iter=1000, Cs=np.linspace(0.001, 10, 50)))
])

grid_param = {
    # 'model__solver': [
    #     'lbfgs',
    #     'liblinear',  # the best with l2
    #     'newton-cg',
    #     'saga',
    #     'sag',
    # ]
}

grid_search = GridSearchCV(pipeline, grid_param, verbose=3)
grid_search.fit(features, target)

print(f'Best score: {grid_search.best_score_}')
print(f'Alpha: {grid_search.best_estimator_.named_steps["model"].C_}')

fig, axes = plt.subplots(nrows=4, ncols=10, figsize=(10, 4))

for ax, image, num, in zip(axes.ravel(), dataset['images'], target):
  ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
  ax.set_title(f"{num}")

plt.show()

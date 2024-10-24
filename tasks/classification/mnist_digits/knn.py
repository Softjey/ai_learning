from tasks.classification.mnist_digits.helpers import get_mnist_dataset
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

features, target, images = get_mnist_dataset()
fig, axes = plt.subplots(nrows=4, ncols=10, figsize=(10, 4))

pipeline = Pipeline([
    ('pca', PCA(n_components=50)),
    ('model', KNeighborsClassifier(n_jobs=-1))
])

grid_param = {
  'pca__n_components': [20, 30, 50, 70],
  'model__n_neighbors': [3, 5, 7, 9, 11],
}

grid_search = GridSearchCV(pipeline, grid_param, n_jobs=-1, verbose=3)
grid_search.fit(features, target)

print(f'Best score: {grid_search.best_score_}')
print(f'Best params: {grid_search.best_params_}')

for ax, image, num, in zip(axes.ravel(), images, target):
  ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
  ax.set_title(f"{num}")

plt.show()

with open(f'knn_model_{grid_search.best_score_}.pkl', 'wb') as file:
  pickle.dump(grid_search.best_estimator_, file)

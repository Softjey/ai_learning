from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
import numpy as np
import pickle

dataset = load_digits()
target = dataset['target']
features = dataset['data']

pipeline = Pipeline([
    ('model', KNeighborsClassifier(n_jobs=-1))
])

grid_param = {
    "model__n_neighbors": np.arange(1, 41),
    "model__weights": ["uniform", "distance"],
    "model__metric": [
        'euclidean',
        'manhattan',
        # 'minkowski',
        # 'chebyshev',
        # 'wminkowski',
        # 'seuclidean',
        # 'mahalanobis',
        # 'hamming',
        # 'canberra',
        # 'braycurtis',
        # 'matching',
        # 'jaccard',
        # 'dice',
        # 'kulsinski',
        # 'rogerstanimoto',
        # 'russellrao',
        # 'sokalmichener',
        # 'sokalsneath',
        # 'yule'
    ],
}

grid_search = GridSearchCV(pipeline, grid_param, verbose=3, n_jobs=-1)

grid_search.fit(features, target)

print(f'Best Score: {grid_search.best_score_}')
print(f'Best Params: {grid_search.best_params_}')

with open(f'knn_model_{round(grid_search.best_score_, 4)}.pkl', 'wb') as file:
  pickle.dump(grid_search.best_estimator_, file)

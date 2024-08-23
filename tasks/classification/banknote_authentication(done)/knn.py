from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from helpers import get_csv_dataset
import numpy as np

dataset = get_csv_dataset(__file__, './banknote_authentication_dataset.csv')

features = dataset.drop(['class'], axis=1)
target = dataset['class']

pipeline = Pipeline([
    ('model', KNeighborsClassifier(n_jobs=None))
])

grid_param = {
    'model__n_neighbors': np.arange(1, 21),
}

grid_search = GridSearchCV(pipeline, grid_param, verbose=3)
grid_search.fit(features, target)

print(f'Best Score: {grid_search.best_score_}')
print(f'Best Params: {grid_search.best_params_}')

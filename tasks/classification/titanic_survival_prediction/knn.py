from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from helpers import get_csv_dataset
import pandas as pd
import numpy as np
import pickle

RANDOM_SEED = 89

raw_dataset = get_csv_dataset(__file__, "./titanic-dataset.csv")
dataset = raw_dataset.drop(["Cabin", "PassengerId", "Ticket", "Name"], axis=1)

target = dataset["Survived"]
categorical_features = dataset.select_dtypes("object")
numerical_features = dataset.select_dtypes(["int64", "float64"]).drop("Survived", axis=1)
features = pd.concat([categorical_features, numerical_features], axis=1)

pipeline = Pipeline([
    ('preprocessing', ColumnTransformer([
        ('numerical', Pipeline([
            ('imputer', SimpleImputer(strategy="mean")),
            ('scaler', StandardScaler())
        ]), numerical_features.columns),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('encoder', OneHotEncoder())
        ]), categorical_features.columns)
    ])),
    ('model', KNeighborsClassifier())
])

grid_param = {
    "model__n_neighbors": np.arange(1, 41),
    "model__weights": ["uniform", "distance"],
    # "model__algorithm": ["ball_tree", "kd_tree", "brute"],
    "model__metric": [
        'wminkowski',
        # 'euclidean',
        'manhattan',
        'chebyshev',
        # 'minkowski',
        'seuclidean',
        # 'mahalanobis',
        'hamming',
        'canberra',
        'braycurtis',
        'matching',
        'jaccard',
        'dice',
        'kulsinski',
        'rogerstanimoto',
        'russellrao',
        'sokalmichener',
        'sokalsneath',
        # 'yule'
    ],
}

grid_search = GridSearchCV(pipeline, grid_param, verbose=3)

grid_search.fit(features, target)

print(f'Best Score: {grid_search.best_score_}')
print(f'Best Params: {grid_search.best_params_}')

with open(f'knn_model_{round(grid_search.best_score_, 4)}.pkl', 'wb') as file:
  pickle.dump(grid_search.best_estimator_, file)

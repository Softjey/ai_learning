from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
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
            ('poly', PolynomialFeatures()),
            ('scaler', StandardScaler()),
        ]), numerical_features.columns),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder())
            # ('encoder', OneHotEncoder(sparse_output=False))
        ]), categorical_features.columns)
    ])),
    # ('model', LogisticRegression(max_iter=2000))
    ('model', LogisticRegressionCV(Cs=np.logspace(-1, 3, 2000), max_iter=10000))
])

param_grid = {
    'preprocessing__numerical__poly__degree': [2],
    'model__solver': [
        # 'lbfgs',
        'liblinear',  # the best with l2
        # 'newton-cg',
        # 'saga',
        # 'sag',
    ],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=3)
grid_search.fit(features, target)

print(f"Models: {param_grid['model__solver']}")
print(f'Scores: {grid_search.cv_results_["mean_test_score"]}',)
print(f'Time  : {grid_search.cv_results_["mean_fit_time"]}',)
print(f'Best Model Alpha: {grid_search.best_estimator_.named_steps["model"].C_}',)

print(f'Best Model {grid_search.best_params_}')
with open('logistic_regression_model.pkl', 'wb') as file:
  pickle.dump(grid_search.best_estimator_, file)

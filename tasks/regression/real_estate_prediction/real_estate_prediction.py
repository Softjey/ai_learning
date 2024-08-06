import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

start = time.time()

RANDOM_SEED = 4135

dataset_path = os.path.join(os.path.dirname(__file__), 'real_estate_dataset.csv')
estate_dataset = pd.read_csv(dataset_path)

features = estate_dataset.drop(['house_price_of_unit_area', 'no'], axis=1)
response = np.log(estate_dataset['house_price_of_unit_area'])

print('Before poly: ', features.shape)

poly = PolynomialFeatures(degree=3)
features_poly = poly.fit_transform(features)

print('After poly: ', features_poly.shape)

scaler = StandardScaler()
features_poly = scaler.fit_transform(features_poly)

model = RidgeCV(alphas=np.linspace(0.1, 10, 100))


def score_fn(model, x, y):
    return mean_absolute_percentage_error(y, model.predict(x))


r2_validation = cross_validate(model, features_poly, response, cv=5, scoring="r2", return_estimator=True)
percentage_error_validation = cross_validate(model, features_poly, response, cv=5, scoring=score_fn, return_train_score=True)
mean_train_error = np.mean(percentage_error_validation['train_score'])
mean_test_error = np.mean(percentage_error_validation['test_score'])
generalization_error = (mean_test_error - mean_train_error) / mean_train_error

print(f'Mean R^2: {np.mean(r2_validation['test_score'])}')
print(f'Mean Training set MAPE: {round(mean_train_error * 100, 2)}%')
print(f'Mean Test set MAPE: {round(mean_test_error * 100, 2)}%')
print(f'Generalization Error: {round(generalization_error * 100, 2)}%')
print(f'Execution time: {time.time() - start}')

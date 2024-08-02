import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import os
import numpy as np

RANDOM_SEED=413

dataset_path = os.path.join(os.path.dirname(__file__), 'real_estate_dataset.csv')
estate_dataset = pd.read_csv(dataset_path)

features = estate_dataset.drop(['house_price_of_unit_area', 'no'], axis=1)
response = np.log(estate_dataset['house_price_of_unit_area'])

poly = PolynomialFeatures(degree=2)
features_poly = poly.fit_transform(features)

train_x, test_x, train_y, test_y = train_test_split(features_poly, response, random_state=RANDOM_SEED)

model = LinearRegression()
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
abs_error = round(mean_absolute_percentage_error(test_y, pred_y) * 100, 2)

def score_fn(model, x, y):
  return mean_absolute_percentage_error(y, model.predict(x))

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
r2_score = cross_val_score(model, features_poly, response, cv=kf, scoring="r2")
abs_error = cross_val_score(model, features_poly, response, cv=kf, scoring=score_fn)

print('Mean Cross-Validation R^2: ', np.mean(r2_score))
print(f'Mean Cross-Validation absolute percentage error: {round(np.mean(abs_error) * 100, 2)}%')

# x_generated = np.array(features)
# x_generated_poly = poly.transform(x_generated)
# y_pred = model.predict(x_generated_poly)

# for i, column in enumerate(features.columns):
#   fig, ax = plt.subplots()
#   column_data = x_generated[:, i]
#   x_generated_column = np.linspace(min(column_data), max(column_data), 100)
 
#   ax.scatter(estate_dataset[column], response, alpha=0.5)
#   ax.plot(x_generated_column, y_pred, color="orange")
#   ax.set_xlabel(column)
#   ax.set_ylabel('House Price of Unit Area')
  
#   plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import os

RANDOM_SEED=1

dataset_path = os.path.join(os.path.dirname(__file__), 'real_estate_dataset.csv')
estate_dataset = pd.read_csv(dataset_path)

features = estate_dataset.drop(['house_price_of_unit_area', 'no'], axis=1)
response = estate_dataset['house_price_of_unit_area']
test_x, train_x, test_y, train_y = train_test_split(features, response, random_state=RANDOM_SEED, test_size=0.2)

model = LinearRegression()

model.fit(train_x, train_y)
pred_y = model.predict(test_x)
abs_error = round(mean_absolute_percentage_error(test_y, pred_y) * 100, 2)

print('Absolute Percentage Error: ', f"{abs_error}%")
print('R^2: ', r2_score(test_y, pred_y))

# for i, column in enumerate(features.columns):
#   fig, ax = plt.subplots()
  
#   ax.scatter(estate_dataset[column], response, alpha=0.5)
#   ax.set_xlabel(column)
#   ax.set_ylabel('House Price of Unit Area')
  
#   plt.show()

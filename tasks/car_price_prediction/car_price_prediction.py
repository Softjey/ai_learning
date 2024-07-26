import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'cars_dataset.csv')

cars_dataset = pd.read_csv(file_path)

one_hot_encoder = OneHotEncoder()
cars_dataset['clean_title'] = cars_dataset['clean_title'].map({ 'Yes': 1, 'No': 0 })
cars_dataset['brand'] = one_hot_encoder.fit_transform(cars_dataset['brand'].values.resize(1, 1))

print('Data loaded successfully', cars_dataset.head())

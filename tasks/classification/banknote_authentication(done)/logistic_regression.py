from helpers import get_csv_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import itertools

dataset = get_csv_dataset(__file__, './banknote_authentication_dataset.csv')

# print(dataset['class'].value_counts())

features = dataset.drop(['class'], axis=1)
target = dataset['class']

pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('model', LogisticRegression())
])

grid_params = {}

grid_search = GridSearchCV(pipeline, grid_params, cv=5)
grid_search.fit(features, target)

print(f'Best score {grid_search.best_score_}')

num_features = len(features.columns)
num_combinations = num_features * (num_features - 1) // 2

fig, axs = plt.subplots(nrows=1, ncols=num_combinations, figsize=(5 * num_combinations, 5))

for i, (feature_x, feature_y) in enumerate(itertools.combinations(features.columns, 2)):
    ax = axs[i] if num_combinations > 1 else axs

    ax.scatter(features[feature_x][target == 0], features[feature_y][target == 0], label='Class 0')
    ax.scatter(features[feature_x][target == 1], features[feature_y][target == 1], label='Class 1')

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend()

plt.tight_layout()
plt.show()
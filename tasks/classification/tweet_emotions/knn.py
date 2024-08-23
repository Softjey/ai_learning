from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import GridSearchCV
from helpers import get_csv_dataset
import numpy as np
import pickle

RANDOM_SEED = 42

dataset = get_csv_dataset(__file__, './tweet_emotions.csv')

# dataset = dataset.sample(frac=1, random_state=RANDOM_SEED)

features = dataset.drop(columns=['tweet_id', 'sentiment'], axis=1)
target = dataset['sentiment']


pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS))),
    ('model', KNeighborsClassifier(n_jobs=-1)),
])

grid_params = {
    # "vectorizer__ngram_range": [(1, 1), (1, 2)],
    'vectorizer__max_df': [0.6, 0.7, 0.8,],
    'model__n_neighbors': np.arange(12, 15),
    # 'model__weights': ['uniform', 'distance'],
}

grid_search = GridSearchCV(pipeline, grid_params, verbose=3, n_jobs=-1)

grid_search.fit(features['content'], target)

print(f'Best score: {grid_search.best_score_}')
print(f'Best params: {grid_search.best_params_}')

with open(f'knn_model_{round(grid_search.best_score_, 4)}.pkl', 'wb') as file:
  pickle.dump(grid_search.best_estimator_, file)

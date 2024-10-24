from tasks.classification.mnist_digits.helpers import get_mnist_dataset
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from helpers import fit_grid_search
import pickle

features, target, images = get_mnist_dataset()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('model', LogisticRegressionCV(n_jobs=-1, max_iter=1000)),
])

grid_param = {
    'pca__n_components': [20, 30, 50, 70],
    # 'model__solver': ['liblinear', 'saga', 'lbfgs'],
    # 'model__penalty': ['l1', 'l2', 'elasticnet'],
    # 'model__Cs': np.logspace(-2, 2, 10),
}

grid_search = GridSearchCV(
    pipeline,
    grid_param,
    n_jobs=-1,
    verbose=3,
    scoring='accuracy'
)

fit_grid_search(grid_search, features, target)

with open(f'logistic_regression_model_{grid_search.best_score_}.pkl', 'wb') as file:
  pickle.dump(grid_search.best_estimator_, file)

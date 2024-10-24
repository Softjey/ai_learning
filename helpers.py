from sklearn.model_selection import GridSearchCV
import os
import pandas as pd
import numpy as np
import pickle


def get_csv_dataset(file_path: str, dataset_path: str) -> pd.DataFrame:
  current_dir = os.path.dirname(file_path)
  dataset_path = os.path.join(current_dir, dataset_path)

  return pd.read_csv(dataset_path, on_bad_lines="warn")


def unpickle(file_path: str, dataset_path: str, **setup):
  current_dir = os.path.dirname(file_path)
  dataset_path = os.path.join(current_dir, dataset_path)

  with open(dataset_path, 'rb') as file:
    dict = pickle.load(file, **setup)
  return dict


def fit_grid_search(grid_search: GridSearchCV, features, target):
  grid_search.fit(features, target)

  print(f'Best score: {grid_search.best_score_}')
  print(f'Best params: {grid_search.best_params_}')


def make_hearts(
    n_samples_heart=1000,
    n_samples_inner=500,
    noise_heart=0.05,
    noise_inner=0.05,
    inner_scale=0.5,
    random_state=None
):
  if random_state is not None:
    np.random.seed(random_state)

  t = np.linspace(0, 2 * np.pi, n_samples_heart)
  x_heart = 16 * np.sin(t) ** 3
  y_heart = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

  x_heart += noise_heart * np.random.randn(n_samples_heart)
  y_heart += noise_heart * np.random.randn(n_samples_heart)
  X_heart = np.c_[x_heart, y_heart]
  y_heart = np.ones(n_samples_heart)

  t_inner = np.linspace(0, 2 * np.pi, n_samples_inner)
  x_inner = inner_scale * 16 * np.sin(t_inner) ** 3
  y_inner = inner_scale * (13 * np.cos(t_inner) - 5 * np.cos(2 * t_inner) -
                           2 * np.cos(3 * t_inner) - np.cos(4 * t_inner))

  x_inner += noise_inner * np.random.randn(n_samples_inner)
  y_inner += noise_inner * np.random.randn(n_samples_inner)
  X_inner = np.c_[x_inner, y_inner]
  y_inner = np.zeros(n_samples_inner)

  X = np.vstack([X_heart, X_inner])
  y = np.hstack([y_heart, y_inner])

  return X, y


def make_checkerboard(n_samples=1000, noise=0.1, random_state=None):
  if random_state is not None:
    np.random.seed(random_state)

  n_cells = int(np.sqrt(n_samples))
  n_cells = n_cells if n_cells % 2 == 0 else n_cells - 1  # Робимо кількість клітинок парною

  x = np.linspace(0, 4, n_cells + 1)  # +1 щоб включити крайні точки
  y = np.linspace(0, 4, n_cells + 1)
  xv, yv = np.meshgrid(x[:-1], y[:-1])  # Виключаємо останню точку, щоб уникнути неповних клітинок

  X = np.c_[xv.ravel(), yv.ravel()]
  y = ((xv.astype(int) % 2) ^ (yv.astype(int) % 2)).ravel()

  X += noise * np.random.randn(*X.shape)

  return X, y

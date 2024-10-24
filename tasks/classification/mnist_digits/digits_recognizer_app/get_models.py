from helpers import unpickle

models = [
    {
        'model_name': 'Knn model',
        'path': '../models/knn_model_0.9794.pkl',
    },
    {
        'model_name': 'Logistic Regression model',
        'path': '../models/logistic_regression_model_0.9116.pkl',
    }
]


def get_models():
  return [
      {
          **model,
          'model': unpickle(__file__, model['path'])
      }
      for model in models
  ]

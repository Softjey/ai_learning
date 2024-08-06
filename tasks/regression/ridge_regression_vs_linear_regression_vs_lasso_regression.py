import math
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_validate


class ModelDict(TypedDict):
    name: str
    estimator: BaseEstimator
    color: str


RANDOM_SEED = 493
x, y = make_regression(n_samples=50, n_features=1, noise=15, random_state=RANDOM_SEED)


def test_models(models: list[ModelDict]):
    x_generated = np.linspace(x.min(), x.max(), 100)
    cols = math.ceil(math.sqrt(len(models)))
    rows = math.ceil(len(models) / cols)
    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(15, 5 * rows))

    if len(models) == 1:
        axes = [axes]

    for model, ax in zip(models, axes.flatten()):
        validation = validate(model["estimator"])

        print(f'\n{model["name"]}:')
        print(f'Mean R^2: {np.mean(validation["test_r2"])}')
        print(f"Mean generalization error: {calc_generalization_error(validation)}%")

        y_generated = np.mean(validation["coefficients"]) * x_generated + np.mean(
            validation["intercepts"]
        )
        ax.plot(x_generated, y_generated, color=model["color"])
        ax.scatter(x, y)
        ax.set_title(model["name"])

    plt.tight_layout()
    plt.show()


def mape_scorer(model, x, y):
    return mean_absolute_percentage_error(y, model.predict(x))


def validate(model):
    validation = cross_validate(
        model,
        x,
        y,
        cv=4,
        return_train_score=True,
        scoring={"mape": mape_scorer, "r2": "r2"},
        return_estimator=True,
    )

    coefficients = [est.coef_[0] for est in validation["estimator"]]
    intercepts = [est.intercept_ for est in validation["estimator"]]

    return {**validation, "coefficients": coefficients, "intercepts": intercepts}


def calc_generalization_error(validation):
    test_score = np.mean(validation["test_mape"])
    train_score = np.mean(validation["train_mape"])
    percentage_error = (test_score - train_score) / train_score

    return round(percentage_error * 100, 2)


test_models(
    [
        {
            "color": "orange",
            "estimator": LinearRegression(),
            "name": "Linear Regression",
        },
        {"color": "red", "estimator": Ridge(), "name": "Ridge Regression"},
        {"color": "red", "estimator": Lasso(), "name": "Lasso Regression"},
    ]
)

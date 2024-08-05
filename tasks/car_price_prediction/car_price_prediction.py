import time
import numpy as np
import re
import pandas as pd
from helpers import get_csv_dataset
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

RANDOM_SEED = 123
DATASET_SIZE = 0.1


def extract_engine_features(engine_str):
    hp = re.search(r"(\d+\.0)HP", engine_str)
    displacement = re.search(r"(\d+\.\d+)L", engine_str)
    engine_type = re.search(r"(V\d+|Straight \d+|Flat \d+|I\d+)", engine_str)
    fuel_type = re.search(r"(Gasoline|Hybrid|Electric|Flex Fuel|Diesel|CNG|LPG|E85 Flex Fuel)", engine_str, re.IGNORECASE)
    return {
        "horse_power": float(hp.group(1)) if hp else None,
        "displacement": float(displacement.group(1)) if displacement else None,
        "engine_type": engine_type.group(1) if engine_type else None,
        "fuel_type": fuel_type.group(0) if fuel_type else None
    }


dropped_columns = ["id", "engine"]
cars_dataset = get_csv_dataset(__file__, "cars_dataset.csv")
cars_dataset = cars_dataset.replace('-', pd.NA)
engine_features = cars_dataset["engine"].apply(extract_engine_features)
cars_dataset = pd.concat([cars_dataset, engine_features.apply(pd.Series)], axis=1)
cars_dataset = cars_dataset.drop(dropped_columns, axis=1)
cars_dataset = cars_dataset.sample(random_state=RANDOM_SEED, frac=DATASET_SIZE)
categorical_columns = cars_dataset.select_dtypes("object").columns.tolist()

print("Dataset shape: ", cars_dataset.shape)

encoder = OneHotEncoder(sparse_output=False)
categorical_features = encoder.fit_transform(cars_dataset[categorical_columns])

imputer = SimpleImputer(strategy='median')
numerical_columns = cars_dataset.select_dtypes(include=[np.number]).columns.tolist()
cars_dataset[numerical_columns] = imputer.fit_transform(cars_dataset[numerical_columns])

poly = PolynomialFeatures(degree=3)
numerical_features = poly.fit_transform(
    cars_dataset.drop([*categorical_columns, "price"], axis=1)
)

scaler = StandardScaler()
numerical_features = scaler.fit_transform(numerical_features)

x = np.hstack((categorical_features, numerical_features))
y = np.log(cars_dataset["price"])  # use log for exponential transform

print("Features shape after transform: ", x.shape)

start = time.time()
model = RidgeCV(alphas=np.logspace(-5, 5, 100), cv=5)
validation = cross_validate(model, x, y, scoring="r2", cv=5)
print("Mean R^2: ", np.mean(validation["test_score"]))
print("Ridge time: ", time.time() - start)

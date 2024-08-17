from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

RANDOM_SEED = 14

cancers = load_breast_cancer()
x: np.ndarray = cancers.data
y: np.ndarray = cancers.target

unique, counts = np.unique(y, return_counts=True)
counts_dict = dict(zip(unique, counts))

# print(counts_dict)
# print(np.isnan(x).any())
# print(cancers.feature_names)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=RANDOM_SEED)

model = LogisticRegression(max_iter=5000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))

[print(k, v) for k, v in zip(cancers.feature_names, model.coef_[0])]

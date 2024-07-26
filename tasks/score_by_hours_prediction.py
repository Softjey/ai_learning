import numpy
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model

data_amount = 100
random_seed = 42
numpy.random.seed(random_seed)
hours = numpy.random.rand(data_amount, 1) * 10
scores: list[float] = [[hour[0] * 2.234 + numpy.random.randn() * 5] for hour in hours]

separated_data = sklearn.model_selection.train_test_split(hours, scores, test_size=0.2, random_state=random_seed)
hours_train, hours_test, scores_train, scores_test = separated_data

model = sklearn.linear_model.LinearRegression()
model.fit(hours_train, scores_train)

scores_pred = model.predict(hours_test)
abs_error = sklearn.metrics.mean_absolute_percentage_error(scores_test, scores_pred)

plt.scatter(hours, scores, color='blue', label='Data points')
plt.plot(hours_test, scores_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel(f'Hours of Study Percent error: {numpy.floor(abs_error * 100)}')
plt.ylabel('Exam Scores')
plt.legend()
plt.show()

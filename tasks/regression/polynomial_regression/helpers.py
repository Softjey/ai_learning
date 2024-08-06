import numpy as np

def generate_polynomial_data(X, coefficients):
    y = np.zeros_like(X)

    for power, coef in enumerate(reversed(coefficients)):
        y += coef * X ** power

    return y
  
def generate_percentage_noise(Y, noise_level):
    noise = Y * noise_level * np.random.randn(*Y.shape)

    return Y + noise
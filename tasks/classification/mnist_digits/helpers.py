from helpers import unpickle
import numpy as np


def get_mnist_dataset():
  dataset: dict = unpickle(__file__, './MNIST-120k')
  images: np.ndarray = dataset['data']
  target: np.ndarray = dataset['labels'].ravel()
  features = images.reshape(images.shape[0], -1)

  return features, target, images
from typing import Callable
import numpy as np


def calculate_results(models: list[dict], get_canvas_state: Callable):
  state: np.ndarray = get_canvas_state()
  state = state.reshape(-1, 784)

  return [
      {**model, "prediction": model['model'].predict(state)}
      for model in models
  ]

import tkinter as tk
from tasks.classification.mnist_digits.digits_recognizer_app.create_canvas import create_canvas
from tasks.classification.mnist_digits.digits_recognizer_app.create_results_table import create_results_table
from tasks.classification.mnist_digits.digits_recognizer_app.get_models import get_models
from tasks.classification.mnist_digits.digits_recognizer_app.set_interval import SetInterval
from tasks.classification.mnist_digits.digits_recognizer_app.calculate_results import calculate_results

root = tk.Tk()
root.title("Digits recognizer")

frame = tk.Frame(root)
frame.pack(expand=True, fill="both")

canvas, get_canvas_state = create_canvas(frame, 560, 28)
canvas.pack(side="left", expand=True, fill="both")

table = create_results_table(frame, [])
table.pack(side="right", fill="y")

models = get_models()


def get_current_table_data():
  current_data = {}
  for row_id in table.get_children():
    values = table.item(row_id, 'values')
    model_name = values[0]
    prediction = values[1]
    current_data[model_name] = prediction
  return current_data


def update_predictions():
  results = calculate_results(models, get_canvas_state)
  current_data = get_current_table_data()
  for result in results:
    model_name = result['model_name']
    new_prediction = result['prediction']

    if model_name in current_data:
      if current_data[model_name] != new_prediction:
        for row_id in table.get_children():
          values = table.item(row_id, 'values')
          if values[0] == model_name:
            table.item(row_id, values=(model_name, new_prediction))
            break
    else:
      table.insert("", "end", values=(model_name, new_prediction))


interval = SetInterval(0.1, update_predictions)

root.mainloop()

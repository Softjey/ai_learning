import tkinter as tk
from tkinter import ttk


def create_results_table(root: tk.Tk, models_results: list[dict]):
  columns = ("Model Name", "Model Prediction")
  tree = ttk.Treeview(root, columns=columns, show="headings")

  tree.heading("Model Name", text="Model Name")
  tree.heading("Model Prediction", text="Model Prediction")

  for model in models_results:
    tree.insert("", "end", values=(model['model_name'], model['prediction']))

  tree.pack(expand=True, fill="both")
  
  return tree

import tkinter as tk
import numpy as np


def create_canvas(root_window: tk.Tk, size: int, pixeled_size: int):
  pixel_size = int(size / pixeled_size)
  canvas = tk.Canvas(root_window, width=size, height=size, bg="white")
  canvas.pack()

  def get_grid_pos(x, y):
    return (x // pixel_size) * pixel_size, (y // pixel_size) * pixel_size

  def draw(event: tk.Event):
    grid_x, grid_y = get_grid_pos(event.x, event.y)
    canvas.create_rectangle(grid_x, grid_y, grid_x + pixel_size, grid_y + pixel_size, fill="black", outline="black")

  def clear_canvas():
    canvas.delete("all")

  def get_canvas_state():
    canvas_state = np.zeros((pixeled_size, pixeled_size), dtype=int)

    for item in canvas.find_all():
      coords = canvas.coords(item)
      if len(coords) == 4:
        x1, y1, x2, y2 = coords
        grid_x = int(x1 // pixel_size)
        grid_y = int(y1 // pixel_size)
        if 0 <= grid_x < pixeled_size and 0 <= grid_y < pixeled_size:
          canvas_state[grid_y, grid_x] = 255
    return canvas_state

  clear_button = tk.Button(root_window, text="Clear", command=clear_canvas)
  clear_button.pack()

  canvas.bind("<B1-Motion>", draw)
  canvas.bind("<ButtonPress-1>", draw)

  return canvas, get_canvas_state

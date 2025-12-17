import tkinter as tk
import tkinter.ttk as ttk


def MyWidget(frame):
    frame = ttk.Frame(frame, padding=10)
    frame.grid()
    _ = ttk.Label(frame, text="hello").grid(row=0)
    _ = ttk.Button(frame, text="world").grid(row=1)
    return frame


def Buttons(frame):
    frame = ttk.Frame(frame)
    frame.grid()
    _ = MyWidget(frame).grid(column=0, row=0)
    _ = MyWidget(frame).grid(column=1, row=0)
    _ = MyWidget(frame).grid(column=2, row=0)
    _ = MyWidget(frame).grid(column=3, row=0)
    _ = MyWidget(frame).grid(column=4, row=0)
    return frame


def WhichButtonPressed(frame):
    return ttk.Label(frame, text="1")


def App(frame):
    frame = ttk.Frame(frame)
    frame.grid()
    _ = Buttons(frame).grid(row=0)
    _ = WhichButtonPressed(frame).grid(row=1)
    return frame


root = tk.Tk()
_ = app(root)
root.mainloop()

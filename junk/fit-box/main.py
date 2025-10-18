import sys
import argparse
from dataclasses import dataclass

import pandas as pd
import torch
from pandas import DataFrame
from tqdm import trange
import matplotlib.pyplot as plt


@dataclass
class Conf:
    line_label: str = "Fit"
    line_color: str = "red"
    line_width: float = 2.0
    marker_size: float = 3.0
    marker_color: str = "blue"
    marker_shape: str = "o"
    x_label: str = "x"
    y_label: str = "y"
    title: str = "Data with Error Bars and Predicted Line"
    epochs: int = 10000
    lr: float = 0.1


def loss(pred, true, errors):
    """Error-aware loss function."""
    # TODO: Account for x-errors
    min_, max_ = true - errors, true + errors
    losses, _ = torch.max(
        torch.stack([(min_ - pred), (pred - max_), torch.zeros_like(pred)], dim=1),
        dim=1,
    )
    return torch.mean(losses, dim=0) + torch.mean(pred - true).mean(dim=0) / 10


def visualize(data: pd.DataFrame, model: torch.nn.Module, conf: Conf):
    # Plot data points with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(data['x'], data['y'], yerr=data['y_err'], fmt=conf.marker_shape, label='Data',
                 capsize=conf.marker_size, color=conf.marker_color)

    # Plot predicted line
    x_line = torch.linspace(data['x'].min(), data['x'].max(), 100)
    with torch.no_grad():
        y_line = model(x_line.reshape(-1, 1)).numpy()

    plt.plot(x_line, y_line, color=conf.line_color, label=conf.line_label, linewidth=conf.line_width)

    plt.xlabel(conf.x_label)
    plt.ylabel(conf.y_label)
    plt.title(conf.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def parse_conf(args) -> Conf:
    return Conf(
        epochs=args.epochs,
        title=args.title,
        x_label=args.xlabel,
        y_label=args.ylabel,
        lr=args.learning_rate,
    )


def main():
    parser = argparse.ArgumentParser(description='Fit a line to data points with error bars')
    parser.add_argument('data_file', type=str, help='Path to CSV data file')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate for optimizer')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--title', type=str, default='Data with Error Bars and Predicted Line', help='Plot title')
    parser.add_argument('--xlabel', type=str, default='x', help='X-axis label')
    parser.add_argument('--ylabel', type=str, default='y', help='Y-axis label')

    args = parser.parse_args()
    conf = parse_conf(args)

    data: DataFrame = pd.read_csv(args.data_file)
    x = torch.tensor(data["x"].values, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(data["y"].values, dtype=torch.float32).reshape(-1, 1)
    y_err = torch.tensor(data["y_err"].values, dtype=torch.float32).reshape(-1, 1)

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)

    for _ in trange(conf.epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss_value = loss(y_pred, y, y_err)
        loss_value.backward()
        optimizer.step()

    print("Trained parameters:", file=sys.stderr)
    print(f"Weight (tangent): {model.weight.item():.4f}", file=sys.stderr)
    print(f"Bias   (offset):  {model.bias.item():.4f}", file=sys.stderr)

    if not args.no_plot:
        visualize(data, model, conf)


if __name__ == "__main__":
    main()

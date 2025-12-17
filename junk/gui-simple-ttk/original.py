import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from typing import Optional, Tuple, List
from sklearn.linear_model import LinearRegression

# Define color schemes as constants
COLORS_FLAT = [
    ['#6BAED6', '#4292C6', '#2171B5'],  # Blue shades
    ['#74C476', '#41AB5D', '#238B45'],  # Green shades
    ['#FDAE6B', '#FD8D3C', '#E6550D']   # Orange shades
]

def plot_default_setup(ax: Axes, name: str, x_label: str, y_label: str) -> None:
    """Set up the plot with grid lines, labels, and other default settings.

    Args:
        ax: The matplotlib axes object to configure.
        name: The title of the plot.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
    """
    # Configure axes and grid
    ax.xaxis.set_major_locator(plt.AutoLocator())
    ax.yaxis.set_major_locator(plt.AutoLocator())
    ax.grid(True, which='major', c='black', linestyle='-', linewidth=0.6)
    ax.grid(True, which='minor', c='gray', linestyle='-', linewidth=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax.tick_params(axis='y', rotation=90)  # Rotate y-axis labels by 90 degrees
    ax.tick_params(axis='both', which='major', labelsize=6)  # Decrease font size for axis labels
    ax.tick_params(axis='both', which='minor', direction='in')  # Disable minor ticks outside the plot
    ax.axvline(x=0, color='r')
    ax.axhline(y=0, color='r')
    ax.set_title(name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def one_scatter(data: pd.DataFrame, ax: Axes, x_err_multiply: float, y_err_multiply: float, colour_err: str, colour_dots: str) -> None:
    """Plot data points with error bars.

    Args:
        data: DataFrame containing the data to plot. Expected columns are x, y, x_err, y_err.
        ax: The matplotlib axes object to plot on.
        x_err_multiply: Multiplier for x-axis error bars.
        y_err_multiply: Multiplier for y-axis error bars.
        colour_err: Color for the error bars.
        colour_dots: Color for the data points.
    """
    # Error rectangles
    for i in range(len(data)):
        x_val = float(data.iloc[i, 0])
        y_val = float(data.iloc[i, 1])
        x_err_val = float(data.iloc[i, 2]) * x_err_multiply
        y_err_val = float(data.iloc[i, 3]) * y_err_multiply
        rect = patches.Rectangle(
            (x_val - x_err_val, y_val - y_err_val),
            width=2 * x_err_val,
            height=2 * y_err_val,
            linewidth=1,
            edgecolor=colour_err,
            facecolor='none',
            alpha=0.8,
            zorder=3
        )
        ax.add_patch(rect)
    # Points on top of rectangles
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=colour_dots, s=1, alpha=1, zorder=5)

def line(ax: Axes, slope: float, intercept: float, number: str, colour: str, annotate: Optional[str], annotate_x_slide_multiply: float, annotate_y_slide_multiply: float) -> None:
    """Draw a line on the plot with optional annotation.

    Args:
        ax: The matplotlib axes object to plot on.
        slope: The slope of the line.
        intercept: The y-intercept of the line.
        number: A string to identify the line in the legend.
        colour: The color of the line.
        annotate: The text to annotate the line with. If None, no annotation is added.
        annotate_x_slide_multiply: Multiplier for the x-position of the annotation.
        annotate_y_slide_multiply: Multiplier for the y-position of the annotation.
    """
    # Calculate points for the line
    x_line = np.linspace(xmin, xmax, 100)
    y_line = slope * x_line + intercept

    # Plot the line
    ax.plot(x_line, y_line, '-.', color=colour, linewidth=1,
            label=f'y{number} = {slope:.4f} * x + {intercept:.4f}')

    # Calculate and plot the intersection point with the x-axis
    x_intersect = -intercept / slope
    ax.plot(x_intersect, 0, 'o', color=colour, markersize=4, label=f"x{number} = {x_intersect:.4f}")
    ax.legend(loc='upper left')

    if annotate is not None:
        # Calculate angle for the annotation
        yticks = ax.get_yticks()
        xticks = ax.get_xticks()
        dx = x_line[-1] - x_line[0]
        dy = y_line[-1] - y_line[0]
        k = (yticks[-1] - yticks[0]) / (xticks[-1] - xticks[0])
        angle_rad = math.atan2(dy, dx * k)
        angle_deg = math.degrees(angle_rad)

        # Position the annotation
        idx = round(len(x_line) * 0.75)
        x_pos = x_line[idx] + annotate_x_slide_multiply * dx
        y_pos = y_line[idx] + annotate_y_slide_multiply * dy

        ax.text(x_pos, y_pos, annotate, rotation=angle_deg,
                rotation_mode='anchor',
                ha='center', va='bottom',
                fontsize=6, color=colour,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def regr(data: pd.DataFrame, flag: int) -> Tuple[float, float]:
    """Perform linear regression on the data if flag is set to 1.

    Args:
        data: DataFrame containing the data to plot. Expected columns are x, y, x_err, y_err.
        flag: If 1, perform linear regression. Otherwise, return (0, 0).

    Returns:
        A tuple containing the slope and intercept of the regression line. If flag is not 1, returns (0, 0).
    """
    if flag not in (0, 1):
        print(f"Invalid flag value: {flag}")
        return 0.0, 0.0
    elif flag == 1:
        # Extract data and errors
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        x_err = data.iloc[:, 2].values
        y_err = data.iloc[:, 3].values

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(x_err) | np.isnan(y_err))
        x_clean, y_clean = x[mask], y[mask]
        x_err_clean, y_err_clean = x_err[mask], y_err[mask]

        # Perform linear regression
        x_clean_reshaped = x_clean.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_clean_reshaped, y_clean)
        slope, intercept = model.coef_[0], model.intercept_
        return slope, intercept
    else:
        return 0.0, 0.0

def workspace_setup(desktop_path: str = 'Desktop', folder_name: str = 'MMpaper') -> None:
    """Set up the workspace directory for saving plots.

    Args:
        desktop_path: The path to the desktop directory. Defaults to 'Desktop'.
        folder_name: The name of the folder to create for saving plots. Defaults to 'MMpaper'.
    """
    desktop_path = os.path.join(os.path.expanduser('~'), desktop_path)
    if not os.path.exists(desktop_path):
        os.makedirs(desktop_path)
    os.chdir(desktop_path)

    plot_dir = os.path.join(desktop_path, folder_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    os.chdir(plot_dir)

def main(
    data1: pd.DataFrame,
    data2: Optional[pd.DataFrame] = None,
    data3: Optional[pd.DataFrame] = None,
    flag: int = 0,
    xmin: float = -0.1,
    xmax: float = 1.1,
    x_err_multiply: float = 1,
    y_err_multiply: float = 1,
    name: str = "empty name",
    x_label: str = "x_label",
    y_label: str = "y_label",
    markdown: str = ""
) -> None:
    """Main function to create and save plots with optional regression lines.

    Args:
        data1: Primary DataFrame containing the data to plot.
        data2: Optional secondary DataFrame. Defaults to None.
        data3: Optional tertiary DataFrame. Defaults to None.
        flag: If 1, perform linear regression. Defaults to 0.
        xmin: Minimum value for the x-axis. Defaults to -0.1.
        xmax: Maximum value for the x-axis. Defaults to 1.1.
        x_err_multiply: Multiplier for x-axis error bars. Defaults to 1.
        y_err_multiply: Multiplier for y-axis error bars. Defaults to 1.
        name: Title of the plot. Defaults to "empty name".
        x_label: Label for the x-axis. Defaults to "x_label".
        y_label: Label for the y-axis. Defaults to "y_label".
        markdown: Additional string to append to the filename. Defaults to "".
    """
    workspace_setup()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=1000)

    # Calculate y-axis limits based on data3 if provided
    if data3 is not None:
        ymin = (data3.iloc[:, 1] - data3.iloc[:, 3]).min() - 5
        ymax = (data3.iloc[:, 1] + data3.iloc[:, 3]).max() + 10
    else:
        # Default y-axis limits if data3 is not provided
        ymin, ymax = -1, 1

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if flag == 1 and data1 is not None:
        slope, intercept = regr(data1, flag)
        # Plot regression line
        x_line = np.linspace(xmin, xmax, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'g-.', linewidth=2,
                label=f'y = {slope:.4f}x + {intercept:.4f}')
        # Calculate and plot intersection point with the x-axis
        x0 = -intercept / slope
        ax.plot(x0, 0, 'go', markersize=6, label=f"x0 = {x0:.4f}")
        ax.legend(loc='best')

    # Print information
    print(f"X limits: {xmin}, {xmax}")
    print(f"Y limits: {ymin}, {ymax}")
    if flag == 1 and data1 is not None:
        print(f"Intercept: {intercept}")
        print(f"X0: {x0}")

    plot_default_setup(ax, name, x_label, y_label)

    # Save and show plot
    filename = f"{name}{markdown}.pdf"
    plt.savefig(filename, bbox_inches='tight', dpi=1000)
    plt.show()

    # Return to desktop directory
    os.chdir(os.path.expanduser('~'))

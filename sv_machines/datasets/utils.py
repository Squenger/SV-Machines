"""Module containing utility functuions to complete dataset points."""

# Standard library imports
from typing import Callable

# Third party libraries imports
import numpy as np
import matplotlib.pyplot as plt


def add_prediction_function_to_plot(
    ax: plt.Axes,
    x_dataset: np.ndarray,
    prediction_function: Callable,
    epsilon: float,
    prediction_function_label: str,
) -> None:
    """Generates a plot of the dataset along with the prediction function.

    Parameters
    ----------
    x_dataset : np.ndarray
        X values of the dataset points.
    y_dataset : np.ndarray
        Y values of the dataset points.
    prediction_function : Callable
        A function that takes an input array and returns predicted outputs.
    plot_title : str
        Title of the plot.
    x_label : str
        Label for the X-axis.
    y_label : str
        Label for the Y-axis.
    """

    x_regression = np.linspace(min(x_dataset), max(x_dataset), 50)
    y_regression = prediction_function(x_regression)
    ax.plot(
        x_regression,
        y_regression,
        linestyle="--",
        linewidth=3,
        color="orange",
        label=prediction_function_label,
    )

    ax.plot(
        x_regression,
        y_regression + epsilon,
        color="orange",
        linestyle="--",
        linewidth=1,
        label=f"+/- espilon={epsilon} around prediction function",
    )
    ax.plot(
        x_regression,
        y_regression - epsilon,
        color="orange",
        linestyle="--",
        linewidth=1,
    )
    return None

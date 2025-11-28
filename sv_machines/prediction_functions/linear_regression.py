"""Module returning a prediction function computed using linear regression."""

# Third party libraries imports
import numpy as np


def get_y_linear_regression(x_data: np.ndarray, y_data: np.ndarray):
    """Returns a prediction function based on linear regression.

    Args:
        x_data (np.ndarray): Input feature data.
        y_data (np.ndarray): Target output data.

    Returns:
        function: A function that takes an input array and returns predicted outputs.
    """
    from sklearn.linear_model import LinearRegression

    # Reshape x_data if it's one-dimensional
    if x_data.ndim == 1:
        x_data = x_data.reshape(-1, 1)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(x_data, y_data)

    # Define the prediction function
    def prediction_function(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return model.predict(x)

    return prediction_function

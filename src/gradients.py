import numpy as np
import numpy.typing as npt

from typing import Callable

from tqdm import tqdm


def gradient_approximation(f: Callable, x: npt.NDArray[np.float32], epsilon: float = 1e-2) -> npt.NDArray[np.float32]:
    """Approximate the gradient of a scalar function f at x using finite differences."""

    # compute the gradient of f at x
    gradient = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_plus[i] = np.clip(x_plus[i], 0, 1)
        x_minus = x.copy()
        x_minus[i] -= epsilon
        x_minus[i] = np.clip(x_minus[i], 0, 1)
        actual_step_size = x_plus[i] - x_minus[i]
        gradient[i] = (f(x_plus) - f(x_minus)) / actual_step_size
    return gradient


def gradient_descent(f: Callable, x_start: npt.NDArray[np.float32], learning_rate: float = 1e-2, num_steps: int = 100, gradient_approximation_epsilon: float = 1e-2, show_progress: bool = False) -> npt.NDArray[np.float32]:
    """Perform gradient descent on a scalar function f starting at x_start."""

    x = x_start.copy()
    best_error = np.inf
    best_x = x_start.copy()
    errors = []
    errors.append(f(x))
    for _ in tqdm(range(num_steps), leave=False) if show_progress else range(num_steps):
        gradient = gradient_approximation(f, x, gradient_approximation_epsilon)
        x -= learning_rate * gradient
        x = np.clip(x, 0, 1)
        error = f(x)
        if error < best_error:
            best_error = error
            best_x = x.copy()
        errors.append(error)
    return best_x, errors


def greedy_descent(f: Callable, x_start: npt.NDArray[np.float32], num_steps: int = 100, step_size: float = 1e-2, show_progress: bool = False) -> npt.NDArray[np.float32]:
    """Perform greedy gradient descent on a scalar function f starting at x_start."""

    x = x_start.copy()
    errors = []
    errors.append(f(x))
    for _ in tqdm(range(num_steps), leave=False) if show_progress else range(num_steps):
        for i in range(x.shape[0]):
            x_plus = x.copy()
            x_plus[i] += step_size
            x_plus[i] = np.clip(x_plus[i], 0, 1)
            x_minus = x.copy()
            x_minus[i] -= step_size
            x_minus[i] = np.clip(x_minus[i], 0, 1)
            best_option = np.argmin([f(x_minus), errors[-1], f(x_plus)])
            x[i] = [x_minus[i], x[i], x_plus[i]][best_option]
            errors.append(f(x))
    return x, errors


def greedy_descent_2(f: Callable, x_start: npt.NDArray[np.float32], num_steps: int = 100, step_size: float = 1e-2, show_progress: bool = False) -> npt.NDArray[np.float32]:
    """Perform greedy gradient descent on a scalar function f starting at x_start."""

    x = x_start.copy()
    errors = []
    errors.append(f(x))
    for _ in tqdm(range(num_steps), leave=False) if show_progress else range(num_steps):
        options = [x]
        for i in range(x.shape[0]):
            x_plus = x.copy()
            x_plus[i] += step_size
            x_plus[i] = np.clip(x_plus[i], 0, 1)
            options.append(x_plus)
            x_minus = x.copy()
            x_minus[i] -= step_size
            x_minus[i] = np.clip(x_minus[i], 0, 1)
            options.append(x_minus)
        best_option = np.argmin([f(option) for option in options])
        x = options[best_option]
        errors.append(f(x))
    return x, errors

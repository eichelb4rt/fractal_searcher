from functools import partial
from multiprocessing import Pool
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


def gradient_descent(f: Callable, x_start: npt.NDArray[np.float32], learning_rate: float = 1e-2, num_steps: int = 100, gradient_approximation_epsilon: float = 1e-2, show_progress: bool = False) -> tuple[npt.NDArray[np.float32], list[float]]:
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


def gradient_descent_2(f: Callable, x_start: npt.NDArray[np.float32], initial_step_size: float, max_step_size_tries: float, step_size_decrease: float, expected_gradient_gain: float, num_steps: int = 100, gradient_approximation_epsilon: float = 1e-2, show_progress: bool = False) -> npt.NDArray[np.float32]:
    """Perform gradient descent with backtracking line search on a scalar function f starting at x_start."""

    x = x_start.copy()
    best_error = np.inf
    best_x = x_start.copy()
    errors = []
    errors.append(f(x))
    for _ in tqdm(range(num_steps), leave=False) if show_progress else range(num_steps):
        step_size = initial_step_size
        gradient = gradient_approximation(f, x, gradient_approximation_epsilon)
        walk_direction = - gradient / np.linalg.norm(gradient)
        for _ in range(max_step_size_tries):
            x_plus = x + step_size * walk_direction
            x_plus = np.clip(x_plus, 0, 1)
            error_plus = f(x_plus)
            if error_plus < errors[-1] + expected_gradient_gain * step_size * gradient @ walk_direction:
                break
            step_size *= step_size_decrease
        x = x_plus
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


def parallel_gradient_descent(f: Callable, x_starts: list[npt.NDArray[np.float32]], learning_rate: float = 1e-2, num_steps: int = 100, gradient_approximation_epsilon: float = 1e-2, show_progress: bool = False) -> tuple[list[npt.NDArray[np.float32]], list[list[float]], list[float]]:
    """Does gradient descent on multiple initial points in parallel. Returns a list with the best parameter vectors found and their respective error cources. The list is sorted with the objective function. Also returns the sorted error courses, and the errors of the best parameter vectors."""
    pool = Pool()
    gradient_descent_partial = partial(gradient_descent, f, learning_rate=learning_rate, num_steps=num_steps, gradient_approximation_epsilon=gradient_approximation_epsilon, show_progress=show_progress)
    results = pool.map(gradient_descent_partial, x_starts)
    param_vectors, error_courses = zip(*results)
    errors = [f(param_vector) for param_vector in param_vectors]
    ranks = np.argsort(errors)
    param_vectors_sorted = [param_vectors[i] for i in ranks]
    error_courses_sorted = [error_courses[i] for i in ranks]
    errors_sorted = [errors[i] for i in ranks]
    return param_vectors_sorted, error_courses_sorted, errors_sorted

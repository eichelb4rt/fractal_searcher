from rectangle import Rectangle, to_affine_function
import numpy as np
import torch
from torch import Tensor


N_HIDDEN_POINTS = 100


def generate_fractal(rectangles: list[Rectangle], seed: int = 0, num_points: int = 1000) -> Tensor:
    np.random.seed(seed)
    starting_point = torch.from_numpy(np.random.rand(2).astype(np.float32))
    selected_indices = np.random.randint(0, len(rectangles), num_points)
    affine_functions = [to_affine_function(rectangle) for rectangle in rectangles]
    points = torch.empty(num_points, 2, dtype=torch.float32)
    points[0] = starting_point
    for i, selected_index in enumerate(selected_indices):
        transformation_matrix, offset = affine_functions[selected_index]
        points[i] = points[i - 1] @ transformation_matrix.T + offset
    return points[N_HIDDEN_POINTS:]

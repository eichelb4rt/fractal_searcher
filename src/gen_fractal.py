import torch
from torch import Tensor
from tqdm import tqdm

from rectangle import Rectangle, to_affine_function

N_HIDDEN_POINTS = 100
N_POINTS_FOR_PROGRESS_BAR = 100_000


def generate_fractal(function_system: list[tuple[Tensor, Tensor]], num_points: int = 1000, starting_point: Tensor = None, selected_indices: list[int] = None, seed: int = None, show_progress: bool = None) -> Tensor:
    """Generates a list of points that form a fractal. Either provide a seed to determine the starting point and selected indices, or provide the starting point and selected indices directly. If none of these are provided, a seed of 0 is used.

    Parameters
    ----------
    function_system : list[tuple[Tensor, Tensor]]
        The iterative function system that defines the fractal.
    num_points : int, optional
        Number of points generated, by default 1000
    starting_point : Tensor, optional
        Random (maybe seeded) starting point for the procedure. Provide together with selected_indices. If you provide this, you should not provide a seed.
    selected_indices : list[int], optional
        Random (maybe seeded) list that determines at which point which affine function is applied. Provide together with starting_point. If you provide this, you should not provide a seed.
    seed : int, optional
        Seed that determines the starting_point and selected_indices if they are not provided. If you provide this, you should not provide starting_point and selected_indices.
    show_progress : bool, optional
        Show a progress bar. If not set, it is set to True if the number of points is larger than 100_000.

    Returns
    -------
    Tensor
        List of (x, y) points that form the fractal. The first N_HIDDEN_POINTS points are discarded.
    """

    # parse arguments
    if seed is not None:
        assert starting_point is None and selected_indices is None, "If you provide a seed, you should not provide starting_point and selected_indices."
    if starting_point is not None or selected_indices is not None:
        assert starting_point is not None and selected_indices is not None, "Either both starting_point and selected_indices should be provided, or neither."
    if starting_point is not None and selected_indices is not None:
        assert len(selected_indices) == num_points, "The list with selected indices should have the same length as the number of points."
    if seed is None and starting_point is None and selected_indices is None:
        seed = 0
    if show_progress is None:
        show_progress = num_points > N_POINTS_FOR_PROGRESS_BAR
    # maybe generate the starting point and selected indices (if not provided)
    if starting_point is None and selected_indices is None:
        torch.manual_seed(seed)
        starting_point = torch.rand(2).to(torch.float32)
        selected_indices = torch.randint(0, len(function_system), (num_points,))
    # generate the fractal
    points = [None] * (num_points + 1)
    points[0] = starting_point
    if show_progress:
        iterator_selected_indices = tqdm(enumerate(selected_indices))
    else:
        iterator_selected_indices = enumerate(selected_indices)
    for i, selected_index in iterator_selected_indices:
        transformation_matrix, offset = function_system[selected_index]
        points[i + 1] = points[i] @ transformation_matrix.T + offset
    return torch.stack(points[N_HIDDEN_POINTS:])

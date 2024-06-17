import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from rectangle import Rectangle, to_affine_function
from gen_fractal_interface import gen_fractal_wrapper

N_HIDDEN_POINTS = 100
N_POINTS_FOR_PROGRESS_BAR = 100_000


def generate_fractal(function_system: npt.NDArray[np.float32], num_points: int = 1000, starting_point: npt.NDArray[np.float32] = None, selected_indices: list[int] = None, seed: int = None, show_progress: bool = None) -> npt.NDArray[np.float32]:
    """Generates a list of points that form a fractal. Either provide a seed to determine the starting point and selected indices, or provide the starting point and selected indices directly. If none of these are provided, a seed of 0 is used.

    Parameters
    ----------
    function_system : list[tuple[Tensor, Tensor]]
        The iterative function system that defines the fractal. Shape: (n_rectangles * 6,).
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
    # maybe generate the starting point and selected indices (if not provided)
    if starting_point is None and selected_indices is None:
        np.random.seed(seed)
        starting_point = np.random.rand(2).astype(np.float32)
        selected_indices = np.random.randint(0, len(function_system) // 6, (num_points,)).astype(np.uint32)
    # generate the fractal
    points = gen_fractal_wrapper(function_system, num_points, starting_point, selected_indices)
    return np.clip(points[N_HIDDEN_POINTS:], 0, 1)
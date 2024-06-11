import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from compute_image_interface import compute_image
from gen_fractal import generate_fractal
from rectangle import Rectangle, to_affine_function, to_contiguous_affine_function
from draw import draw_image, draw_points
from gradients import gradient_descent


N_POINTS_PER_PIXEL = 5
DEFAULT_N_RUNS = 10
DEFAULT_NUM_STEPS = 100
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_GRADIENT_APPROXIMATION_EPSILON = 1e-1


def error(function_system: npt.NDArray[np.float32], target_image: npt.NDArray[np.float32], num_points: int, starting_point: npt.NDArray[np.float32], selected_indices: list[int]) -> float:
    # generate the fractal
    points = generate_fractal(function_system, num_points=num_points, starting_point=starting_point, selected_indices=selected_indices)
    # compute the difference to the target image
    image = compute_image(points, target_image.shape[1], target_image.shape[0])
    return np.mean((image - target_image) ** 2)


def find_fractal(target_image: npt.NDArray[np.float32], n_rectangles: int, n_runs: int = DEFAULT_N_RUNS, n_steps: int = DEFAULT_NUM_STEPS, learning_rate: float = DEFAULT_LEARNING_RATE, gradient_descent_epsilon: float = DEFAULT_GRADIENT_APPROXIMATION_EPSILON) -> list[Rectangle]:
    # half the image size until it the image consists of only 16 pixels in the smallest dimension
    image_sizes = [(target_image.shape[0] // 2**i, target_image.shape[1] // 2**i) for i in range(int(np.log2(min(target_image.shape))) - 3)][::-1]
    # make some initial guesses
    np.random.seed(0)
    x_size = 5 * n_rectangles
    top_x = np.random.uniform(0, 1, x_size).reshape((x_size)).astype(np.float32).clip(0, 1)
    for width, height in image_sizes:
        # subsample the image
        subsampled_target = subsample_image(target_image, width, height)
        # set some hyperparameters
        np.random.seed(0)
        n_points = N_POINTS_PER_PIXEL * width * height
        starting_point = np.random.rand(2)
        selected_indices = np.random.randint(0, n_rectangles, (n_points,))
        # build the objective function with that fixed seed
        objective_function = lambda x: error(x, subsampled_target, num_points=n_points, starting_point=starting_point, selected_indices=selected_indices)  # NOSONAR
        # create a few random starts for the gradient descent
        top_x, _ = gradient_descent(objective_function, top_x, learning_rate=learning_rate, num_steps=n_steps, gradient_approximation_epsilon=gradient_descent_epsilon)
    return [Rectangle(*top_x[i:i + 5]) for i in range(0, len(top_x), 5)]


def subsample_image(image: npt.NDArray[np.float32], sub_image_width: int, sub_image_height: int) -> npt.NDArray[np.float32]:
    if sub_image_width == image.shape[1] and sub_image_height == image.shape[0]:
        return image
    bin_width = image.shape[1] // sub_image_width
    bin_height = image.shape[0] // sub_image_height
    return np.array([[np.max(image[j * bin_height:(j + 1) * bin_height, i * bin_width:(i + 1) * bin_width]) for i in range(sub_image_width)] for j in range(sub_image_height)], dtype=image.dtype)


def main():
    target_file = "images/sierpinski_small.png"
    target_image = Image.open(target_file).convert('L')
    target_image = np.array(target_image).astype(np.float32) / 255
    rectangles = find_fractal(target_image, 3)
    function_system = np.stack([to_contiguous_affine_function(rectangle) for rectangle in rectangles])
    points = generate_fractal(function_system, seed=0, num_points=1_000_000)
    draw_points(points, image_width=target_image.shape[1], image_height=target_image.shape[0], file_path="images/found.png")
    # target = Image.open(target_file).convert('L')
    # target = np.array(target).astype(np.float32) / 255
    # subsampled_image = subsample_image(target, 16, 16)
    # draw_image(subsampled_image, "images/sierpinski_subsampled.png")


if __name__ == "__main__":
    main()

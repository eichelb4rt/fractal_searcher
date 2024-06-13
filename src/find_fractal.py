import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm

from compute_image_interface import compute_image
from gen_fractal import generate_fractal
from rectangle import Rectangle, to_affine_function, rectangle_to_contiguous_affine_function, vector_to_contiguous_affine_function, vectors_to_function_system
from draw import draw_image, draw_points
from gradients import gradient_descent


N_POINTS_PER_PIXEL = 5
DEFAULT_N_RUNS = 10
DEFAULT_NUM_STEPS = 100
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_GRADIENT_APPROXIMATION_EPSILON = 1e-1


def error(rectangles_param_vector: npt.NDArray[np.float32], target_image: npt.NDArray[np.float32], num_points: int, starting_point: npt.NDArray[np.float32], selected_indices: list[int]) -> float:
    # generate the fractal
    function_system = vectors_to_function_system(rectangles_param_vector)
    points = generate_fractal(function_system, num_points=num_points, starting_point=starting_point, selected_indices=selected_indices)
    # compute the difference to the target image
    image = compute_image(points, target_image.shape[1], target_image.shape[0])
    return np.mean((image - target_image) ** 2)


def find_fractal(target_image: npt.NDArray[np.float32], n_rectangles: int) -> list[Rectangle]:
    subsampled_target_image_32 = subsample_image(target_image, 32, 32)
    subsampled_target_image_64 = subsample_image(target_image, 64, 64)
    np.random.seed(0)
    starting_point = np.random.rand(2)
    selected_indices = np.random.randint(0, n_rectangles, (N_POINTS,)).astype(np.uint32)
    optimizer_1 = lambda initial_param_vectors: gradient_descent(lambda rectangles_param_vector: error(rectangles_param_vector, subsampled_target_image_32, num_points=32 * 32 * N_POINTS_PER_PIXEL, starting_point=np.random.rand(2), selected_indices=np.random.randint(0, n_rectangles, (32 * 32 * N_POINTS_PER_PIXEL,)).astype(np.uint32)), initial_param_vectors, learning_rate=DEFAULT_LEARNING_RATE, num_steps=DEFAULT_NUM_STEPS, gradient_approximation_epsilon=DEFAULT_GRADIENT_APPROXIMATION_EPSILON)


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
    function_system = np.stack([rectangle_to_contiguous_affine_function(rectangle) for rectangle in rectangles])
    points = generate_fractal(function_system, seed=0, num_points=1_000_000)
    draw_points(points, image_width=target_image.shape[1], image_height=target_image.shape[0], file_path="images/found.png")
    # target = Image.open(target_file).convert('L')
    # target = np.array(target).astype(np.float32) / 255
    # subsampled_image = subsample_image(target, 16, 16)
    # draw_image(subsampled_image, "images/sierpinski_subsampled.png")


if __name__ == "__main__":
    main()

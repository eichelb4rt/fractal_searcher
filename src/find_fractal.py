from multiprocessing import Pool
import numpy as np
import numpy.typing as npt
from PIL import Image
from tqdm import tqdm
from functools import partial

from compute_image_interface import compute_image
from gen_fractal import generate_fractal
from rectangle import Rectangle, to_affine_function, rectangle_to_contiguous_affine_function, vector_to_contiguous_affine_function, vectors_to_function_system, vectors_to_rectangles
from draw import draw_image, draw_points
from gradients import gradient_descent, parallel_gradient_descent


N_POINTS_PER_PIXEL = 5
N_PROCESSORS = 32
STD_STRAY = 1e-1
STD_STRAY_FINE = 1e-3
N_POINTS_32 = 32 * 32 * N_POINTS_PER_PIXEL
N_POINTS_64 = 64 * 64 * N_POINTS_PER_PIXEL
N_POINTS_128 = 128 * 128 * N_POINTS_PER_PIXEL

N_STEPS_32 = 2000
LEARNING_RATE_32 = 1e-1
GRADIENT_APPROXIMATION_EPSILON_32 = 1e-2

N_STEPS_32_2 = 2000
LEARNING_RATE_32_2 = 5e-2
GRADIENT_APPROXIMATION_EPSILON_32_2 = 1e-2

N_STEPS_64 = 2000
LEARNING_RATE_64 = 1e-2
GRADIENT_APPROXIMATION_EPSILON_64 = 1e-3

N_STEPS_ITERATIVE = 500
LEARNING_RATE_ITERATIVE = 1e-2
GRADIENT_APPROXIMATION_EPSILON_ITERATIVE = 1e-3

N_STEPS_FINETUNE_64 = 500
LEARNING_RATE_FINETUNE_64 = 5e-4
GRADIENT_APPROXIMATION_EPSILON_FINETUNE_64 = 1e-3

N_STEPS_FINETUNE_128 = 100
LEARNING_RATE_FINETUNE_128 = 1e-4
GRADIENT_APPROXIMATION_EPSILON_FINETUNE_128 = 1e-3


def glue_params(variable_param_vector: npt.NDArray[np.float32], variable_rectangle_index: int, fixed_param_vector: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    param_vector_index = 5 * variable_rectangle_index
    return np.concatenate([fixed_param_vector[:param_vector_index], variable_param_vector, fixed_param_vector[param_vector_index:]])


def error(rectangles_param_vector: npt.NDArray[np.float32], target_image: npt.NDArray[np.float32], num_points: int, starting_point: npt.NDArray[np.float32], selected_indices: list[int]) -> float:
    # generate the fractal
    function_system = vectors_to_function_system(rectangles_param_vector)
    points = generate_fractal(function_system, num_points=num_points, starting_point=starting_point, selected_indices=selected_indices)
    # compute the difference to the target image
    image = compute_image(points, target_image.shape[1], target_image.shape[0])
    return np.mean((image - target_image) ** 2)


def fixed_params_error(variable_param_vector: npt.NDArray[np.float32], variable_rectangle_index: int, fixed_param_vector: npt.NDArray[np.float32], target_image: npt.NDArray[np.float32], num_points: int, starting_point: npt.NDArray[np.float32], selected_indices: list[int]) -> float:
    return error(glue_params(variable_param_vector, variable_rectangle_index, fixed_param_vector), target_image, num_points, starting_point, selected_indices)


def find_fractal(target_image: npt.NDArray[np.float32], n_rectangles: int) -> list[Rectangle]:
    # subsample the target image
    subsampled_target_image_32 = subsample_image(target_image, 32, 32)
    subsampled_target_image_64 = subsample_image(target_image, 64, 64)
    subsampled_target_image_128 = subsample_image(target_image, 128, 128)
    # figure out how we generate the fractal
    np.random.seed(0)
    starting_point = np.random.rand(2)
    selected_indices_32 = np.random.randint(0, n_rectangles, (N_POINTS_32,)).astype(np.uint32)
    selected_indices_64 = np.random.randint(0, n_rectangles, (N_POINTS_64,)).astype(np.uint32)
    selected_indices_128 = np.random.randint(0, n_rectangles, (N_POINTS_128,)).astype(np.uint32)
    # create the objective functions with the target images
    objective_function_32 = partial(error, target_image=subsampled_target_image_32, num_points=N_POINTS_32, starting_point=starting_point, selected_indices=selected_indices_32)
    objective_function_64 = partial(error, target_image=subsampled_target_image_64, num_points=N_POINTS_64, starting_point=starting_point, selected_indices=selected_indices_64)
    objective_function_128 = partial(error, target_image=subsampled_target_image_128, num_points=N_POINTS_128, starting_point=starting_point, selected_indices=selected_indices_128)
    # figure out with what parameters we start the gradient descent, and what noise we add between gradient descents
    param_vector_size = 5 * n_rectangles
    initial_param_vectors_array = np.random.uniform(0, 1, N_PROCESSORS * param_vector_size).reshape((N_PROCESSORS, param_vector_size)).astype(np.float32)
    noise_32_2 = np.random.normal(0, STD_STRAY, N_PROCESSORS * param_vector_size).reshape((N_PROCESSORS, param_vector_size)).astype(np.float32)
    noise_64 = np.random.normal(0, STD_STRAY, N_PROCESSORS * param_vector_size).reshape((N_PROCESSORS, param_vector_size)).astype(np.float32)
    noise_finetune_64 = np.random.normal(0, STD_STRAY_FINE, N_PROCESSORS * param_vector_size).reshape((N_PROCESSORS, param_vector_size)).astype(np.float32)
    noise_finetune_128 = np.random.normal(0, STD_STRAY_FINE, N_PROCESSORS * param_vector_size).reshape((N_PROCESSORS, param_vector_size)).astype(np.float32)
    # prepare tqdm
    progress_bar = tqdm(total=6, leave=False)

    # do gradient descent on the subsampled 32x32 image
    progress_bar.set_description("Stage 1: 32x32")
    args_32 = [initial_param_vectors_array[i] for i in range(N_PROCESSORS)]
    result_param_vectors_32, _, _ = parallel_gradient_descent(objective_function_32, args_32, learning_rate=LEARNING_RATE_32, num_steps=N_STEPS_32, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON_32)
    progress_bar.update()

    # do it again, starting with the best parameters from the last descent + noise, and with finer learning rate
    progress_bar.set_description("Stage 2: 32x32 finer")
    best_param_vector_32 = result_param_vectors_32[0]
    initial_param_vectors_32_2 = np.clip(noise_32_2 + best_param_vector_32, 0, 1).astype(np.float32)
    args_32_2 = [initial_param_vectors_32_2[i] for i in range(N_PROCESSORS)]
    result_param_vectors_32_2, _, _ = parallel_gradient_descent(objective_function_32, args_32_2, learning_rate=LEARNING_RATE_32_2, num_steps=N_STEPS_32_2, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON_32_2)
    progress_bar.update()

    # do it again on 64x64, starting with the best parameters from the last descent + noise, and with finer learning rate and better gradient approximation
    progress_bar.set_description("Stage 3: 64x64")
    best_param_vector_32_2 = result_param_vectors_32_2[0]
    initial_param_vectors_64 = (noise_64 + best_param_vector_32_2).clip(0, 1)
    args_64 = [initial_param_vectors_64[i] for i in range(N_PROCESSORS)]
    result_param_vectors_64, _, _ = parallel_gradient_descent(objective_function_64, args_64, learning_rate=LEARNING_RATE_64, num_steps=N_STEPS_64, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON_64)
    progress_bar.update()

    # do gradient descent for each rectangle, one at a time, with all the others fixed, on 64x64
    progress_bar.set_description("Stage 4: 64x64 iteratively")
    old_top_vector = result_param_vectors_64[0]
    for optimized_rectangle_index in range(n_rectangles):
        fixed_params = np.concatenate([old_top_vector[:5 * optimized_rectangle_index], old_top_vector[5 * (optimized_rectangle_index + 1):]])
        objective_function = partial(fixed_params_error, variable_rectangle_index=optimized_rectangle_index, fixed_param_vector=fixed_params, target_image=subsampled_target_image_64, num_points=N_POINTS_64, starting_point=starting_point, selected_indices=selected_indices_64)
        initial_variable_params = np.random.uniform(0, 1, 5 * N_PROCESSORS).reshape(N_PROCESSORS, 5).astype(np.float32)
        args = [initial_variable_params[i] for i in range(N_PROCESSORS)]
        variable_param_vectors, _, _ = parallel_gradient_descent(objective_function, args, learning_rate=LEARNING_RATE_ITERATIVE, num_steps=N_STEPS_ITERATIVE, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON_ITERATIVE)
        # glue together the param vectors again
        old_top_vector = glue_params(variable_param_vectors[0], optimized_rectangle_index, fixed_params)
    progress_bar.update()

    # finetune with a short and fine gradient descent over 64x64
    progress_bar.set_description("Stage 5: 64x64 finetune")
    initial_param_vectors_finetune_64 = (noise_finetune_64 + old_top_vector).clip(0, 1)
    args_finetune_64 = [initial_param_vectors_finetune_64[i] for i in range(N_PROCESSORS)]
    result_param_vectors_finetune_64, _, _ = parallel_gradient_descent(objective_function_64, args_finetune_64, learning_rate=LEARNING_RATE_FINETUNE_64, num_steps=N_STEPS_FINETUNE_64, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON_FINETUNE_64)
    progress_bar.update()

    # # finetune with a short and fine gradient descent over 128x128
    progress_bar.set_description("Stage 6: 128x128 finetune")
    best_param_vector_finetune_64 = result_param_vectors_finetune_64[0]
    initial_param_vectors_finetune_128 = (noise_finetune_128 + best_param_vector_finetune_64).clip(0, 1)
    args_finetune_128 = [initial_param_vectors_finetune_128[i] for i in range(N_PROCESSORS)]
    result_param_vectors_finetune_128, _, _ = parallel_gradient_descent(objective_function_128, args_finetune_128, learning_rate=LEARNING_RATE_FINETUNE_128, num_steps=N_STEPS_FINETUNE_128, gradient_approximation_epsilon=GRADIENT_APPROXIMATION_EPSILON_FINETUNE_128)
    progress_bar.close()

    # return the best parameters found
    return vectors_to_rectangles(result_param_vectors_finetune_128[0])


def subsample_image(image: npt.NDArray[np.float32], sub_image_width: int, sub_image_height: int) -> npt.NDArray[np.float32]:
    if sub_image_width == image.shape[1] and sub_image_height == image.shape[0]:
        return image
    bin_width = image.shape[1] // sub_image_width
    bin_height = image.shape[0] // sub_image_height
    return np.array([[np.max(image[j * bin_height:(j + 1) * bin_height, i * bin_width:(i + 1) * bin_width]) for i in range(sub_image_width)] for j in range(sub_image_height)], dtype=image.dtype)


def main():
    target_file = "images/sierpinski.png"
    target_image = Image.open(target_file).convert('L')
    target_image = np.array(target_image).astype(np.float32) / 255
    rectangles = find_fractal(target_image, 3)
    print(rectangles)
    function_system = np.concatenate([rectangle_to_contiguous_affine_function(rectangle) for rectangle in rectangles])
    points = generate_fractal(function_system, seed=0, num_points=1_000_000)
    draw_points(points, image_width=target_image.shape[1], image_height=target_image.shape[0], file_path="images/found.png")


if __name__ == "__main__":
    main()
